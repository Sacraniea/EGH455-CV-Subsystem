"""Check that one sampled frame can deliver all of its targets to GCS."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from run_gcs_forwarder import LogSampler, main
from tests.test_gcs_forwarder import RECORD, source_handler
from tests.test_stream_mqtt import accept_client, broker_peer, read_packet


def target_frame(*, include_aruco=False):
    # Use the same result fields that TargetDetectionProcessor puts in a frame.
    record = copy.deepcopy(RECORD)

    if not include_aruco:
        record["processors"] = []

    record["processors"].append({
        "processor": "targets",
        "failure_reason": None,
        "data": {
            "results": [
                {
                    "class_name": "gauge",
                    "bbox_xyxy_px": [20, 40, 80, 100],
                    "gauge_reading": {
                        "valid": True,
                        "value_bar": 2.9,
                        "failure_reason": None,
                    },
                },
                {
                    "class_name": "gauge",
                    "bbox_xyxy_px": [100, 40, 160, 100],
                    "gauge_reading": {
                        "valid": False,
                        "value_bar": None,
                        "failure_reason": "DIAL_TOO_SMALL",
                    },
                },
                {
                    "class_name": "openValve",
                    "bbox_xyxy_px": [180, 40, 240, 100],
                    "gauge_reading": None,
                },
            ],
        },
    })
    return record


class TargetLogForwarderTests(unittest.TestCase):
    def test_sampling_keeps_all_targets_and_aruco_from_the_selected_frame(self):
        now = [0.0]
        sampler = LogSampler(interval_s=5.0, clock=lambda: now[0])
        record = target_frame(include_aruco=True)
        selected = sampler.select(record)
        self.assertIsInstance(selected, list)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [payload["action"] for payload in selected],
            [
                "ArUco detected - pose unavailable",
                "Gauge detected",
                "Gauge detected",
                "Valve detected",
            ],
        )

        # A later frame inside the interval must not create another batch of logs.
        now[0] = 4.0
        record["sequence_number"] = 13
        self.assertIsNone(sampler.select(record))
        now[0] = 5.0
        record["sequence_number"] = 14
        self.assertEqual(len(sampler.select(record)), 4)

        # Receiving that frame again must not duplicate its messages.
        now[0] = 10.0
        self.assertIsNone(sampler.select(record))

    def test_empty_target_frame_does_not_delay_the_next_detection(self):
        sampler = LogSampler(interval_s=5.0, clock=lambda: 0.0)
        empty = target_frame()
        empty["sequence_number"] = 11
        empty["processors"][0]["data"]["results"] = []
        self.assertIsNone(sampler.select(empty))
        self.assertEqual(len(sampler.select(target_frame())), 3)

    def test_target_only_preview_saves_each_message_without_opening_gcs(self):
        record = target_frame()

        with TemporaryDirectory() as tmp, broker_peer(source_handler(record)) as port:
            output = Path(tmp) / "preview.jsonl"

            with patch(
                "run_gcs_forwarder.MqttResultPublisher",
                side_effect=AssertionError("Preview opened a GCS connection"),
            ):
                code = main(
                    [
                        "--local-port", str(port),
                        "--max-frames", "1",
                        "--timeout-s", "3",
                        "--output", str(output),
                    ],
                    printer=lambda *a, **k: None,
                )

            self.assertEqual(code, 0)
            entries = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(entries), 3)
            self.assertIn("Pressure: 2.9 bar", entries[0]["payload"]["content"])
            self.assertIn("Pressure unavailable", entries[1]["payload"]["content"])
            self.assertIn("Open valve", entries[2]["payload"]["content"])

            for entry in entries:
                self.assertEqual(entry["delivery"], "preview")
                self.assertEqual(entry["run_id"], record["run_id"])
                self.assertEqual(entry["sequence_number"], 12)

    def test_mixed_frame_delivers_every_message_to_separate_local_mqtt_broker(self):
        record = target_frame(include_aruco=True)
        received = []

        def destination(connection):
            accept_client(connection)

            # Acknowledge each message separately, as the GCS broker would do.
            while True:
                header, body = read_packet(connection)

                if header == 0xE0:
                    return

                self.assertEqual(header, 0x32)
                length = int.from_bytes(body[:2], "big")
                topic = body[2 : 2 + length].decode()
                payload = json.loads(body[4 + length :])
                received.append((topic, payload))
                connection.sendall(b"\x40\x02" + body[2 + length : 4 + length])

        with TemporaryDirectory() as tmp, broker_peer(
            source_handler(record)
        ) as source_port, broker_peer(destination) as destination_port:
            output = Path(tmp) / "sent.jsonl"
            code = main(
                [
                    "--publish",
                    "--broker", "127.0.0.1",
                    "--port", str(destination_port),
                    "--local-port", str(source_port),
                    "--max-frames", "1",
                    "--timeout-s", "3",
                    "--output", str(output),
                ],
                printer=lambda *a, **k: None,
            )
            self.assertEqual(code, 0)
            entries = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(entries), 4)
            self.assertEqual(
                received,
                [("gcs/drone_log", entry["payload"]) for entry in entries],
            )

            for entry in entries:
                self.assertEqual(entry["delivery"], "broker_acknowledged")
                self.assertEqual(set(entry["payload"]), {"action", "content"})
                self.assertEqual(entry["run_id"], record["run_id"])
                self.assertEqual(entry["sequence_number"], 12)

    def test_failed_second_message_keeps_acknowledged_failed_and_unsent_outcomes(self):
        def receive_one_frame(subscription, *, on_record, printer):
            on_record(target_frame())
            return 1

        with TemporaryDirectory() as tmp, patch(
            "run_gcs_forwarder.receive", side_effect=receive_one_frame
        ), patch("run_gcs_forwarder.MqttResultPublisher") as publisher_type:
            output = Path(tmp) / "failed.jsonl"
            publisher = publisher_type.return_value
            publisher.publish.side_effect = [None, RuntimeError("Broker disconnected")]
            code = main(
                [
                    "--publish", "--broker", "127.0.0.1",
                    "--output", str(output),
                ],
                printer=lambda *a, **k: None,
            )
            self.assertEqual(code, 1)
            entries = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [entry["delivery"] for entry in entries],
                ["broker_acknowledged", "failed", "not_attempted"],
            )
            self.assertIn("Broker disconnected", entries[1]["error"])
            self.assertEqual(entries[2]["payload"]["action"], "Valve detected")
            self.assertIn("Earlier message", entries[2]["error"])
            self.assertEqual(publisher.publish.call_count, 2)
            publisher.close.assert_called_once()

    def test_interrupted_send_is_not_misreported_as_a_preview(self):
        def receive_one_frame(subscription, *, on_record, printer):
            on_record(target_frame())
            return 1

        with TemporaryDirectory() as tmp, patch(
            "run_gcs_forwarder.receive", side_effect=receive_one_frame
        ), patch("run_gcs_forwarder.MqttResultPublisher") as publisher_type:
            output = Path(tmp) / "interrupted.jsonl"
            publisher = publisher_type.return_value
            publisher.publish.side_effect = [None, KeyboardInterrupt()]
            code = main(
                [
                    "--publish", "--broker", "127.0.0.1",
                    "--output", str(output),
                ],
                printer=lambda *a, **k: None,
            )
            self.assertEqual(code, 130)
            entries = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [entry["delivery"] for entry in entries],
                ["broker_acknowledged", "failed", "not_attempted"],
            )
            self.assertIn("KeyboardInterrupt", entries[1]["error"])
            self.assertEqual(publisher.publish.call_count, 2)
            publisher.close.assert_called_once()
