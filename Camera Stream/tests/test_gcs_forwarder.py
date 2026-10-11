import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.test_stream_mqtt import accept_client, broker_peer, packet, read_packet

RECORD = {
    "producer": "camera_stream",
    "run_id": "run-gcs-test",
    "source_kind": "camera",
    "sequence_number": 12,
    "device_timestamp_s": 20.0,
    "host_received_timestamp_s": 20.05,
    "image_size_px": [1920, 1080],
    "processors": [
        {
            "processor": "aruco",
            "failure_reason": None,
            "data": {
                "results": [
                    {
                        "marker_id": 7,
                        "corners_px": [[1, 2], [3, 2], [3, 4], [1, 4]],
                        "valid": False,
                        "pose": None,
                        "failure_reason": "MISSING_CALIBRATION",
                        "sequence_number": 12,
                        "device_timestamp_s": 20.0,
                        "host_received_timestamp_s": 20.05,
                        "reprojection_error_px": None,
                    }
                ]
            },
        }
    ],
}


def source_handler(record):
    # Return a broker handler that sends the supplied frame after accepting a subscription.
    def handle(connection):
        accept_client(connection)
        header, body = read_packet(connection)

        if header != 0x82:
            raise AssertionError("Expected subscription")

        connection.sendall(b"\x90\x03" + body[:2] + b"\x01")
        # Follow the subscription acknowledgement with a QoS 1 frame message.
        topic = b"taip/local/results"
        connection.sendall(
            packet(
                0x32,
                len(topic).to_bytes(2, "big")
                + topic
                + b"\x00\x01"
                + json.dumps(record).encode(),
            )
        )

        # Consume the publish acknowledgement and wait for the subscriber to disconnect.
        while read_packet(connection)[0] != 0xE0:
            pass

    return handle


class GcsForwarderTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("run_gcs_forwarder"),
            "GCS forwarding command has not been implemented",
        )
        import run_gcs_forwarder

        return run_gcs_forwarder

    def test_repeated_frames_are_sampled_and_qos_duplicates_are_not_relogged(self):
        module = self.module()
        now = [0.0]
        sampler = module.LogSampler(interval_s=5.0, clock=lambda: now[0])
        self.assertIsNotNone(sampler.select(RECORD))
        # Submit the same frame after the interval; it must still be skipped as a duplicate.
        now[0] = 10.0
        self.assertIsNone(sampler.select(RECORD))
        second = copy.deepcopy(RECORD)
        second["sequence_number"] = 13
        second["processors"][0]["data"]["results"][0]["sequence_number"] = 13
        self.assertIsNotNone(sampler.select(second))
        now[0] = 11.0
        third = copy.deepcopy(second)
        third["sequence_number"] = 14
        third["processors"][0]["data"]["results"][0]["sequence_number"] = 14
        self.assertIsNone(sampler.select(third))

    def test_empty_frame_does_not_consume_the_log_interval(self):
        module = self.module()
        sampler = module.LogSampler(interval_s=5.0, clock=lambda: 0.0)
        empty = copy.deepcopy(RECORD)
        empty["sequence_number"] = 11
        empty["processors"][0]["data"]["results"] = []
        self.assertIsNone(sampler.select(empty))
        self.assertIsNotNone(sampler.select(RECORD))

    def test_dry_run_subscribes_and_saves_payload_without_opening_gcs_connection(self):
        module = self.module()

        with TemporaryDirectory() as tmp, broker_peer(source_handler(RECORD)) as port:
            output = Path(tmp) / "preview.jsonl"

            with patch(
                "run_gcs_forwarder.MqttResultPublisher",
                side_effect=AssertionError("GCS connection opened"),
            ):
                code = module.main(
                    [
                        "--local-port",
                        str(port),
                        "--max-frames",
                        "1",
                        "--timeout-s",
                        "3",
                        "--output",
                        str(output),
                    ],
                    printer=lambda *a, **k: None,
                )

            self.assertEqual(code, 0)
            audit = json.loads(output.read_text())
            self.assertEqual(audit["delivery"], "preview")
            self.assertEqual(audit["topic"], "gcs/drone_log")
            self.assertEqual(set(audit["payload"]), {"action", "content"})
            self.assertEqual(
                audit["payload"]["action"], "ArUco detected - pose unavailable"
            )
            self.assertIn(
                "Marker 7: pose unavailable (calibration missing)",
                audit["payload"]["content"],
            )
            self.assertIn(
                "Frame 12 | Device time 20.000 s | Run run-gcs-test",
                audit["payload"]["content"],
            )

    def test_publishes_exact_gcs_schema_to_a_separate_broker_and_records_ack(self):
        module = self.module()
        received = []

        # Receive forwarded logs on a second broker, separate from the frame source.
        def destination(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            # Separate the topic and JSON body, skipping the two-byte publish message ID.
            length = int.from_bytes(body[:2], "big")
            received.append(
                (header, body[2 : 2 + length].decode(), json.loads(body[4 + length :]))
            )
            # Acknowledge this message ID so the forwarder can record broker acceptance.
            connection.sendall(b"\x40\x02" + body[2 + length : 4 + length])
            read_packet(connection)

        with TemporaryDirectory() as tmp, broker_peer(
            source_handler(RECORD)
        ) as source_port, broker_peer(destination) as dest_port:
            output = Path(tmp) / "sent.jsonl"
            code = module.main(
                [
                    "--publish",
                    "--broker",
                    "127.0.0.1",
                    "--port",
                    str(dest_port),
                    "--local-port",
                    str(source_port),
                    "--max-frames",
                    "1",
                    "--timeout-s",
                    "3",
                    "--output",
                    str(output),
                ],
                printer=lambda *a, **k: None,
            )
            self.assertEqual(code, 0)
            audit = json.loads(output.read_text())
            self.assertEqual(audit["delivery"], "broker_acknowledged")
            self.assertEqual(received, [(0x32, "gcs/drone_log", audit["payload"])])

    def test_invalid_interval_is_rejected(self):
        module = self.module()

        for interval in (0, -1, float("inf"), float("nan")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                module.LogSampler(interval_s=interval)
