"""Check log delivery without opening a camera or network connection."""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import paho.mqtt.client as mqtt

import MQTT_config
import run_stream
from detection_pipeline import FrameIdentity, ProcessingResult
from stream_types import FramePacket


def targets(confirmed = True):
    return [
        {
            "class_name": "gauge",
            "bbox_xyxy_px": [10, 10, 30, 30],
            "reading_status": "confirmed" if confirmed else "pending",
            "gauge_reading": {
                "valid": True,
                "value_bar": 2.5,
                "failure_reason": None,
            },
        },
        {
            "class_name": "closedValve",
            "bbox_xyxy_px": [30, 10, 50, 30],
        },
    ]


class MQTTLogTests(unittest.TestCase):
    def run_frames(
        self,
        times,
        detections,
        *,
        connected = True,
        broker = True,
        result = None,
        targets_enabled = True,
    ):
        packets = [
            FramePacket(
                image = np.zeros((40, 60, 3), dtype = np.uint8),
                sequence_number = index,
                device_timestamp_s = 1000.0 + index,
                host_received_timestamp_s = 1000.01 + index,
            )
            for index in range(len(times))
        ]
        source = Mock()
        source.read.side_effect = [*packets, KeyboardInterrupt()]
        sender = Mock()
        client = Mock()
        if isinstance(connected, list):
            client.is_connected.side_effect = connected

        else:
            client.is_connected.return_value = connected

        if isinstance(result, Exception):
            client.publish.side_effect = result

        else:
            client.publish.return_value = SimpleNamespace(
                rc = mqtt.MQTT_ERR_SUCCESS if result is None else result,
                mid = 17,
            )

        # Supply known detections while exercising the real pipeline and formatter.
        processor = Mock()
        processor.name = "targets"
        processor.process.side_effect = lambda packet: ProcessingResult(
            processor = "targets",
            identity = FrameIdentity.from_packet(packet),
            data = {"results": detections[packet.sequence_number]},
        )
        processor.draw.side_effect = lambda overlay, result: overlay
        output = StringIO()
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        arguments = ["run_stream.py", "--source", "oak", "--no-aruco", "--no-photos", "--output-dir", folder.name]

        if broker:
            arguments.append("example.invalid")

        if not targets_enabled:
            arguments.append("--no-targets")

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch("sys.argv", arguments))
            replacements = {
                "prepare_targets": (processor, {}),
                "open_source": source,
                "create_sender": sender,
                "create_mqtt_client": client,
            }

            for name, value in replacements.items():
                replacement = stack.enter_context(
                    patch.object(run_stream, name, return_value = value)
                )

                if name == "prepare_targets" and not targets_enabled:
                    replacement.side_effect = AssertionError(
                        "Camera-only mode must not load the detection model"
                    )

            stack.enter_context(
                patch.object(run_stream, "monotonic", side_effect = times)
            )
            run_stream.main()

        # Video must still carry every processed frame and its original identity.
        expected_detections = len(packets) if targets_enabled else 0
        self.assertEqual(processor.process.call_count, expected_detections)
        self.assertEqual(sender.send.call_count, len(packets))
        records = [call.kwargs["message"] for call in sender.send.call_args_list]
        self.assertEqual(
            [record["sequence_number"] for record in records],
            list(range(len(packets))),
        )
        self.assertEqual(len({record["run_id"] for record in records}), 1)

        if not targets_enabled:
            for record in records:
                self.assertEqual(record["processors"], [])

        for packet in packets:
            self.assertFalse(packet.image.any())

        source.close.assert_called_once()
        sender.close.assert_called_once()

        if broker:
            client.disconnect.assert_called_once()
            client.loop_stop.assert_called_once()

        return client, output.getvalue()

    def test_sends_each_target_only_from_five_second_sampled_frames(self):
        client, _ = self.run_frames(
            [100.0, 104.99, 105.0],
            [targets(), targets(), targets(False)],
        )
        messages = [
            json.loads(call.kwargs["payload"])
            for call in client.publish.call_args_list
        ]
        self.assertEqual(messages, [
            {
                "action": "Gauge detected",
                "content": "Pressure: 2.5 bar; image position (20, 20) px; Frame 0.",
            },
            {
                "action": "Valve detected",
                "content": "Closed valve; image position (40, 20) px; Frame 0.",
            },
            {
                "action": "Gauge detected",
                "content": (
                    "Pressure unavailable (awaiting confirmation); "
                    "image position (20, 20) px; Frame 2."
                ),
            },
            {
                "action": "Valve detected",
                "content": "Closed valve; image position (40, 20) px; Frame 2.",
            },
        ])
        for call in client.publish.call_args_list:
            self.assertEqual(call.kwargs["topic"], "gcs/drone_log")
            self.assertEqual(call.kwargs["qos"], 1)
            self.assertIs(call.kwargs["retain"], False)

    def test_empty_frame_does_not_delay_the_first_detection_log(self):
        client, _ = self.run_frames([100.0, 100.1], [[], targets()])
        self.assertEqual(client.publish.call_count, 2)
        self.assertIn("Frame 1.", client.publish.call_args.kwargs["payload"])

    def test_disconnected_frames_are_skipped_until_connection_returns(self):
        client, _ = self.run_frames(
            [100.0, 100.1],
            [targets(), targets()],
            connected = [False, True],
        )
        self.assertEqual(client.publish.call_count, 2)
        self.assertIn("Frame 1.", client.publish.call_args.kwargs["payload"])

    def test_video_only_mode_never_publishes(self):
        client, _ = self.run_frames([100.0], [targets()], broker = False)
        client.publish.assert_not_called()

    def test_no_targets_keeps_video_without_loading_model_or_sending_detections(self):
        try:
            client, _ = self.run_frames(
                [100.0, 106.0],
                [targets(), targets()],
                targets_enabled = False,
            )

        except SystemExit as error:
            self.fail(f"The sender rejected --no-targets: {error}")

        client.publish.assert_not_called()

    def test_delivery_failures_leave_video_running_without_per_frame_retries(self):
        for result, expected in (
            (mqtt.MQTT_ERR_QUEUE_SIZE, "not queued"),
            (mqtt.MQTT_ERR_NO_CONN, "awaiting reconnection"),
            (OSError("connection interrupted"), "unconfirmed"),
        ):
            with self.subTest(result = result):
                client, output = self.run_frames(
                    [100.0, 100.1, 100.2],
                    [targets(), targets(), targets()],
                    result = result,
                )
                self.assertEqual(client.publish.call_count, 2)
                self.assertIn(expected, output)
                self.assertNotIn("broker acknowledged", output)

    def test_acknowledgement_is_reported_only_when_callback_runs(self):
        client = Mock()
        client.publish.return_value = SimpleNamespace(
            rc = mqtt.MQTT_ERR_SUCCESS,
            mid = 42,
        )
        output = StringIO()
        self.assertTrue(callable(getattr(MQTT_config, "publish_log", None)))

        with redirect_stdout(output):
            MQTT_config.publish_log(
                client,
                {"action": "Valve detected", "content": "Closed valve"},
            )
        self.assertNotIn("broker acknowledged", output.getvalue())

        with patch.object(MQTT_config.mqtt, "Client", return_value = client):
            client.loop_start.return_value = mqtt.MQTT_ERR_SUCCESS
            MQTT_config.create_mqtt_client("example.invalid")
        with redirect_stdout(output):
            client.on_publish(
                client,
                None,
                42,
                SimpleNamespace(is_failure = False),
                None,
            )
        self.assertIn("broker acknowledged message 42", output.getvalue())


if __name__ == "__main__":
    unittest.main()
