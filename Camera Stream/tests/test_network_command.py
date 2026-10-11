import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from run_camera_stream import main


class NetworkCommandTests(unittest.TestCase):
    def test_synthetic_trial_does_not_open_hardware_and_keeps_evidence_labelled(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            # Fail immediately if synthetic mode tries to open the physical camera.
            with patch(
                "run_camera_stream.DepthAIFrameSource",
                side_effect=AssertionError("opened hardware"),
            ):
                code = main(
                    [
                        "--synthetic",
                        "--aruco",
                        "--permitted-ids",
                        "7",
                        "--headless",
                        "--max-frames",
                        "2",
                        "--output-dir",
                        str(output),
                    ],
                    printer=lambda _: None,
                )

            self.assertEqual(code, 0)
            records = [
                json.loads(line)
                for line in (output / "frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["source_kind"], "synthetic")
            self.assertTrue(records[0]["processors"][0]["data"]["results"][0]["valid"])
            self.assertIn(
                "synthetic-demo",
                records[0]["processors"][0]["data"]["calibration"]["source"],
            )
            manifest = json.loads((output / "run.json").read_text())
            self.assertIn(
                "Synthetic image generation",
                manifest["metadata"]["measurement"]["latency"],
            )
            acquisition = json.loads(
                next((output / "acquisition").glob("*.json")).read_text()
            )
            self.assertEqual(acquisition["source_kind"], "synthetic")
            self.assertEqual(
                acquisition["measurement"]["latency_boundary"],
                "synthetic_image_generation_start_to_finish",
            )

    def test_mqtt_and_video_receive_same_frame_identity(self):
        # Collect MQTT records and video identities from the same one-frame command run.
        messages = []
        video = []
        closed = []

        class Mqtt:
            def __init__(self, **kw):
                pass

            def publish(self, record):
                messages.append(dict(record))

            def close(self):
                closed.append("mqtt")

        class Video:
            def __init__(self, **kw):
                pass

            def send(self, packet, *, run_id, source_kind):
                video.append((packet.sequence_number, run_id, source_kind))

            def close(self):
                closed.append("video")

        with TemporaryDirectory() as tmp, patch(
            "stream_mqtt.MqttResultPublisher", Mqtt
        ), patch("stream_video.NetGearVideoSender", Video):
            main(
                [
                    "--synthetic",
                    "--headless",
                    "--max-frames",
                    "1",
                    "--mqtt",
                    "--video",
                    "--output-dir",
                    str(Path(tmp) / "run"),
                ],
                printer=lambda _: None,
            )

        self.assertEqual(
            video[0],
            (messages[0]["sequence_number"], messages[0]["run_id"], "synthetic"),
        )
        self.assertEqual(sorted(closed), ["mqtt", "video"])
