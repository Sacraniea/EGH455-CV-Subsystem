import base64
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tests.test_stream_mqtt import accept_client, broker_peer, read_packet


class SendGcsDetectionTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("send_gcs_detection"),
            "GCS image command has not been implemented",
        )
        import send_gcs_detection

        return send_gcs_detection

    def image(self, directory):
        # Write a PNG fixture and return its path for the image-byte checks.
        path = Path(directory) / "original.png"
        self.assertTrue(cv2.imwrite(str(path), np.full((13, 17, 3), 128, np.uint8)))
        return path

    def test_preview_preserves_file_bytes_without_network_connection(self):
        module = self.module()

        with TemporaryDirectory() as tmp:
            path = self.image(tmp)
            original = path.read_bytes()
            output = Path(tmp) / "payload.json"

            with patch(
                "send_gcs_detection.MqttResultPublisher",
                side_effect=AssertionError("network opened"),
            ):
                code = module.main(
                    [
                        str(path),
                        "--type",
                        "Valve",
                        "--description",
                        "Test image",
                        "--output",
                        str(output),
                    ],
                    printer=lambda *a, **k: None,
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.read_text())
            # Decode the text wrapper and compare the exact original file bytes.
            self.assertEqual(base64.b64decode(payload["image"]), original)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                set(payload), {"type", "description", "mime", "image", "timestamp"}
            )

    def test_explicit_publish_sends_original_image_on_detection_topic(self):
        module = self.module()
        received = []

        def handler(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            # Extract the topic and image JSON, then acknowledge the publish message ID.
            length = int.from_bytes(body[:2], "big")
            received.append(
                (header, body[2 : 2 + length].decode(), json.loads(body[4 + length :]))
            )
            connection.sendall(b"\x40\x02" + body[2 + length : 4 + length])
            read_packet(connection)

        with TemporaryDirectory() as tmp, broker_peer(handler) as port:
            path = self.image(tmp)
            code = module.main(
                [
                    str(path),
                    "--type",
                    "Valve",
                    "--description",
                    "Test image",
                    "--publish",
                    "--broker",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                printer=lambda *a, **k: None,
            )
            self.assertEqual(code, 0)
            self.assertEqual(received[0][:2], (0x32, "gcs/detection"))
            self.assertEqual(
                base64.b64decode(received[0][2]["image"]), path.read_bytes()
            )

    def test_missing_image_fails_before_opening_broker(self):
        module = self.module()

        with TemporaryDirectory() as tmp, patch(
            "send_gcs_detection.MqttResultPublisher",
            side_effect=AssertionError("network opened"),
        ):
            code = module.main(
                [
                    str(Path(tmp) / "missing.png"),
                    "--type",
                    "Valve",
                    "--description",
                    "Test",
                    "--publish",
                    "--broker",
                    "127.0.0.1",
                ],
                printer=lambda *a, **k: None,
            )
            self.assertEqual(code, 1)
