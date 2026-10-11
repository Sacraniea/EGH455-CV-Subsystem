"""Upload saved annotations while retaining and supporting original captures."""

import base64
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from run_gcs_image_forwarder import ImageForwarder
from test_gcs_image_forwarder import RecordingPublisher, capture


def annotated_capture(directory):
    # Save a visibly different image alongside the unchanged original capture.
    sidecar, metadata, original = capture(directory)
    pixels = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_COLOR)
    pixels[0, :] = (0, 255, 0)
    success, encoded = cv2.imencode(".png", pixels)
    assert success
    annotated = encoded.tobytes()
    image = directory / "frame-1.annotated.png"
    image.write_bytes(annotated)
    metadata["annotated_image"] = dict(
        image_file=image.name,
        image_bytes=len(annotated),
        image_sha256=hashlib.sha256(annotated).hexdigest(),
    )
    sidecar.write_text(json.dumps(metadata))
    return sidecar, metadata, original, annotated


class AnnotatedImageForwarderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def forwarder(self, directory=None, **kwargs):
        uploader = ImageForwarder(directory or self.directory, **kwargs)
        self.addCleanup(uploader.close)
        return uploader

    def test_upload_uses_saved_annotation_and_records_its_actual_file_identity(self):
        sidecar, metadata, original, annotated = annotated_capture(self.directory)
        before_sidecar = sidecar.read_bytes()
        publisher = RecordingPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )

        entry = uploader.step()

        self.assertEqual(entry["delivery"], "broker_acknowledged")
        self.assertEqual(len(publisher.payloads), 1)
        payload = publisher.payloads[0]
        self.assertEqual(
            set(payload), {"type", "description", "mime", "image", "timestamp"}
        )
        self.assertEqual(base64.b64decode(payload["image"]), annotated)
        self.assertNotEqual(annotated, original)
        self.assertEqual(payload["timestamp"], "2026-10-08T01:02:03Z")
        self.assertEqual(payload["mime"], "image/png")
        self.assertIn("Marker 7: pose unavailable", payload["description"])
        self.assertEqual(
            entry["payload_wire_bytes"],
            len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        )
        receipt = json.loads(sidecar.with_suffix(".sent.json").read_text())
        for key, value in metadata["annotated_image"].items():
            self.assertEqual(entry[key], value)
            self.assertEqual(receipt[key], value)

        self.assertEqual(sidecar.read_bytes(), before_sidecar)
        self.assertEqual((self.directory / metadata["image_file"]).read_bytes(), original)
        self.assertEqual(
            (self.directory / metadata["annotated_image"]["image_file"]).read_bytes(),
            annotated,
        )
        self.assertIsNone(uploader.step())

    def test_preview_uses_annotation_without_consuming_the_capture(self):
        sidecar, metadata, _, annotated = annotated_capture(self.directory)

        entry = self.forwarder().step()

        self.assertEqual(entry["delivery"], "preview")
        self.assertEqual(base64.b64decode(entry["payload"]["image"]), annotated)
        self.assertEqual(entry["image_file"], metadata["annotated_image"]["image_file"])
        self.assertTrue(sidecar.with_suffix(".preview.json").exists())
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        self.assertFalse(sidecar.with_suffix(".failed.json").exists())

    def test_legacy_capture_without_annotation_uploads_original_bytes(self):
        sidecar, metadata, original = capture(self.directory)
        publisher = RecordingPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )

        entry = uploader.step()

        self.assertEqual(entry["delivery"], "broker_acknowledged")
        self.assertEqual(base64.b64decode(publisher.payloads[0]["image"]), original)
        for key in ("image_file", "image_bytes", "image_sha256"):
            self.assertEqual(entry[key], metadata[key])
        self.assertEqual(json.loads(sidecar.read_text()), metadata)

    def test_invalid_annotation_fails_without_falling_back_to_original(self):
        cases = (
            "null",
            "list",
            "empty_descriptor",
            "missing_file",
            "corrupt_file",
            "wrong_size",
            "invalid_size",
            "invalid_digest",
            "invalid_image",
            "parent_path",
            "absolute_path",
            "backslash_path",
            "external_symlink",
        )

        for case in cases:
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sidecar, metadata, original, annotated = annotated_capture(directory)
                descriptor = metadata["annotated_image"]
                image = directory / descriptor["image_file"]

                if case == "null":
                    metadata["annotated_image"] = None
                elif case == "list":
                    metadata["annotated_image"] = []
                elif case == "empty_descriptor":
                    metadata["annotated_image"] = {}
                elif case == "missing_file":
                    image.unlink()
                elif case == "corrupt_file":
                    image.write_bytes(annotated[:-1] + bytes([annotated[-1] ^ 1]))
                elif case == "wrong_size":
                    descriptor["image_bytes"] += 1
                elif case == "invalid_size":
                    descriptor["image_bytes"] = True
                elif case == "invalid_digest":
                    descriptor["image_sha256"] = "not-a-digest"
                elif case == "invalid_image":
                    image.write_bytes(b"not an image")
                    descriptor.update(
                        image_bytes=12,
                        image_sha256=hashlib.sha256(b"not an image").hexdigest(),
                    )
                elif case == "parent_path":
                    descriptor["image_file"] = "../frame-1.annotated.png"
                elif case == "absolute_path":
                    descriptor["image_file"] = str(image)
                elif case == "backslash_path":
                    descriptor["image_file"] = "..\\frame-1.annotated.png"
                elif case == "external_symlink":
                    # A real file outside the capture folder must not be followed.
                    image.unlink()
                    image.symlink_to(Path(__file__).resolve())

                sidecar.write_text(json.dumps(metadata))
                before_sidecar = sidecar.read_bytes()
                publisher = RecordingPublisher()
                uploader = self.forwarder(
                    directory,
                    publish=True,
                    broker="127.0.0.1",
                    publisher_factory=lambda **kwargs: publisher,
                )

                entry = uploader.step()

                self.assertEqual(entry["delivery"], "failed")
                self.assertFalse(entry["publish_attempted"])
                self.assertEqual(publisher.payloads, [])
                self.assertTrue(sidecar.with_suffix(".failed.json").exists())
                self.assertFalse(sidecar.with_suffix(".sent.json").exists())
                self.assertEqual(sidecar.read_bytes(), before_sidecar)
                self.assertEqual((directory / metadata["image_file"]).read_bytes(), original)
                self.assertIsNone(uploader.step())

    def test_annotation_must_fit_upload_budget_even_when_original_is_small(self):
        sidecar, metadata, original, _ = annotated_capture(self.directory)
        # The original fits easily, but this annotated PNG exceeds the JSON budget.
        pixels = np.random.default_rng(5).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", pixels)
        self.assertTrue(success)
        annotated = encoded.tobytes()
        descriptor = metadata["annotated_image"]
        (self.directory / descriptor["image_file"]).write_bytes(annotated)
        descriptor.update(
            image_bytes=len(annotated),
            image_sha256=hashlib.sha256(annotated).hexdigest(),
        )
        sidecar.write_text(json.dumps(metadata))
        publisher = RecordingPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            max_payload_mb=0.001,
            publisher_factory=lambda **kwargs: publisher,
        )

        entry = uploader.step()

        self.assertEqual(entry["delivery"], "failed")
        self.assertIn("exceed", entry["error"].lower())
        self.assertFalse(entry["publish_attempted"])
        self.assertEqual(publisher.payloads, [])
        self.assertEqual((self.directory / metadata["image_file"]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
