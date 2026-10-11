"""Keep original evidence while sending the matching overlay to detection history."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stream_controller import run_stream
from stream_detection_capture import DetectionCapture, _write_new_atomic
from test_detection_capture import frame, target, target_frame
from test_stream_controller import SequenceFrameSource
from test_stream_integration import Display
from test_stream_processing import Processor


class AnnotatedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "detections"
        self.packet, self.record = target_frame(
            12, 1.0, [target(), target("openValve")], marker_ids=(7,)
        )
        overlay = self.packet.image.copy()
        overlay[2:9, 2:9] = (0, 255, 0)
        self.annotated = replace(self.packet, image=overlay)

    def capture(self, **options):
        capture = DetectionCapture(self.directory, stable_s=0, **options)
        self.addCleanup(capture.close)
        return capture

    def test_saves_both_images_and_one_event_without_changing_either_input(self):
        capture = self.capture()
        original_before = self.packet.image.copy()
        overlay_before = self.annotated.image.copy()
        self.packet.image.flags.writeable = False
        self.annotated.image.flags.writeable = False

        sidecar = capture(
            self.packet, self.record, annotated_packet=self.annotated
        )
        self.assertIsNotNone(sidecar)
        event = json.loads(sidecar.read_text())
        self.assertEqual(len(list(self.directory.glob("*.event.json"))), 1)
        self.assertEqual(len(list(self.directory.glob("*.png"))), 2)
        self.assertEqual(event["record"], self.record)
        self.assertEqual(len(event["target_results"]), 2)

        # Decode each file and compare it with the exact packet it should preserve.
        total_bytes = 0
        for descriptor, expected in (
            (event, original_before),
            (event["annotated_image"], overlay_before),
        ):
            path = self.directory / descriptor["image_file"]
            data = path.read_bytes()
            np.testing.assert_array_equal(cv2.imread(str(path)), expected)
            self.assertEqual(descriptor["image_bytes"], len(data))
            self.assertEqual(
                descriptor["image_sha256"], hashlib.sha256(data).hexdigest()
            )
            total_bytes += len(data)

        self.assertEqual(capture.stats["image_bytes"], total_bytes)
        np.testing.assert_array_equal(self.packet.image, original_before)
        np.testing.assert_array_equal(self.annotated.image, overlay_before)

    def test_wrong_frame_overlay_is_rejected_before_consuming_the_sighting(self):
        capture = self.capture()
        for change in (
            dict(sequence_number=13),
            dict(device_timestamp_s=99.0),
            dict(host_received_timestamp_s=99.0),
            dict(image=np.zeros((12, 16, 3), np.uint8)),
            dict(image=np.zeros((24, 32), np.uint8)),
            dict(image=self.annotated.image.astype(np.float32)),
        ):
            with self.subTest(change=list(change)):
                self.assertIsNone(
                    capture(
                        self.packet,
                        self.record,
                        annotated_packet=replace(self.annotated, **change),
                    )
                )

        self.assertEqual(list(self.directory.glob("*.png")), [])
        self.assertEqual(capture.stats["failed_events"], 6)
        self.assertIsNotNone(
            capture(self.packet, self.record, annotated_packet=self.annotated)
        )

    def test_storage_limit_counts_both_images_before_writing(self):
        original_bytes = len(cv2.imencode(".png", self.packet.image)[1])
        capture = self.capture(max_storage_bytes=original_bytes)
        self.assertIsNone(
            capture(self.packet, self.record, annotated_packet=self.annotated)
        )
        self.assertEqual(capture.stats["skipped_storage"], 1)
        self.assertEqual(capture.stats["image_bytes"], 0)
        self.assertEqual(list(self.directory.glob("*.png")), [])
        self.assertEqual(list(self.directory.glob("*.event.json")), [])

    def test_failed_overlay_write_keeps_original_and_never_exposes_ready_event(self):
        capture = self.capture()

        def write(path, data):
            if path.name.endswith(".annotated.png"):
                raise OSError("Disk write failed")
            _write_new_atomic(path, data)

        with patch("stream_detection_capture._write_new_atomic", side_effect=write):
            self.assertIsNone(
                capture(self.packet, self.record, annotated_packet=self.annotated)
            )

        original = self.directory / "frame-12.png"
        self.assertTrue(original.exists())
        self.assertEqual(capture.stats["image_bytes"], original.stat().st_size)
        self.assertEqual(capture.stats["failed_events"], 1)
        self.assertEqual(list(self.directory.glob("*.event.json")), [])

    def test_stream_pairs_original_and_composed_frame_without_running_detector_twice(self):
        packet, _ = frame(8, 1.02)
        source = SequenceFrameSource([packet])
        processor = Processor("aruco")
        received = []

        run_stream(
            source,
            Display(),
            requested_fps=30.0,
            max_frames=1,
            processors=[processor],
            on_capture=lambda original, overlay, record: received.append(
                (original, overlay, record)
            ),
        )
        self.assertEqual(len(received), 1)
        self.assertEqual(len(processor.seen), 1)
        original, overlay, record = received[0]
        self.assertIs(original, packet)
        self.assertEqual(original.sequence_number, overlay.sequence_number)
        self.assertEqual(original.device_timestamp_s, overlay.device_timestamp_s)
        self.assertEqual(
            original.host_received_timestamp_s, overlay.host_received_timestamp_s
        )
        self.assertEqual(record["sequence_number"], original.sequence_number)
        np.testing.assert_array_equal(original.image, frame(8, 1.02)[0].image)
        self.assertTrue(np.any(overlay.image != original.image))
        self.assertFalse(np.shares_memory(original.image, overlay.image))
