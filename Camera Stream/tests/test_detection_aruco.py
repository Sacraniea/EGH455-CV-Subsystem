"""Exercise ArUco processing with real generated markers and no camera."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from detection_aruco import add_processing_arguments, prepare_aruco
from detection_pipeline import FrameIdentity, ResultCompositor, process_frame
from stream_types import FramePacket, get_stream_profile
from synthetic_source import SyntheticFrameSource


def processing_arguments(*options):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default = "oak")
    parser.add_argument("--calibration-mode", default = "exact")
    add_processing_arguments(parser)
    return parser.parse_args(options)


def marker_packet(marker_id):
    image = np.full((480, 640, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
    image[160:320, 240:400] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return FramePacket(image, 42, 10.0, 10.02)


class ArUcoProcessingTests(unittest.TestCase):
    def test_default_ids_include_dictionary_boundaries_without_inventing_pose(self):
        processor, metadata = prepare_aruco(
            processing_arguments(),
            get_stream_profile("480p20"),
        )
        self.assertEqual(metadata["permitted_ids"], list(range(100)))
        self.assertIsNone(metadata["calibration"])

        for marker_id in (0, 23, 99):
            with self.subTest(marker_id = marker_id):
                packet = marker_packet(marker_id)
                result, = process_frame(packet, (processor,))
                marker, = result.data["results"]

                self.assertIsNone(result.failure_reason)
                self.assertEqual(result.identity, FrameIdentity.from_packet(packet))
                self.assertEqual(marker["marker_id"], marker_id)
                self.assertEqual(marker["sequence_number"], 42)
                self.assertEqual(marker["failure_reason"], "MISSING_CALIBRATION")
                self.assertFalse(marker["valid"])
                self.assertIsNone(marker["pose"])
                self.assertEqual(len(marker["corners_px"]), 4)

    def test_explicit_id_selection_withholds_unpermitted_marker_pose(self):
        processor, metadata = prepare_aruco(
            processing_arguments("--permitted-ids", "7", "7"),
            get_stream_profile("480p20"),
        )
        result, = process_frame(marker_packet(23), (processor,))
        marker, = result.data["results"]

        self.assertEqual(metadata["permitted_ids"], [7])
        self.assertEqual(marker["marker_id"], 23)
        self.assertEqual(marker["failure_reason"], "MARKER_ID_NOT_ALLOWED")
        self.assertIsNone(marker["pose"])

    def test_processing_and_drawing_preserve_source_pixels_and_identity(self):
        processor, _ = prepare_aruco(
            processing_arguments(),
            get_stream_profile("480p20"),
        )
        packet = marker_packet(23)
        original = packet.image.copy()
        results = process_frame(packet, (processor,))
        display = ResultCompositor((processor,)).compose(packet, results)

        np.testing.assert_array_equal(packet.image, original)
        self.assertTrue(packet.image.flags.writeable)
        self.assertFalse(np.shares_memory(display.image, packet.image))
        self.assertTrue(np.any(display.image != original))
        self.assertEqual(FrameIdentity.from_packet(display), FrameIdentity.from_packet(packet))

    def test_synthetic_pose_requires_explicit_generated_calibration(self):
        profile = get_stream_profile("480p20")
        source = SyntheticFrameSource(profile)
        self.addCleanup(source.close)
        processor, metadata = prepare_aruco(
            processing_arguments("--source", "synthetic"),
            profile,
        )
        first = source.read()
        uncalibrated, = process_frame(first, (processor,))
        marker, = uncalibrated.data["results"]
        self.assertEqual(marker["failure_reason"], "MISSING_CALIBRATION")
        self.assertIsNone(metadata["calibration"])

        # Only this explicit synthetic source supplies invented intrinsics.
        processor.calibration = source.calibration
        calibrated, = process_frame(first, (processor,))
        marker, = calibrated.data["results"]
        self.assertEqual(marker["marker_id"], 7)
        self.assertTrue(marker["valid"])
        self.assertIsNone(marker["failure_reason"])
        self.assertGreater(marker["pose"]["translation_m"][2], 0.0)
        self.assertEqual(marker["pose"]["coordinate_frame"], "camera_optical")
        self.assertIn("invented", calibrated.data["calibration"]["source"])

        second = source.read()
        self.assertEqual(second.sequence_number, first.sequence_number + 1)
        self.assertGreater(second.device_timestamp_s, first.device_timestamp_s)
        self.assertFalse(np.shares_memory(first.image, second.image))
        source.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            source.read()

    def test_mismatched_calibration_size_preserves_detection_without_pose(self):
        profile = get_stream_profile("480p20")
        source = SyntheticFrameSource(profile)
        self.addCleanup(source.close)
        processor, _ = prepare_aruco(
            processing_arguments("--source", "synthetic"),
            profile,
        )
        processor.calibration = replace(
            source.calibration,
            image_size_px = (1280, 720),
        )
        result, = process_frame(source.read(), (processor,))
        marker, = result.data["results"]

        self.assertEqual(marker["marker_id"], 7)
        self.assertEqual(marker["failure_reason"], "CALIBRATION_SIZE_MISMATCH")
        self.assertIsNone(marker["pose"])

    def test_camera_calibration_options_are_rejected_for_synthetic(self):
        for options in (
            ("--calibration", "unused-oak.json"),
            ("--calibration-mode", "oak-1080"),
            ("--calibration", "unused-oak.json", "--calibration-mode", "oak-1080"),
        ):
            with self.subTest(options = options):
                args = processing_arguments("--source", "synthetic", *options)
                with self.assertRaisesRegex(ValueError, "synthetic"):
                    prepare_aruco(args, get_stream_profile("720p30"))

    def test_oak_calibration_uses_shared_geometry_and_retains_file_evidence(self):
        record = {
            "camera_socket": "CAM_A",
            "device_mxid": "test-device",
            "distortion_model": "CameraModel.Perspective",
            "image_size_px": [3840, 2160],
            "camera_matrix": [
                [2400.0, 0.0, 1920.0],
                [0.0, 2400.0, 1080.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.0] * 5,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            raw = json.dumps(record).encode("utf-8")
            path.write_bytes(raw)
            processor, metadata = prepare_aruco(
                processing_arguments(
                    "--source", "oak",
                    "--calibration", str(path),
                    "--calibration-mode", "oak-1080",
                ),
                get_stream_profile("720p30"),
            )

            self.assertEqual(processor.calibration.image_size_px, (1280, 720))
            self.assertEqual(processor.calibration.camera_matrix, (
                (1200.0, 0.0, 640.0),
                (0.0, 1200.0, 360.0),
                (0.0, 0.0, 1.0),
            ))
            evidence = metadata["calibration"]
            self.assertEqual(evidence["transform"], "oak-1080")
            self.assertEqual(evidence["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(evidence["original_record"], record)
            self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
