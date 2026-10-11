"""Check image and live commands with generated markers and saved results."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_types import ArUcoConfig, CameraCalibration
from aruco_reader import read_aruco
from aruco_image import main as image_main


def marker_image(ids=(7,)):
    # Build one colour image containing the requested markers, each with a white border.
    image = np.full((300, 300 * len(ids), 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    for i, marker_id in enumerate(ids):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
        image[50:250, 50 + 300 * i : 250 + 300 * i] = cv2.cvtColor(
            marker, cv2.COLOR_GRAY2BGR
        )

    return image


def packet(image=None, sequence=42):
    # Attach fixed capture metadata to the supplied image or to a generated marker image.
    return SimpleNamespace(
        image=marker_image() if image is None else image,
        sequence_number=sequence,
        device_timestamp_s=12.5,
        host_received_timestamp_s=12.6,
    )


CONFIG = ArUcoConfig(0.2, "DICT_5X5_100", frozenset({7}))


class ReaderBoundaryTests(unittest.TestCase):
    def test_invalid_limit_rejected_even_without_detections(self):
        for limit in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                read_aruco(
                    packet(np.zeros((30, 30, 3), np.uint8)),
                    None,
                    CONFIG,
                    max_reprojection_error_px=limit,
                )

    def test_invalid_images_raise_a_clear_value_error(self):
        for image in (
            None,
            "not an image",
            np.zeros((0, 20, 3), np.uint8),
            np.zeros((20, 20, 4), np.uint8),
            np.zeros((20, 20, 3), float),
        ):
            with (
                self.subTest(shape=getattr(image, "shape", None)),
                self.assertRaises(ValueError),
            ):
                read_aruco(
                    (
                        packet(image)
                        if image is not None
                        else SimpleNamespace(image=None)
                    ),
                    None,
                    CONFIG,
                    max_reprojection_error_px=1.0,
                )

    def test_duplicate_ids_preserve_both_observations_and_withhold_poses(self):
        calibration = CameraCalibration(
            ((600.0, 0.0, 300.0), (0.0, 600.0, 150.0), (0.0, 0.0, 1.0)),
            (0.0,) * 5,
            (600, 300),
            "synthetic",
        )

        results = read_aruco(
            packet(marker_image((7, 7))),
            calibration,
            CONFIG,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(len(results), 2)

        for result in results:
            self.assertFalse(result.valid)
            self.assertIsNone(result.pose)
            self.assertEqual(result.failure_reason.value, "DUPLICATE_MARKER_ID")
            self.assertEqual(result.sequence_number, 42)

        self.assertLess(results[0].corners_px[0][0], results[1].corners_px[0][0])


class ImageCommandTests(unittest.TestCase):
    def test_marker_size_and_provenance_are_saved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "marker.png"
            cv2.imwrite(str(source), marker_image())
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    dict(
                        camera_socket="CAM_A",
                        device_mxid="test",
                        distortion_model="CameraModel.Perspective",
                        image_size_px=[300, 300],
                        camera_matrix=[[600, 0, 150], [0, 600, 150], [0, 0, 1]],
                        distortion_coefficients=[0] * 5,
                    )
                )
            )

            with contextlib.redirect_stdout(io.StringIO()):
                code = image_main(
                    [
                        str(source),
                        "--output-dir",
                        str(root / "out"),
                        "--permitted-ids",
                        "7",
                        "--marker-side-length-m",
                        ".15",
                        "--calibration",
                        str(calibration),
                    ]
                )

            self.assertEqual(code, 0)
            report = json.loads((root / "out" / "results.json").read_text())
            self.assertEqual(report["config"]["marker_side_length_m"], 0.15)
            self.assertEqual(report["image_size_px"], [300, 300])
            self.assertEqual(report["corners_coordinate_frame"], "source_px")
            self.assertGreaterEqual(report["processing_time_ms"], 0.0)
            self.assertEqual(
                report["calibration"]["source"], str(calibration.resolve())
            )
            # Compare depth with focal length times physical size, divided by marker pixels.
            self.assertAlmostEqual(
                report["results"][0]["pose"]["translation_m"][2],
                600 * 0.15 / 199,
                places=5,
            )

    def test_write_failure_returns_nonzero_and_does_not_claim_success(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "marker.png"
            cv2.imwrite(str(source), marker_image())
            stdout = io.StringIO()

            with (
                patch("aruco_image.cv2.imwrite", return_value=False),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = image_main(
                    [
                        str(source),
                        "--output-dir",
                        str(root / "out"),
                        "--permitted-ids",
                        "7",
                    ]
                )

            self.assertNotEqual(code, 0)
            self.assertNotIn("Saved", stdout.getvalue())
            self.assertFalse((root / "out" / "results.json").exists())

    def test_existing_output_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "marker.png"), marker_image())
            out = root / "out"
            out.mkdir()
            (out / "results.json").write_text("previous evidence")

            with contextlib.redirect_stderr(io.StringIO()):
                code = image_main(
                    [
                        str(root / "marker.png"),
                        "--output-dir",
                        str(out),
                        "--permitted-ids",
                        "7",
                    ]
                )

            self.assertNotEqual(code, 0)
            self.assertEqual((out / "results.json").read_text(), "previous evidence")

    def test_help_works_from_another_directory(self):
        script = Path(__file__).resolve().parents[1] / "aruco_image.py"

        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", str(script), "--help"],
                cwd=directory,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--marker-side-length-m", result.stdout)


class StreamCalibrationTests(unittest.TestCase):
    def test_explicit_oak_mode_applies_1080_resize_then_720_centre_crop(self):
        from aruco_calibration import prepare_stream_calibration

        native = CameraCalibration(
            ((2400.0, 0.0, 1920.0), (0.0, 2400.0, 1080.0), (0.0, 0.0, 1.0)),
            (0.0,) * 5,
            (3840, 2160),
            "synthetic",
        )

        c = prepare_stream_calibration(
            native,
            image_size_px=(1280, 720),
            mode="oak-1080",
        )

        # Check the halved focal lengths and the optical centre shifted by the 720p crop.
        self.assertEqual(
            c.camera_matrix,
            ((1200.0, 0.0, 640.0), (0.0, 1200.0, 360.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(c.image_size_px, (1280, 720))
        self.assertEqual(native.image_size_px, (3840, 2160))

    def test_exact_mode_never_guesses_a_transform(self):
        from aruco_calibration import prepare_stream_calibration

        native = CameraCalibration(
            ((2400.0, 0.0, 1920.0), (0.0, 2400.0, 1080.0), (0.0, 0.0, 1.0)),
            (0.0,) * 5,
            (3840, 2160),
            "synthetic",
        )
        self.assertIs(
            prepare_stream_calibration(
                native,
                image_size_px=(1280, 720),
                mode="exact",
            ),
            native,
        )

    def test_oak_mode_rejects_unrecognised_native_geometry(self):
        from aruco_calibration import prepare_stream_calibration

        native = CameraCalibration(
            ((600.0, 0.0, 200.0), (0.0, 600.0, 150.0), (0.0, 0.0, 1.0)),
            (0.0,) * 5,
            (400, 300),
            "synthetic",
        )

        with self.assertRaises(ValueError):
            prepare_stream_calibration(
                native,
                image_size_px=(1920, 1080),
                mode="oak-1080",
            )


class PresentationTests(unittest.TestCase):
    def test_pose_label_reports_range_and_rejected_record_never_exports_pose(self):
        from dataclasses import replace
        from aruco_overlay import marker_annotation_lines
        from aruco_output import result_to_record
        from aruco_types import ArUcoReadResult, MarkerPose, ArUcoFailureReason

        result = ArUcoReadResult(
            7,
            ((10.0, 10.0), (80.0, 10.0), (80.0, 80.0), (10.0, 80.0)),
            MarkerPose((0.3, 0.4, 1.2), (0.0, 0.0, 0.0), 0.5),
            None,
            42,
            12.0,
            12.1,
            0.5,
        )

        lines = "\n".join(marker_annotation_lines(result))

        self.assertIn("range=1.300 m", lines)
        self.assertIn("Z=1.200", lines)

        # Add a rejection to this existing pose and check that export and labels hide it.
        rejected = replace(
            result,
            failure_reason=ArUcoFailureReason.REPROJECTION_ERROR_TOO_HIGH,
        )
        self.assertIsNone(result_to_record(rejected)["pose"])
        self.assertNotIn("range=", "\n".join(marker_annotation_lines(rejected)))
