from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import aruco_calibration
from aruco_calibration import crop_calibration, load_calibration, resize_calibration
from aruco_types import CameraCalibration


class CalibrationResizeTests(unittest.TestCase):
    def test_resizes_intrinsics_from_1080p_to_720p(self):
        # Resize a known 1080p calibration and check that its pixel values scale by 2/3.
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        resized = resize_calibration(
            calibration,
            image_size_px=(1280, 720),
        )

        self.assertEqual(
            resized.camera_matrix,
            (
                (800.0, 0.0, 640.0),
                (0.0, 800.0, 360.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.assertEqual(resized.image_size_px, (1280, 720))

    def test_rejects_aspect_ratio_change(self):
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        with self.assertRaises(ValueError):
            resize_calibration(
                calibration,
                image_size_px=(640, 480),
            )

    def test_resize_preserves_distortion_coefficients(self):
        distortion = (0.1, -0.05, 0.001, -0.002, 0.01)

        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=distortion,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        resized = resize_calibration(
            calibration,
            image_size_px=(1280, 720),
        )

        self.assertEqual(
            resized.distortion_coefficients,
            distortion,
        )


class CalibrationCropTests(unittest.TestCase):
    def test_crop_shifts_optical_centre_and_keeps_focal_lengths(self):
        # Crop from (320, 180) and check that only the optical-centre coordinates move.
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        cropped = crop_calibration(
            calibration,
            origin_px=(320, 180),
            image_size_px=(1280, 720),
        )

        self.assertEqual(
            cropped.camera_matrix,
            (
                (1200.0, 0.0, 640.0),
                (0.0, 1200.0, 360.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.assertEqual(cropped.image_size_px, (1280, 720))

    def test_rejects_crop_outside_original_image(self):
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        # Move the crop just beyond each image edge and require a ValueError.
        for origin in (
            (-1, 180),
            (320, -1),
            (641, 180),
            (320, 361),
        ):
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    crop_calibration(
                        calibration,
                        origin_px=origin,
                        image_size_px=(1280, 720),
                    )

    def test_accepts_crop_ending_at_image_boundary(self):
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        cropped = crop_calibration(
            calibration,
            origin_px=(640, 360),
            image_size_px=(1280, 720),
        )

        self.assertEqual(
            cropped.camera_matrix,
            (
                (1200.0, 0.0, 320.0),
                (0.0, 1200.0, 180.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.assertEqual(cropped.image_size_px, (1280, 720))

    def test_rejects_unsupported_distortion_coefficient_count(self):
        for count in (3, 6):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    CameraCalibration(
                        camera_matrix=(
                            (1200.0, 0.0, 960.0),
                            (0.0, 1200.0, 540.0),
                            (0.0, 0.0, 1.0),
                        ),
                        distortion_coefficients=(0.0,) * count,
                        image_size_px=(1920, 1080),
                        source="synthetic test fixture",
                    )

    def test_accepts_fourteen_distortion_coefficients(self):
        distortion = (0.0,) * 12 + (-0.005, 0.002)

        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 960.0),
                (0.0, 1200.0, 540.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=distortion,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        self.assertEqual(
            calibration.distortion_coefficients,
            distortion,
        )


class CalibrationLoadTests(unittest.TestCase):
    def test_loads_calibration_from_json_file(self):
        # Save known calibration values, load them with a matching ID and compare them.
        matrix = (
            (1200.0, 0.0, 960.0),
            (0.0, 1200.0, 540.0),
            (0.0, 0.0, 1.0),
        )
        distortion = (0.0,) * 12 + (-0.005, 0.002)

        record = {
            "source": "synthetic test fixture",
            "device_mxid": "test-device",
            "camera_socket": "CAM_A",
            "distortion_model": "CameraModel.Perspective",
            "image_size_px": [1920, 1080],
            "camera_matrix": matrix,
            "distortion_coefficients": distortion,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            expected = CameraCalibration(
                camera_matrix=matrix,
                distortion_coefficients=distortion,
                image_size_px=(1920, 1080),
                source=str(path.resolve()),
            )

            loaded = load_calibration(
                path,
                expected_device_mxid="test-device",
            )
            self.assertEqual(loaded, expected)

    def test_rejects_fisheye_calibration(self):
        record = {
            "source": "synthetic test fixture",
            "device_mxid": "test-device",
            "camera_socket": "CAM_A",
            "distortion_model": "CameraModel.Fisheye",
            "image_size_px": [1920, 1080],
            "camera_matrix": [
                [1200.0, 0.0, 960.0],
                [0.0, 1200.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.0] * 4,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_calibration(path)

    def test_rejects_calibration_from_wrong_camera_socket(self):
        record = {
            "source": "synthetic test fixture",
            "device_mxid": "test-device",
            "camera_socket": "CAM_B",
            "distortion_model": "CameraModel.Perspective",
            "image_size_px": [1920, 1080],
            "camera_matrix": [
                [1200.0, 0.0, 960.0],
                [0.0, 1200.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.0] * 5,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_calibration(path)

    def test_rejects_calibration_from_wrong_device(self):
        record = {
            "source": "synthetic test fixture",
            "device_mxid": "other-device",
            "camera_socket": "CAM_A",
            "distortion_model": "CameraModel.Perspective",
            "image_size_px": [1920, 1080],
            "camera_matrix": [
                [1200.0, 0.0, 960.0],
                [0.0, 1200.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.0] * 5,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_calibration(
                    path,
                    expected_device_mxid="test-device",
                )


class StreamCalibrationEvidenceTests(unittest.TestCase):
    def test_saved_evidence_identifies_original_file_and_effective_geometry(self):
        # Keep the original 4K file separate from the calibration used for this stream.
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
            "distortion_coefficients": [0.0] * 14,
        }
        original_bytes = (json.dumps(record, indent=2) + "\n").encode("utf-8")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_bytes(original_bytes)

            for mode, expected_size, expected_matrix in (
                (
                    "exact",
                    (3840, 2160),
                    (
                        (2400.0, 0.0, 1920.0),
                        (0.0, 2400.0, 1080.0),
                        (0.0, 0.0, 1.0),
                    ),
                ),
                (
                    "oak-1080",
                    (1280, 720),
                    (
                        (1200.0, 0.0, 640.0),
                        (0.0, 1200.0, 360.0),
                        (0.0, 0.0, 1.0),
                    ),
                ),
            ):
                with self.subTest(mode=mode):
                    calibration, evidence = aruco_calibration.load_stream_calibration(
                        path,
                        image_size_px=(1280, 720),
                        mode=mode,
                    )

                    self.assertEqual(calibration.image_size_px, expected_size)
                    self.assertEqual(calibration.camera_matrix, expected_matrix)
                    self.assertEqual(
                        calibration.distortion_coefficients,
                        (0.0,) * 14,
                    )
                    self.assertEqual(
                        evidence,
                        {
                            "file": str(path.resolve()),
                            "sha256": hashlib.sha256(original_bytes).hexdigest(),
                            "original_record": record,
                            "effective": asdict(calibration),
                            "transform": mode,
                        },
                    )
                    self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
