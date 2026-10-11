import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from aruco_image import main


class ArUcoImageTests(unittest.TestCase):
    def test_saves_overlay_and_results_without_calibration(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        image[50:250, 50:250] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "marker.png"
            output_dir = Path(directory) / "output"
            self.assertTrue(cv2.imwrite(str(input_path), image))

            exit_code = main(
                [
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--permitted-ids",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)

            with (output_dir / "results.json").open(encoding="utf-8") as file:
                report = json.load(file)

            self.assertEqual(
                report["source_image"],
                str(input_path.resolve()),
            )
            self.assertEqual(len(report["results"]), 1)
            result = report["results"][0]

            self.assertEqual(result["marker_id"], 7)
            self.assertEqual(result["failure_reason"], "MISSING_CALIBRATION")
            self.assertFalse(result["valid"])
            self.assertIsNone(result["pose"])
            self.assertIsNone(result["sequence_number"])
            self.assertIsNone(result["device_timestamp_s"])
            self.assertIsNone(result["host_received_timestamp_s"])
            np.testing.assert_allclose(
                result["corners_px"],
                [[50, 50], [249, 50], [249, 249], [50, 249]],
                atol=0.5,
                rtol=0,
            )

            overlay = cv2.imread(str(output_dir / "overlay.png"))
            self.assertIsNotNone(overlay)
            self.assertEqual(overlay.shape, image.shape)
            self.assertTrue(np.any(overlay != image))

            # Reload the input file and compare its pixels with the generated source image.
            np.testing.assert_array_equal(cv2.imread(str(input_path)), image)

    def test_saves_valid_pose_with_matching_calibration(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        image[50:250, 50:250] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "marker.png"
            output_dir = Path(directory) / "output"
            self.assertTrue(cv2.imwrite(str(input_path), image))

            calibration_path = Path(directory) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "camera_socket": "CAM_A",
                        "device_mxid": "synthetic-test-device",
                        "distortion_model": "CameraModel.Perspective",
                        "image_size_px": [400, 300],
                        "camera_matrix": [
                            [600.0, 0.0, 200.0],
                            [0.0, 600.0, 150.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "distortion_coefficients": [0.0] * 5,
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--permitted-ids",
                    "7",
                    "--calibration",
                    str(calibration_path),
                    "--max-reprojection-error-px",
                    "1.0",
                ]
            )

            self.assertEqual(exit_code, 0)

            with (output_dir / "results.json").open(encoding="utf-8") as file:
                report = json.load(file)

            self.assertEqual(
                report["source_image"],
                str(input_path.resolve()),
            )
            self.assertEqual(len(report["results"]), 1)
            result = report["results"][0]

            self.assertEqual(result["marker_id"], 7)
            self.assertIsNone(result["failure_reason"])
            self.assertTrue(result["valid"])
            self.assertIsNotNone(result["pose"])
            self.assertEqual(
                result["pose"]["coordinate_frame"],
                "camera_optical",
            )
            self.assertTrue(np.all(np.isfinite(result["pose"]["translation_m"])))
            self.assertGreater(result["pose"]["translation_m"][2], 0.0)
            self.assertLessEqual(result["pose"]["reprojection_error_px"], 1.0)
            self.assertIsNone(result["sequence_number"])
            self.assertIsNone(result["device_timestamp_s"])
            self.assertIsNone(result["host_received_timestamp_s"])
            np.testing.assert_allclose(
                result["corners_px"],
                [[50, 50], [249, 50], [249, 249], [50, 249]],
                atol=0.5,
                rtol=0,
            )

            overlay = cv2.imread(str(output_dir / "overlay.png"))
            self.assertIsNotNone(overlay)
            self.assertEqual(overlay.shape, image.shape)
            self.assertTrue(np.any(overlay != image))

            # Reload the input file and compare its pixels with the generated source image.
            np.testing.assert_array_equal(cv2.imread(str(input_path)), image)


if __name__ == "__main__":
    unittest.main()
