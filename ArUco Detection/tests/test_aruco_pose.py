import unittest
import cv2
import numpy as np

from aruco_types import (
    CameraCalibration,
    MarkerObservation,
    ArUcoFailureReason,
    MarkerPose,
)
from aruco_pose import (
    marker_object_points,
    estimate_marker_pose,
    reprojection_rmse_px,
    pose_rejection_reason,
)


class MarkerGeometryTests(unittest.TestCase):
    def test_returns_marker_corners_in_metres_and_solver_order(self):
        points = marker_object_points(marker_side_length_m=0.2)

        expected = np.array(
            [
                [-0.1, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [0.1, -0.1, 0.0],
                [-0.1, -0.1, 0.0],
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            points,
            expected,
            rtol=0,
            atol=1e-12,
        )


class MarkerPoseTests(unittest.TestCase):
    def test_recovers_known_marker_pose(self):
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 640.0),
                (0.0, 1200.0, 360.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.05, -0.01, 0.001, -0.002, 0.0),
            image_size_px=(1280, 720),
            source="synthetic test fixture",
        )

        expected_translation = np.array([0.2, 0.1, 2.0])
        rotation = np.array([2.8, 0.1, 0.15])

        # Generate image corners from a known pose, then ask the solver to recover it.
        image_points, _ = cv2.projectPoints(
            marker_object_points(marker_side_length_m=0.2),
            rotation,
            expected_translation,
            np.array(calibration.camera_matrix),
            np.array(calibration.distortion_coefficients),
        )

        observation = MarkerObservation(
            marker_id=7,
            corners_px=tuple(
                (float(x), float(y)) for x, y in image_points.reshape(4, 2)
            ),
        )

        pose = estimate_marker_pose(
            observation,
            calibration,
            marker_side_length_m=0.2,
        )

        np.testing.assert_allclose(
            pose.translation_m,
            expected_translation,
            rtol=0,
            atol=1e-6,
        )

        # Convert both rotation vectors to matrices and compare the resulting orientations.
        expected_rotation_matrix, _ = cv2.Rodrigues(rotation)

        recovered_rotation_matrix, _ = cv2.Rodrigues(
            np.asarray(
                pose.rotation_vector_rad,
                dtype=np.float64,
            )
        )

        np.testing.assert_allclose(
            recovered_rotation_matrix,
            expected_rotation_matrix,
            rtol=0,
            atol=1e-6,
        )

        self.assertLess(pose.reprojection_error_px, 1e-6)

    def test_reports_error_for_inconsistent_marker_corners(self):
        calibration = CameraCalibration(
            camera_matrix=(
                (1200.0, 0.0, 640.0),
                (0.0, 1200.0, 360.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1280, 720),
            source="synthetic test fixture",
        )

        observation = MarkerObservation(
            marker_id=7,
            corners_px=(
                (580.0, 300.0),
                (700.0, 300.0),
                (700.0, 420.0),
                (580.0, 430.0),  # Move one corner down by 10 px.
            ),
        )

        pose = estimate_marker_pose(
            observation,
            calibration,
            marker_side_length_m=0.2,
        )

        self.assertGreater(pose.reprojection_error_px, 1.0)
        self.assertLess(pose.reprojection_error_px, 2.0)


class ReprojectionErrorTests(unittest.TestCase):
    def test_measures_rms_corner_error_in_pixels(self):
        observed = np.array(
            [
                [10.0, 10.0],
                [30.0, 10.0],
                [30.0, 30.0],
                [10.0, 30.0],
            ],
            dtype=np.float64,
        )

        projected = observed.copy()
        # Move one corner by (3, 4) px, giving a 5 px error and a four-corner RMS of 2.5 px.
        projected[0] += (3.0, 4.0)

        error = reprojection_rmse_px(observed, projected)

        self.assertAlmostEqual(error, 2.5, places=12)


class PoseRejectionTests(unittest.TestCase):
    def test_rejects_pose_above_reprojection_error_limit(self):
        pose = MarkerPose(
            translation_m=(0.2, 0.1, 2.0),
            rotation_vector_rad=(2.8, 0.1, 0.15),
            reprojection_error_px=2.5,
        )

        reason = pose_rejection_reason(
            pose,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(
            reason,
            ArUcoFailureReason.REPROJECTION_ERROR_TOO_HIGH,
        )

    def test_accepts_pose_at_or_below_reprojection_error_limit(self):
        for error_px in (0.0, 0.5, 1.0):
            with self.subTest(error_px=error_px):
                pose = MarkerPose(
                    translation_m=(0.2, 0.1, 2.0),
                    rotation_vector_rad=(2.8, 0.1, 0.15),
                    reprojection_error_px=error_px,
                )

                reason = pose_rejection_reason(
                    pose,
                    max_reprojection_error_px=1.0,
                )

                self.assertIsNone(reason)

    def test_rejects_non_finite_translation(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                pose = MarkerPose(
                    translation_m=(0.2, 0.1, value),
                    rotation_vector_rad=(2.8, 0.1, 0.15),
                    reprojection_error_px=0.5,
                )

                reason = pose_rejection_reason(
                    pose,
                    max_reprojection_error_px=1.0,
                )

                self.assertEqual(
                    reason,
                    ArUcoFailureReason.NONFINITE_POSE,
                )

    def test_rejects_non_finite_rotation(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                pose = MarkerPose(
                    translation_m=(0.2, 0.1, 2.0),
                    rotation_vector_rad=(value, 0.1, 0.15),
                    reprojection_error_px=0.5,
                )

                reason = pose_rejection_reason(
                    pose,
                    max_reprojection_error_px=1.0,
                )

                self.assertEqual(
                    reason,
                    ArUcoFailureReason.NONFINITE_POSE,
                )

    def test_rejects_non_finite_reprojection_error(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                pose = MarkerPose(
                    translation_m=(0.2, 0.1, 2.0),
                    rotation_vector_rad=(2.8, 0.1, 0.15),
                    reprojection_error_px=value,
                )

                reason = pose_rejection_reason(
                    pose,
                    max_reprojection_error_px=1.0,
                )

                self.assertEqual(
                    reason,
                    ArUcoFailureReason.NONFINITE_POSE,
                )

    def test_rejects_non_positive_depth(self):
        for depth in (0.0, -2.0):
            with self.subTest(depth=depth):
                pose = MarkerPose(
                    translation_m=(0.2, 0.1, depth),
                    rotation_vector_rad=(2.8, 0.1, 0.15),
                    reprojection_error_px=0.5,
                )

                reason = pose_rejection_reason(
                    pose,
                    max_reprojection_error_px=1.0,
                )

                self.assertEqual(
                    reason,
                    ArUcoFailureReason.NONPOSITIVE_DEPTH,
                )

    def test_rejects_invalid_reprojection_error_limit(self):
        pose = MarkerPose(
            translation_m=(0.2, 0.1, 2.0),
            rotation_vector_rad=(2.8, 0.1, 0.15),
            reprojection_error_px=0.5,
        )

        for limit in (
            0.0,
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    pose_rejection_reason(
                        pose,
                        max_reprojection_error_px=limit,
                    )


if __name__ == "__main__":
    unittest.main()
