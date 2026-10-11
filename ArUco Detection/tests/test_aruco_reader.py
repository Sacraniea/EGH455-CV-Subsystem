import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aruco_types import ArUcoConfig, ArUcoFailureReason, CameraCalibration, MarkerPose
from aruco_reader import read_aruco


def marker_packet(
    *,
    sequence_number,
    device_timestamp_s,
    host_received_timestamp_s,
):
    # Place marker 7 at (50, 50) in a clean 300-pixel square, with this test's identity.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
    gray = np.full((300, 300), 255, dtype=np.uint8)
    gray[50:250, 50:250] = marker

    return SimpleNamespace(
        image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        sequence_number=sequence_number,
        device_timestamp_s=device_timestamp_s,
        host_received_timestamp_s=host_received_timestamp_s,
    )


class ArUcoReaderTests(unittest.TestCase):
    def test_missing_calibration_preserves_marker_and_frame_evidence(self):
        # Read a generated marker without calibration and check the retained frame details.
        packet = marker_packet(
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        results = read_aruco(
            packet,
            calibration=None,
            config=config,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(result.marker_id, 7)
        np.testing.assert_allclose(
            result.corners_px,
            (
                (50.0, 50.0),
                (249.0, 50.0),
                (249.0, 249.0),
                (50.0, 249.0),
            ),
            rtol=0,
            atol=0.5,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.pose)
        self.assertEqual(
            result.failure_reason,
            ArUcoFailureReason.MISSING_CALIBRATION,
        )
        self.assertEqual(result.sequence_number, 42)
        self.assertEqual(result.device_timestamp_s, 12.5)
        self.assertEqual(result.host_received_timestamp_s, 12.6)

    def test_returns_valid_pose_with_matching_calibration(self):
        packet = marker_packet(
            sequence_number=43,
            device_timestamp_s=13.5,
            host_received_timestamp_s=13.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 150.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(300, 300),
            source="synthetic test fixture",
        )

        results = read_aruco(
            packet,
            calibration,
            config,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertTrue(result.valid)
        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.pose)
        self.assertEqual(
            result.pose.coordinate_frame,
            "camera_optical",
        )
        self.assertEqual(result.marker_id, 7)
        # Check the depth implied by a 0.2 m marker spanning 199 px at 600 px focal length.
        self.assertAlmostEqual(
            result.pose.translation_m[2],
            0.603015,
            places=5,
        )
        self.assertLessEqual(result.pose.reprojection_error_px, 1.0)
        self.assertEqual(result.sequence_number, 43)
        self.assertEqual(result.device_timestamp_s, 13.5)
        self.assertEqual(result.host_received_timestamp_s, 13.6)

    def test_retains_unpermitted_marker_but_withholds_pose(self):
        packet = marker_packet(
            sequence_number=44,
            device_timestamp_s=14.5,
            host_received_timestamp_s=14.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({23}),
        )

        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 150.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(300, 300),
            source="synthetic test fixture",
        )

        results = read_aruco(
            packet,
            calibration,
            config,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(
            result.failure_reason,
            ArUcoFailureReason.MARKER_ID_NOT_ALLOWED,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.pose)
        self.assertEqual(result.marker_id, 7)
        self.assertEqual(len(result.corners_px), 4)
        self.assertEqual(result.sequence_number, 44)
        self.assertEqual(result.device_timestamp_s, 14.5)
        self.assertEqual(result.host_received_timestamp_s, 14.6)

    def test_withholds_pose_when_calibration_size_mismatches_frame(self):
        packet = marker_packet(
            sequence_number=45,
            device_timestamp_s=15.5,
            host_received_timestamp_s=15.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 150.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        results = read_aruco(
            packet,
            calibration,
            config,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(
            result.failure_reason,
            ArUcoFailureReason.CALIBRATION_SIZE_MISMATCH,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.pose)
        self.assertEqual(result.marker_id, 7)
        self.assertEqual(result.sequence_number, 45)
        self.assertEqual(result.device_timestamp_s, 15.5)
        self.assertEqual(result.host_received_timestamp_s, 15.6)

    def test_rejected_pose_retains_error_and_frame_evidence(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        # Stretch only the width so these corners no longer fit the expected square.
        stretched = cv2.resize(
            marker,
            (240, 200),
            interpolation=cv2.INTER_NEAREST,
        )

        gray = np.full((300, 350), 255, dtype=np.uint8)
        gray[50:250, 50:290] = stretched

        packet = SimpleNamespace(
            image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            sequence_number=46,
            device_timestamp_s=16.5,
            host_received_timestamp_s=16.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 175.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(350, 300),
            source="synthetic test fixture",
        )

        results = read_aruco(
            packet,
            calibration,
            config,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(
            result.failure_reason,
            ArUcoFailureReason.REPROJECTION_ERROR_TOO_HIGH,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.pose)
        self.assertIsNotNone(result.reprojection_error_px)
        self.assertGreater(result.reprojection_error_px, 10.0)
        self.assertLess(result.reprojection_error_px, 20.0)
        self.assertEqual(result.marker_id, 7)
        self.assertEqual(result.sequence_number, 46)
        self.assertEqual(result.device_timestamp_s, 16.5)
        self.assertEqual(result.host_received_timestamp_s, 16.6)

    def test_returns_no_results_when_frame_has_no_markers(self):
        packet = SimpleNamespace(
            image=np.full(
                (300, 300, 3),
                255,
                dtype=np.uint8,
            ),
            sequence_number=47,
            device_timestamp_s=17.5,
            host_received_timestamp_s=17.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        results = read_aruco(
            packet,
            calibration=None,
            config=config,
            max_reprojection_error_px=1.0,
        )

        self.assertEqual(results, ())

    def test_preserves_marker_evidence_when_pose_solver_fails(self):
        packet = marker_packet(
            sequence_number=43,
            device_timestamp_s=13.5,
            host_received_timestamp_s=13.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 150.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(300, 300),
            source="synthetic test fixture",
        )

        # Force an unsuccessful solve and require a rejected result rather than an exception.
        with patch(
            "aruco_pose.cv2.solvePnP",
            return_value=(False, None, None),
        ):
            try:
                results = read_aruco(
                    packet,
                    calibration,
                    config,
                    max_reprojection_error_px=1.0,
                )

            except ValueError as error:
                self.fail(f"Solver failure escaped reader: {error}")

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(
            result.failure_reason,
            ArUcoFailureReason.POSE_SOLVER_FAILED,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.pose)
        self.assertIsNone(result.reprojection_error_px)
        self.assertEqual(result.marker_id, 7)
        self.assertEqual(len(result.corners_px), 4)
        self.assertEqual(result.sequence_number, 43)
        self.assertEqual(result.device_timestamp_s, 13.5)
        self.assertEqual(result.host_received_timestamp_s, 13.6)

    def test_preserves_marker_evidence_when_opencv_raises(self):
        packet = marker_packet(
            sequence_number=43,
            device_timestamp_s=13.5,
            host_received_timestamp_s=13.6,
        )

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 150.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(300, 300),
            source="synthetic test fixture",
        )

        # Force an OpenCV exception and check that the marker and timestamps survive it.
        with patch(
            "aruco_pose.cv2.solvePnP",
            side_effect=cv2.error("Synthetic OpenCV failure"),
        ):
            try:
                results = read_aruco(
                    packet,
                    calibration,
                    config,
                    max_reprojection_error_px=1.0,
                )

            except (ValueError, cv2.error) as error:
                self.fail(f"Solver failure escaped reader: {error}")

        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(
            result.failure_reason,
            ArUcoFailureReason.POSE_SOLVER_FAILED,
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.pose)
        self.assertIsNone(result.reprojection_error_px)
        self.assertEqual(result.marker_id, 7)
        self.assertEqual(len(result.corners_px), 4)
        self.assertEqual(result.sequence_number, 43)
        self.assertEqual(result.device_timestamp_s, 13.5)
        self.assertEqual(result.host_received_timestamp_s, 13.6)

    def test_solver_failure_does_not_stop_next_marker(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        gray = np.full((300, 600), 255, dtype=np.uint8)
        gray[50:250, 50:250] = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        gray[50:250, 350:550] = cv2.aruco.generateImageMarker(dictionary, 23, 200)

        packet = SimpleNamespace(
            image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            sequence_number=48,
            device_timestamp_s=18.5,
            host_received_timestamp_s=18.6,
        )
        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7, 23}),
        )
        calibration = CameraCalibration(
            camera_matrix=(
                (600.0, 0.0, 300.0),
                (0.0, 600.0, 150.0),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(600, 300),
            source="synthetic test fixture",
        )
        accepted_pose = MarkerPose(
            translation_m=(0.2, 0.0, 1.0),
            rotation_vector_rad=(3.0, 0.0, 0.0),
            reprojection_error_px=0.2,
        )

        # Fail the first marker's solve, then return an accepted pose for the next marker.
        with patch(
            "aruco_reader.estimate_marker_pose",
            side_effect=[
                cv2.error("First marker solver failure"),
                accepted_pose,
            ],
        ):
            results = read_aruco(
                packet,
                calibration,
                config,
                max_reprojection_error_px=1.0,
            )

        self.assertEqual([r.marker_id for r in results], [7, 23])
        failed, accepted = results

        self.assertFalse(failed.valid)
        self.assertIsNone(failed.pose)
        self.assertIsNone(failed.reprojection_error_px)
        self.assertEqual(
            failed.failure_reason,
            ArUcoFailureReason.POSE_SOLVER_FAILED,
        )

        self.assertTrue(accepted.valid)
        self.assertIsNone(accepted.failure_reason)
        self.assertEqual(accepted.pose, accepted_pose)
        self.assertEqual(accepted.reprojection_error_px, 0.2)

        for result in results:
            self.assertEqual(len(result.corners_px), 4)
            self.assertEqual(result.sequence_number, 48)
            self.assertEqual(result.device_timestamp_s, 18.5)
            self.assertEqual(result.host_received_timestamp_s, 18.6)


if __name__ == "__main__":
    unittest.main()
