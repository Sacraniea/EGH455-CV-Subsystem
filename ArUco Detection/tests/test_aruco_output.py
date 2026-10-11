import json
import unittest

from aruco_output import result_to_record
from aruco_types import (
    ArUcoFailureReason,
    ArUcoReadResult,
    MarkerPose,
)


class ArUcoOutputTests(unittest.TestCase):
    def test_serializes_rejection_with_marker_and_frame_evidence(self):
        result = ArUcoReadResult(
            marker_id=7,
            corners_px=(
                (50.0, 50.0),
                (249.0, 50.0),
                (249.0, 249.0),
                (50.0, 249.0),
            ),
            pose=None,
            failure_reason=ArUcoFailureReason.MISSING_CALIBRATION,
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
            reprojection_error_px=None,
        )

        record = result_to_record(result)

        self.assertEqual(
            record,
            {
                "schema_version": 1,
                "marker_id": 7,
                "corners_px": [
                    [50.0, 50.0],
                    [249.0, 50.0],
                    [249.0, 249.0],
                    [50.0, 249.0],
                ],
                "valid": False,
                "pose": None,
                "failure_reason": "MISSING_CALIBRATION",
                "sequence_number": 42,
                "device_timestamp_s": 12.5,
                "host_received_timestamp_s": 12.6,
                "reprojection_error_px": None,
            },
        )

        encoded = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(encoded), record)

    def test_serializes_valid_pose(self):
        result = ArUcoReadResult(
            marker_id=7,
            corners_px=(
                (50.0, 50.0),
                (249.0, 50.0),
                (249.0, 249.0),
                (50.0, 249.0),
            ),
            pose=MarkerPose(
                translation_m=(0.2, 0.1, 2.0),
                rotation_vector_rad=(2.8, 0.1, 0.15),
                reprojection_error_px=0.25,
            ),
            failure_reason=None,
            sequence_number=43,
            device_timestamp_s=13.5,
            host_received_timestamp_s=13.6,
            reprojection_error_px=0.25,
        )

        record = result_to_record(result)

        self.assertTrue(record["valid"])
        self.assertIsNone(record["failure_reason"])
        self.assertEqual(
            record["pose"],
            {
                "translation_m": [0.2, 0.1, 2.0],
                "rotation_vector_rad": [2.8, 0.1, 0.15],
                "reprojection_error_px": 0.25,
                "coordinate_frame": "camera_optical",
            },
        )
        self.assertEqual(record["reprojection_error_px"], 0.25)

        encoded = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(encoded), record)

    def test_serializes_nonfinite_error_as_null(self):
        # Export each invalid pixel error and verify that strict JSON stores it as null.
        for error_px in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(error_px=error_px):
                result = ArUcoReadResult(
                    marker_id=7,
                    corners_px=(
                        (50.0, 50.0),
                        (249.0, 50.0),
                        (249.0, 249.0),
                        (50.0, 249.0),
                    ),
                    pose=None,
                    failure_reason=ArUcoFailureReason.NONFINITE_POSE,
                    sequence_number=44,
                    device_timestamp_s=14.5,
                    host_received_timestamp_s=14.6,
                    reprojection_error_px=error_px,
                )

                record = result_to_record(result)

                self.assertIsNone(record["reprojection_error_px"])
                self.assertIsNone(record["pose"])
                self.assertFalse(record["valid"])
                self.assertEqual(
                    record["failure_reason"],
                    "NONFINITE_POSE",
                )

                encoded = json.dumps(record, allow_nan=False)
                self.assertEqual(json.loads(encoded), record)


if __name__ == "__main__":
    unittest.main()
