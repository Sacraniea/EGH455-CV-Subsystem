"""Pure GCS payload mapping: preserve evidence and original image file bytes."""

import base64
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zlib

import cv2
import numpy as np


def native_record():
    # Build frame 42 with marker 7, its accepted pose and matching capture times.
    identity = dict(
        sequence_number=42,
        device_timestamp_s=12.5,
        host_received_timestamp_s=12.6,
        image_size_px=[320, 240],
    )
    observation = dict(
        schema_version=1,
        marker_id=7,
        corners_px=[[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]],
        valid=True,
        failure_reason=None,
        pose=dict(
            translation_m=[0.1, 0.2, 2.0],
            rotation_vector_rad=[0.0, 0.0, 0.1],
            reprojection_error_px=0.25,
            coordinate_frame="camera_optical",
        ),
        sequence_number=42,
        device_timestamp_s=12.5,
        host_received_timestamp_s=12.6,
        reprojection_error_px=0.25,
    )
    data = dict(
        schema_version=1,
        producer="aruco",
        **identity,
        corners_coordinate_frame="source_px",
        processing_time_ms=1.5,
        config=dict(
            dictionary_name="DICT_5X5_100",
            marker_side_length_m=0.2,
            permitted_ids=[7],
            max_reprojection_error_px=1.0,
        ),
        calibration=None,
        results=[observation]
    )
    processor = dict(
        processor="aruco",
        identity=dict(identity),
        data=data,
        failure_reason=None,
        error=None,
        processing_time_ms=2.0,
    )
    return dict(
        schema_version=1,
        producer="camera_stream",
        run_id="run-abc",
        source_kind="camera",
        **identity,
        coordinates="source_px",
        status="complete",
        processors=[processor],
        capture_to_dequeue_ms=100.0,
        timings_ms={"processing": 2.0}
    )


class GcsOutputTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("gcs_output"),
            "The pure GCS payload module must exist",
        )
        import gcs_output

        self.output = gcs_output

    def test_log_shows_readable_essentials_without_changing_native_evidence(self):
        record = native_record()
        original = deepcopy(record)

        payload = self.output.gcs_log_payload(record)

        self.assertEqual(set(payload), {"action", "content"})
        self.assertEqual(payload["action"], "ArUco detected")
        self.assertIsInstance(payload["content"], str)
        self.assertEqual(
            payload["content"],
            "Marker 7: 10.0 cm right, 20.0 cm down, 200.0 cm forward; image position (40, 40) px.\n"
            "Frame 42 | Device time 12.500 s | Run run-abc",
        )
        self.assertNotIn("calibration", payload["content"])
        self.assertNotIn("rotation_vector", payload["content"])
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        self.assertEqual(record, original)

    def test_rejected_pose_is_withheld_without_losing_decoded_marker_or_identity(self):
        record = native_record()
        result = record["processors"][0]["data"]["results"][0]
        result.update(
            valid=False,
            failure_reason="REPROJECTION_ERROR_TOO_HIGH",
            reprojection_error_px=4.0,
        )
        original = deepcopy(record)

        # Check that the rejected pose is removed while marker 7 and its frame identity remain.
        payload = self.output.gcs_log_payload(record)
        self.assertEqual(payload["action"], "ArUco detected - pose unavailable")
        self.assertIn(
            "Marker 7: pose unavailable (corner fit error too high: 4.00 px)",
            payload["content"],
        )
        self.assertIn("image position (40, 40) px", payload["content"])
        self.assertIn(
            "Frame 42 | Device time 12.500 s | Run run-abc", payload["content"]
        )
        self.assertNotIn("forward", payload["content"])
        self.assertNotIn("REPROJECTION_ERROR_TOO_HIGH", payload["content"])
        self.assertEqual(record, original)

    def test_failure_reason_overrides_an_inconsistent_valid_flag(self):
        record = native_record()
        record["processors"][0]["data"]["results"][0][
            "failure_reason"
        ] = "MISSING_CALIBRATION"
        payload = self.output.gcs_log_payload(record)
        self.assertEqual(payload["action"], "ArUco detected - pose unavailable")
        self.assertIn("calibration missing", payload["content"])
        self.assertNotIn("forward", payload["content"])

    def test_mixed_observations_keep_accepted_pose_and_withhold_rejected_pose(self):
        record = native_record()
        results = record["processors"][0]["data"]["results"]
        rejected = deepcopy(results[0])
        rejected.update(marker_id=8, valid=False, failure_reason="ID_NOT_PERMITTED")
        results.append(rejected)
        payload = self.output.gcs_log_payload(record)
        self.assertEqual(payload["action"], "ArUco detected")
        self.assertIn(
            "Marker 7: 10.0 cm right, 20.0 cm down, 200.0 cm forward",
            payload["content"],
        )
        self.assertIn(
            "Marker 8: pose unavailable (marker ID not permitted)", payload["content"]
        )
        self.assertEqual(payload["content"].count("forward"), 1)

    def test_negative_translation_uses_camera_left_and_up_not_negative_distance(self):
        record = native_record()
        record["processors"][0]["data"]["results"][0]["pose"]["translation_m"] = [
            -0.1,
            -0.2,
            2.0,
        ]
        text = self.output.gcs_log_payload(record)["content"]
        self.assertIn("10.0 cm left, 20.0 cm up, 200.0 cm forward", text)
        self.assertNotIn("distance", text.lower())

    def test_synthetic_records_are_clearly_labelled(self):
        record = native_record()
        record["source_kind"] = "synthetic"
        self.assertTrue(
            self.output.gcs_log_payload(record)["content"].startswith(
                "Synthetic demo.\n"
            )
        )

    def test_unknown_rejection_is_readable_and_does_not_leak_candidate_pose(self):
        record = native_record()
        record["processors"][0]["data"]["results"][0].update(
            valid=False, failure_reason="CUSTOM_REJECTION"
        )
        text = self.output.gcs_log_payload(record)["content"]
        self.assertIn("pose unavailable (custom rejection)", text)
        self.assertNotIn("forward", text)

    def test_wrong_coordinate_frame_or_invalid_translation_is_not_labelled_as_camera_pose(
        self,
    ):
        for change in (
            {"coordinate_frame": "world"},
            {"translation_m": [0.1, 0.2]},
            {"translation_m": [0.1, 0.2, -2.0]},
        ):
            record = native_record()
            record["processors"][0]["data"]["results"][0]["pose"].update(change)

            with self.subTest(change=change):
                payload = self.output.gcs_log_payload(record)
                self.assertEqual(payload["action"], "ArUco detected - pose unavailable")
                self.assertNotIn("forward", payload["content"])

    def test_empty_or_absent_aruco_observations_produce_no_log(self):
        record = native_record()
        record["processors"][0]["data"]["results"] = []
        self.assertIsNone(self.output.gcs_log_payload(record))
        record["processors"] = []
        self.assertIsNone(self.output.gcs_log_payload(record))

    def test_unrelated_record_schema_is_rejected(self):
        record = native_record()
        record["producer"] = "different_service"

        with self.assertRaises(ValueError):
            self.output.gcs_log_payload(record)

    def test_processor_failure_has_its_own_action_and_never_publishes_candidate_pose(
        self,
    ):
        record = native_record()
        record["status"] = "partial"
        processor = record["processors"][0]
        processor.update(
            failure_reason="PROCESSOR_ERROR", error="RuntimeError: detector failed"
        )
        payload = self.output.gcs_log_payload(record)
        self.assertEqual(payload["action"], "ArUco processing error")
        self.assertIn(
            "Detection processing failed; no pose reported.", payload["content"]
        )
        self.assertIn(
            "Marker 7: pose unavailable (detection processing failed)",
            payload["content"],
        )
        self.assertNotIn("forward", payload["content"])
        processor["data"] = {}
        self.assertEqual(
            self.output.gcs_log_payload(record)["action"], "ArUco processing error"
        )

    def test_only_aruco_branch_is_mapped_without_invented_gauge_values(self):
        record = native_record()
        record["processors"].insert(0, dict(processor="gauge", data={"value_bar": 9.0}))
        content = self.output.gcs_log_payload(record)["content"]
        self.assertIn("Marker 7", content)
        self.assertNotIn("value_bar", content)
        self.assertNotIn("gauge", content.lower())

    def test_nonfinite_evidence_is_rejected_by_strict_json(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            record = native_record()
            record["processors"][0]["data"]["results"][0]["pose"]["translation_m"][
                0
            ] = value

            with self.subTest(value=value), self.assertRaises(ValueError):
                self.output.gcs_log_payload(record)

    def test_nonfinite_rejected_candidate_pose_is_removed_before_serializing(self):
        record = native_record()
        result = record["processors"][0]["data"]["results"][0]
        result.update(valid=False, failure_reason="NONFINITE_POSE")
        result["pose"]["translation_m"][0] = float("nan")
        payload = self.output.gcs_log_payload(record)
        self.assertIn("pose calculation returned invalid numbers", payload["content"])
        self.assertNotIn("forward", payload["content"])
        self.assertNotIn("nan", payload["content"])
        json.dumps(payload, allow_nan=False)

    def test_detection_preserves_original_jpeg_and_png_bytes_and_explicit_timestamp(
        self,
    ):
        image = np.full((8, 12, 3), 80, np.uint8)

        with TemporaryDirectory() as directory:
            for suffix, mime in (
                (".jpg", "image/jpeg"),
                (".jpeg", "image/jpeg"),
                (".png", "image/png"),
            ):
                encoded = cv2.imencode(suffix, image)[1].tobytes()
                # Add file metadata that would be lost if the image were decoded and encoded again.
                comment = b"Original file metadata must survive unchanged"

                # Insert the metadata after the JPEG start marker or before the PNG end chunk.
                if mime == "image/jpeg":
                    encoded = (
                        encoded[:2]
                        + b"\xff\xfe"
                        + (len(comment) + 2).to_bytes(2, "big")
                        + comment
                        + encoded[2:]
                    )

                else:
                    data = b"Comment\x00" + comment
                    chunk = (
                        len(data).to_bytes(4, "big")
                        + b"tEXt"
                        + data
                        + zlib.crc32(b"tEXt" + data).to_bytes(4, "big")
                    )
                    encoded = encoded[:-12] + chunk + encoded[-12:]

                path = Path(directory) / ("original" + suffix)
                path.write_bytes(encoded)

                with self.subTest(suffix=suffix):
                    payload = self.output.gcs_detection_payload(
                        path,
                        "ArUco",
                        "Observed ID 7",
                        timestamp="2026-10-08T01:02:03.456Z",
                    )
                    self.assertEqual(
                        set(payload),
                        {"type", "description", "mime", "image", "timestamp"},
                    )
                    self.assertEqual(payload["type"], "ArUco")
                    self.assertEqual(payload["description"], "Observed ID 7")
                    self.assertEqual(payload["mime"], mime)
                    self.assertEqual(payload["timestamp"], "2026-10-08T01:02:03.456Z")
                    self.assertEqual(
                        base64.b64decode(payload["image"], validate=True), encoded
                    )
                    self.assertEqual(path.read_bytes(), encoded)
                    json.dumps(payload, allow_nan=False)

    def test_default_image_timestamp_is_current_utc_publication_time(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "original.png"
            path.write_bytes(
                cv2.imencode(".png", np.zeros((4, 6, 3), np.uint8))[1].tobytes()
            )
            before = datetime.now(timezone.utc)
            payload = self.output.gcs_detection_payload(
                path, "ArUco", "Operator supplied image"
            )
            after = datetime.now(timezone.utc)
            published = datetime.fromisoformat(
                payload["timestamp"].replace("Z", "+00:00")
            )
            self.assertEqual(published.utcoffset().total_seconds(), 0.0)
            self.assertLessEqual(before, published)
            self.assertLessEqual(published, after)

    def test_detection_rejects_bad_images_mismatched_extensions_and_unsupported_files(
        self,
    ):
        png = cv2.imencode(".png", np.zeros((4, 6, 3), np.uint8))[1].tobytes()

        with TemporaryDirectory() as directory:
            for name, raw in [
                ("fake.jpg", b"not an image"),
                ("mismatch.jpg", png),
                ("unsupported.gif", png),
                ("empty.png", b""),
                ("truncated.png", b"\x89PNG\r\n\x1a\n"),
            ]:
                path = Path(directory) / name
                path.write_bytes(raw)

                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.output.gcs_detection_payload(path, "ArUco", "Unusable input")

    def test_detection_requires_text_metadata_and_an_explicit_timestamp_with_timezone(
        self,
    ):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "original.png"
            path.write_bytes(
                cv2.imencode(".png", np.zeros((4, 6, 3), np.uint8))[1].tobytes()
            )

            for target_type, description, timestamp in [
                (None, "text", None),
                ("ArUco", None, None),
                ("ArUco", "text", 12.5),
                ("ArUco", "text", "2026-10-08T01:02:03"),
            ]:
                with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                    self.output.gcs_detection_payload(
                        path, target_type, description, timestamp=timestamp
                    )


if __name__ == "__main__":
    unittest.main()
