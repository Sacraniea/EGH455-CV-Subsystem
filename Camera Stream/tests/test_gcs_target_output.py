"""Check readable target logs without loading a detector or changing frame evidence."""

from copy import deepcopy
import json
import unittest

import gcs_output
from tests.test_gcs_output import native_record


def detection(class_name="gauge", *, value=2.9, box=(20, 10, 60, 50)):
    return dict(
        class_name=class_name,
        bbox_xyxy_px=box,
        gauge_reading=dict(valid=True, value_bar=value, failure_reason=None),
    )


def target_record(*detections, include_aruco=False):
    # Reuse a complete native frame and add target results with that frame's identity.
    record = native_record()
    identity = deepcopy(record["processors"][0]["identity"])

    if not include_aruco:
        record["processors"] = []

    record["processors"].append(
        dict(
            processor="targets",
            identity=identity,
            data=dict(schema_version=1, results=list(detections)),
            failure_reason=None,
            error=None,
            processing_time_ms=3.0,
        )
    )
    return record


class GcsTargetOutputTests(unittest.TestCase):
    def test_pending_confirmation_cannot_publish_a_candidate_pressure(self):
        observation = detection(value=6.22)
        observation["reading_status"] = "pending"
        payload = gcs_output.gcs_log_payloads(target_record(observation))[0]

        self.assertIn("awaiting confirmation", payload["content"])
        self.assertNotIn("6.22", payload["content"])
        self.assertNotIn("Pressure:", payload["content"])

    def test_valid_gauge_has_pressure_pixel_centre_and_compact_frame_identity(self):
        record = target_record(detection())

        self.assertEqual(
            gcs_output.gcs_log_payloads(record),
            [
                {
                    "action": "Gauge detected",
                    "content": (
                        "Pressure: 2.9 bar; image position (40, 30) px; Frame 42."
                    ),
                }
            ],
        )

    def test_pressure_keeps_two_decimals_without_spurious_trailing_zeros(self):
        for value, expected in (
            (2.934, "2.93"),
            (2.90000001, "2.9"),
            (0, "0"),
            (3, "3"),
        ):
            with self.subTest(value=value):
                payload = gcs_output.gcs_log_payloads(
                    target_record(detection(value=value))
                )[0]
                self.assertTrue(
                    payload["content"].startswith(f"Pressure: {expected} bar;")
                )

    def test_missing_rejected_or_invalid_reading_never_becomes_a_pressure(self):
        readings = [
            None,
            {},
            "2.9",
            {"valid": False, "value_bar": 2.9, "failure_reason": None},
            {"valid": 1, "value_bar": 2.9, "failure_reason": None},
            {"valid": True, "value_bar": 2.9, "failure_reason": "NO_NEEDLE"},
            {"valid": True, "value_bar": 2.9, "failure_reason": ""},
        ]
        readings.extend(
            dict(valid=True, value_bar=value, failure_reason=None)
            for value in (
                None,
                True,
                False,
                "2.9",
                float("nan"),
                float("inf"),
                -float("inf"),
            )
        )

        for reading in readings:
            with self.subTest(reading=reading):
                observation = detection()
                observation["gauge_reading"] = reading
                payloads = gcs_output.gcs_log_payloads(target_record(observation))

                self.assertEqual(payloads[0]["action"], "Gauge detected")
                self.assertEqual(
                    payloads[0]["content"],
                    "Pressure unavailable; image position (40, 30) px; Frame 42.",
                )
                json.dumps(payloads, allow_nan=False)

        observation = detection()
        del observation["gauge_reading"]
        self.assertIn(
            "Pressure unavailable",
            gcs_output.gcs_log_payloads(target_record(observation))[0]["content"],
        )

    def test_open_and_closed_valves_keep_distinct_states(self):
        payloads = gcs_output.gcs_log_payloads(
            target_record(detection("openValve"), detection("closedValve"))
        )

        self.assertEqual(
            payloads,
            [
                {
                    "action": "Valve detected",
                    "content": "Open valve; image position (40, 30) px; Frame 42.",
                },
                {
                    "action": "Valve detected",
                    "content": "Closed valve; image position (40, 30) px; Frame 42.",
                },
            ],
        )

    def test_each_target_keeps_its_own_reading_and_box(self):
        payloads = gcs_output.gcs_log_payloads(
            target_record(
                detection(value=2.9),
                detection(value=7.2, box=[100, 20, 161, 81]),
                detection("openValve", box=(200, 100, 300, 200)),
            )
        )

        self.assertEqual(len(payloads), 3)
        self.assertIn(
            "Pressure: 2.9 bar; image position (40, 30) px", payloads[0]["content"]
        )
        self.assertIn(
            "Pressure: 7.2 bar; image position (130.5, 50.5) px",
            payloads[1]["content"],
        )
        self.assertIn(
            "Open valve; image position (250, 150) px", payloads[2]["content"]
        )

    def test_mixed_frame_keeps_the_existing_aruco_message_before_target_messages(self):
        record = target_record(
            detection(), detection("closedValve"), include_aruco=True
        )
        original = deepcopy(record)
        expected_aruco = gcs_output.gcs_log_payload(record)

        payloads = gcs_output.gcs_log_payloads(record)

        self.assertEqual(payloads[0], expected_aruco)
        self.assertEqual(
            [payload["action"] for payload in payloads],
            ["ArUco detected", "Gauge detected", "Valve detected"],
        )
        self.assertEqual(record, original)
        self.assertEqual(json.loads(json.dumps(payloads, allow_nan=False)), payloads)

    def test_nonfinite_target_does_not_block_an_accepted_aruco_log(self):
        record = target_record(detection(value=float("nan")), include_aruco=True)
        original_json = json.dumps(record)

        payloads = gcs_output.gcs_log_payloads(record)

        self.assertEqual(payloads[0]["action"], "ArUco detected")
        self.assertIn("Pressure unavailable", payloads[1]["content"])
        self.assertEqual(json.loads(json.dumps(payloads, allow_nan=False)), payloads)
        self.assertEqual(json.dumps(record), original_json)

    def test_synthetic_label_is_present_on_each_target_message(self):
        record = target_record(detection(), detection("openValve"))
        record["source_kind"] = "synthetic"

        for payload in gcs_output.gcs_log_payloads(record):
            self.assertTrue(payload["content"].startswith("Synthetic demo. "))

    def test_failed_target_branch_reports_one_error_and_discards_stale_detections(self):
        record = target_record(detection(), detection("openValve"), include_aruco=True)
        record["processors"][-1].update(
            failure_reason="PROCESSOR_ERROR", error="Model failed after a prior frame"
        )
        original = deepcopy(record)

        payloads = gcs_output.gcs_log_payloads(record)

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["action"], "ArUco detected")
        self.assertEqual(
            payloads[1],
            {
                "action": "Target processing error",
                "content": "Target processing failed; Frame 42.",
            },
        )
        self.assertEqual(record, original)

    def test_synthetic_branch_error_keeps_the_synthetic_label(self):
        record = target_record()
        record["source_kind"] = "synthetic"
        record["processors"][0]["failure_reason"] = "PROCESSOR_ERROR"

        self.assertEqual(
            gcs_output.gcs_log_payloads(record)[0]["content"],
            "Synthetic demo. Target processing failed; Frame 42.",
        )

    def test_absent_empty_and_unknown_target_results_add_no_messages(self):
        self.assertEqual(gcs_output.gcs_log_payloads(target_record()), [])
        self.assertEqual(
            gcs_output.gcs_log_payloads(target_record(detection("unknown"))), []
        )
        record = native_record()
        self.assertEqual(
            gcs_output.gcs_log_payloads(record), [gcs_output.gcs_log_payload(record)]
        )
        record["processors"] = []
        self.assertEqual(gcs_output.gcs_log_payloads(record), [])

    def test_invalid_boxes_do_not_invent_pixel_positions(self):
        for box in (
            None,
            [],
            [1, 2, 3],
            [True, 0, 10, 20],
            [0, 0, float("inf"), 20],
            [0, 0, float("nan"), 20],
            [20, 0, 10, 20],
            [0, 5, 10, 5],
        ):
            with self.subTest(box=box):
                payloads = gcs_output.gcs_log_payloads(
                    target_record(detection(box=box))
                )
                self.assertIn("image position unavailable", payloads[0]["content"])
                json.dumps(payloads, allow_nan=False)

    def test_non_native_producer_is_rejected(self):
        record = target_record(detection())
        record["producer"] = "another_service"

        with self.assertRaises(ValueError):
            gcs_output.gcs_log_payloads(record)


if __name__ == "__main__":
    unittest.main()
