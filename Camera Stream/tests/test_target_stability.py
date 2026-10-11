"""Check that gauge confirmation follows each current target without hiding jumps."""

import json
import unittest
from unittest.mock import patch

import numpy as np

import stream_target_detection
from stream_target_detection import TargetDetectionProcessor
from gauge_types import FailureReason, GaugeReadResult
from stream_processing import process_frame
from stream_types import FramePacket


def reading(value):
    return GaugeReadResult.success(
        value_bar=value,
        angle_degrees=0.0,
        dial_quality=0.8,
        needle_quality=0.7,
        centre=(40.0, 40.0),
        tip=(40.0, 15.0),
        radius=None,
    )


class PreparedPredictor:
    def __init__(self):
        self.detections = []
        self.calls = 0
        self.labels = []

    def predict_detections(self, image):
        self.calls += 1
        return self.detections

    def annotate_image(self, image, labels):
        self.labels = labels
        return image.copy()


class TargetStabilityTests(unittest.TestCase):
    def setUp(self):
        self.predictor = PreparedPredictor()
        self.processor = TargetDetectionProcessor(predictor=self.predictor)
        self.image = np.zeros((200, 400, 3), np.uint8)
        self.sequence = 0

    def frame(self, values, *, boxes=None, timestamp=None, classes=None):
        self.sequence += 1
        boxes = boxes if boxes is not None else [(20, 20, 100, 100)] * len(values)
        classes = classes if classes is not None else ["gauge"] * len(values)
        self.predictor.detections = [
            {
                "class_name": name,
                "bbox_xyxy_px": box,
                "box_xywhn": (0.15, 0.3, 0.2, 0.4),
                "gauge_crop": self.image[box[1]:box[3], box[0]:box[2]].copy(),
            }
            for name, box in zip(classes, boxes)
        ]
        timestamp = self.sequence / 10 if timestamp is None else timestamp
        self.packet = FramePacket(
            self.image,
            self.sequence,
            timestamp,
            timestamp + 0.02,
        )
        supplied = [
            value if isinstance(value, GaugeReadResult) else reading(value)
            for name, value in zip(classes, values)
            if name == "gauge"
        ]

        with patch("stream_target_detection.read_gauge", side_effect=supplied):
            self.result, = process_frame(self.packet, [self.processor])

        self.assertIsNone(self.result.failure_reason, self.result.error)
        json.dumps(self.result.to_record(), allow_nan=False)
        return self.result.data["results"]

    def test_recorded_spike_is_withheld_and_real_change_confirms_after_three_reads(self):
        # The 2.26 -> 6.22 -> 2.30 jump came from the recorded same-box gauge.
        values = [2.26, 2.30, 2.26, 6.22, 2.30, 2.26, 2.30, 6.22, 6.20, 6.22]
        expected = [None, None, 2.26, None, None, None, 2.30, None, None, 6.22]
        tracks = []

        for value, accepted in zip(values, expected):
            result, = self.frame([value])
            tracks.append(result["track_id"])
            self.assertEqual(result["raw_gauge_reading"]["value_bar"], value)
            self.assertEqual(result["gauge_reading"]["value_bar"], accepted)
            self.assertEqual(result["gauge_reading"]["valid"], accepted is not None)
            self.assertEqual(
                result["reading_status"],
                "pending" if accepted is None else "confirmed",
            )

        self.assertEqual(len(set(tracks)), 1)
        self.assertEqual(self.predictor.calls, len(values))

    def test_invalid_reading_clears_history_and_never_replays_old_pressure(self):
        for reason in (
            FailureReason.NO_NEEDLE,
            FailureReason.DIAL_TOO_BLURRED,
            FailureReason.NO_PIVOT,
        ):
            with self.subTest(reason=reason):
                for _ in range(3):
                    self.frame([2.3])

                rejected = GaugeReadResult.failure(reason)
                result, = self.frame([rejected])
                self.assertEqual(result["reading_status"], "unavailable")
                self.assertEqual(result["gauge_reading"]["failure_reason"], reason.value)
                self.assertIsNone(result["gauge_reading"]["value_bar"])
                self.assertEqual(result["raw_gauge_reading"], result["gauge_reading"])
                self.assertIsNone(result["gauge_reading"]["centre"])
                self.assertIsNone(result["gauge_reading"]["tip"])
                self.assertIsNone(result["gauge_reading"]["radius"])

                for expected in (None, None, 2.3):
                    result, = self.frame([2.3])
                    self.assertEqual(result["gauge_reading"]["value_bar"], expected)

    def test_nonfinite_or_boolean_pressure_is_unavailable_and_json_safe(self):
        for value in (float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(value=value):
                for _ in range(3):
                    self.frame([2.3])

                result, = self.frame([value])
                self.assertEqual(result["reading_status"], "unavailable")
                self.assertIsNone(result["gauge_reading"]["value_bar"])
                self.assertEqual(
                    result["gauge_reading"]["failure_reason"],
                    "INVALID_READING",
                )
                raw = result["raw_gauge_reading"]
                self.assertTrue(raw["valid"])

                if value is True:
                    self.assertIs(raw["value_bar"], True)

                else:
                    self.assertIsNone(raw["value_bar"])
                    self.assertEqual(raw["metadata"]["nonfinite_fields"], ["value_bar"])

                following, = self.frame([2.3])
                self.assertEqual(following["reading_status"], "pending")

        result, = self.frame([2.3])
        self.assertEqual(result["reading_status"], "pending")

    def test_two_gauges_keep_separate_histories_when_detection_order_changes(self):
        left = (20, 20, 100, 100)
        right = (220, 20, 300, 100)
        first = self.frame([2.3, 7.1], boxes=[left, right])
        left_id, right_id = (target["track_id"] for target in first)
        self.assertNotEqual(left_id, right_id)
        self.frame([7.1, 2.3], boxes=[right, left])
        current = self.frame([7.1, 2.3], boxes=[right, left])

        self.assertEqual(
            [target["track_id"] for target in current],
            [right_id, left_id],
        )
        self.assertEqual(
            [target["gauge_reading"]["value_bar"] for target in current],
            [7.1, 2.3],
        )

    def test_missing_detection_resets_confirmation_and_long_loss_expires_identity(self):
        for _ in range(3):
            original, = self.frame([2.3])

        self.frame([])
        returned, = self.frame([2.3])
        self.assertEqual(returned["track_id"], original["track_id"])
        self.assertEqual(returned["reading_status"], "pending")
        expired, = self.frame([2.3], timestamp=3.0)
        self.assertNotEqual(expired["track_id"], original["track_id"])
        self.assertEqual(expired["reading_status"], "pending")

    def test_new_box_and_reset_clock_do_not_inherit_confirmed_pressure(self):
        for _ in range(3):
            original, = self.frame([2.3])

        moved, = self.frame([2.3], boxes=[(220, 20, 300, 100)])
        self.assertNotEqual(moved["track_id"], original["track_id"])
        self.assertEqual(moved["reading_status"], "pending")
        reset, = self.frame([2.3], timestamp=0.01)
        self.assertNotEqual(reset["track_id"], original["track_id"])
        self.assertEqual(reset["reading_status"], "pending")

    def test_ambiguous_overlap_cannot_borrow_a_confirmed_gauge(self):
        for _ in range(3):
            original, = self.frame([2.3])

        overlapped = self.frame(
            [2.3, 2.3],
            boxes=[(20, 20, 100, 100), (22, 20, 102, 100)],
        )

        for target in overlapped:
            self.assertNotEqual(target["track_id"], original["track_id"])
            self.assertEqual(target["reading_status"], "pending")

        following, = self.frame([2.3])
        self.assertEqual(following["reading_status"], "pending")

    def test_pending_overlay_uses_current_geometry_with_no_valid_pressure_label(self):
        result, = self.frame([2.3])
        raw = result["raw_gauge_reading"]
        self.assertTrue(raw["valid"])
        self.assertFalse(result["gauge_reading"]["valid"])

        with patch(
            "stream_target_detection.render_overlay",
            wraps=stream_target_detection.render_overlay,
        ) as renderer:
            overlay = self.processor.draw(self.image, self.result)

        self.assertIn("awaiting confirmation", self.predictor.labels[0][0])
        self.assertNotIn("2.3 bar", self.predictor.labels[0][0])
        self.assertFalse(renderer.call_args.args[1].valid)
        self.assertEqual(renderer.call_args.args[1].centre, (40.0, 40.0))
        self.assertFalse(np.shares_memory(overlay, self.image))
        np.testing.assert_array_equal(self.image, 0)

    def test_readings_must_share_one_tolerance_window_without_smoothing(self):
        for value in (2.0, 2.19, 2.38, 2.4):
            target, = self.frame([value])
            self.assertEqual(target["reading_status"], "pending")

        target, = self.frame([2.39])
        self.assertEqual(target["reading_status"], "confirmed")
        self.assertEqual(target["gauge_reading"]["value_bar"], 2.39)

    def test_processing_error_resets_confirmation(self):
        for site in ("predictor", "reader"):
            with self.subTest(site=site):
                for _ in range(3):
                    original, = self.frame([2.3])

                if site == "predictor":
                    failing_call = patch.object(
                        self.predictor,
                        "predict_detections",
                        side_effect=RuntimeError("test failure"),
                    )

                else:
                    failing_call = patch(
                        "stream_target_detection.read_gauge",
                        side_effect=RuntimeError("test failure"),
                    )

                with failing_call:
                    failed, = process_frame(self.packet, [self.processor])

                self.assertEqual(failed.failure_reason, "PROCESSOR_ERROR")
                recovered, = self.frame([2.3])
                self.assertNotEqual(recovered["track_id"], original["track_id"])
                self.assertEqual(recovered["reading_status"], "pending")

    def test_resized_frame_cannot_inherit_a_confirmed_reading(self):
        for _ in range(3):
            original, = self.frame([2.3])

        self.image = np.zeros((240, 480, 3), np.uint8)
        resized, = self.frame([2.3])
        self.assertNotEqual(resized["track_id"], original["track_id"])
        self.assertEqual(resized["reading_status"], "pending")

    def test_valve_state_change_keeps_identity_without_inheriting_pressure(self):
        opened, = self.frame([None], classes=["openValve"])
        closed, = self.frame([None], classes=["closedValve"])
        self.assertEqual(closed["track_id"], opened["track_id"])
        self.assertIsNone(closed["reading_status"])
        self.assertIsNone(closed["gauge_reading"])
        self.assertIsNone(closed["raw_gauge_reading"])

        gauge, = self.frame([2.3])
        self.assertNotEqual(gauge["track_id"], closed["track_id"])
        self.assertEqual(gauge["reading_status"], "pending")


if __name__ == "__main__":
    unittest.main()
