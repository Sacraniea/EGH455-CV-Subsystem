"""Check crop reading, confirmation and drawing without loading the YOLO model."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from detection_targets import GAUGE_DIRECTORY, TargetDetectionProcessor, read_gauge
from annotater import Annotater
from detection_pipeline import FrameIdentity, ResultCompositor, process_frame
from gcs_output import gcs_log_payloads
from NetGear_config import build_frame_record
from stream_types import FramePacket


class ControlledPredictor:
    """Supply known model outputs while retaining the existing target renderer."""

    def __init__(self, detections):
        self.detections = detections
        self.inferences = 0
        self.labels = []

    def predict_detections(self, image):
        self.inferences += 1
        return self.detections

    def annotate_image(self, image, labels):
        self.labels = list(labels)
        return Annotater.annotate_image(self, image, labels)


def prediction(name, box, crop, image):
    left, top, right, bottom = box
    height, width = image.shape[:2]
    return {
        "class_name": name,
        "bbox_xyxy_px": box,
        "box_xywhn": (
            (left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height,
        ),
        "gauge_crop": crop,
    }


def frame_record(packet, result):
    return build_frame_record(packet, (result,), run_id = "target-test", source_kind = "camera")


class TargetDetectionTests(unittest.TestCase):
    def test_reads_each_supplied_crop_once_and_retains_rejected_gauges_and_valve(self):
        image = np.zeros((80, 160, 3), np.uint8)
        crops = [np.full((24, 24, 3), 50, np.uint8), np.full((32, 32, 3), 150, np.uint8)]
        predictor = ControlledPredictor([
            prediction("gauge", (12, 4, 36, 28), crops[0], image),
            prediction("gauge", (64, 24, 96, 56), crops[1], image),
            prediction("openValve", (112, 24, 144, 56), None, image),
        ])
        processor = TargetDetectionProcessor(predictor = predictor)
        packet = FramePacket(image, 10, 1.0, 1.02)

        with patch("detection_targets.read_gauge", wraps = read_gauge) as reader:
            result, = process_frame(packet, (processor,))

        self.assertIsNone(result.failure_reason, result.error)
        self.assertEqual(result.identity, FrameIdentity.from_packet(packet))
        self.assertEqual(reader.call_count, 2)
        for call, crop in zip(reader.call_args_list, crops):
            self.assertIs(call.args[0], crop)
        first, second, valve = result.data["results"]
        for gauge, origin in ((first, [12, 4]), (second, [64, 24])):
            reading = gauge["gauge_reading"]
            self.assertFalse(reading["valid"])
            self.assertIsNone(reading["value_bar"])
            self.assertEqual(reading["failure_reason"], "DIAL_TOO_SMALL")
            self.assertEqual(reading["metadata"]["crop_origin_px"], origin)
        self.assertEqual(valve["class_name"], "openValve")
        self.assertIsNone(valve["gauge_reading"])
        self.assertEqual(predictor.inferences, 1)
        self.assertEqual(len(json.loads(json.dumps(result.to_record(), allow_nan = False))["data"]["results"]), 3)
        np.testing.assert_array_equal(image, 0)
        np.testing.assert_array_equal(crops[0], 50)
        np.testing.assert_array_equal(crops[1], 150)

    def test_real_crop_confirms_after_three_readings_and_draws_all_targets_without_rereading(self):
        crop = cv2.imread(str(GAUGE_DIRECTORY / "tests/fixtures/gauge_offcentre_frame_258.png"))
        self.assertIsNotNone(crop)
        original_crop = crop.copy()
        height, width = crop.shape[:2]
        left, top = 50, 80
        image = np.zeros((500, 1000, 3), np.uint8)
        image[top:top + height, left:left + width] = crop
        original = image.copy()
        predictor = ControlledPredictor([
            prediction("gauge", (left, top, left + width, top + height), crop, image),
            prediction("gauge", (780, 100, 812, 132), np.zeros((32, 32, 3), np.uint8), image),
            prediction("openValve", (780, 300, 900, 420), None, image),
        ])
        processor = TargetDetectionProcessor(predictor = predictor)
        tracks = []

        for sequence, status in enumerate(("pending", "pending", "confirmed"), 1):
            packet = FramePacket(image, sequence, sequence / 10, sequence / 10 + 0.02)
            result, = process_frame(packet, (processor,))
            self.assertIsNone(result.failure_reason, result.error)
            gauge, rejected, valve = result.data["results"]
            raw, accepted = gauge["raw_gauge_reading"], gauge["gauge_reading"]
            tracks.append(gauge["track_id"])
            self.assertTrue(raw["valid"], raw["failure_reason"])
            self.assertGreaterEqual(raw["value_bar"], 1.8)
            self.assertLessEqual(raw["value_bar"], 2.5)
            np.testing.assert_allclose(raw["centre"], (173, 160), atol = 10)
            np.testing.assert_allclose(raw["tip"], (87, 136), atol = 12)
            self.assertEqual(gauge["reading_status"], status)
            self.assertEqual(accepted["valid"], status == "confirmed")
            self.assertEqual(rejected["gauge_reading"]["failure_reason"], "DIAL_TOO_SMALL")
            self.assertIsNone(valve["gauge_reading"])
            payloads = gcs_log_payloads(frame_record(packet, result))
            self.assertEqual([payload["action"] for payload in payloads], ["Gauge detected", "Gauge detected", "Valve detected"])
            self.assertIn("Pressure unavailable", payloads[1]["content"])
            self.assertIn("Open valve; image position (840, 360) px", payloads[2]["content"])
            if status == "confirmed":
                self.assertEqual(accepted["value_bar"], raw["value_bar"])
                pressure = payloads[0]["content"].split(";", 1)[0]
                self.assertAlmostEqual(float(pressure.removeprefix("Pressure: ").removesuffix(" bar")), raw["value_bar"], delta = 0.005)
            else:
                self.assertIsNone(accepted["value_bar"])
                self.assertIn("awaiting confirmation", payloads[0]["content"])

        with patch("detection_targets.read_gauge", side_effect = AssertionError("Drawing reread a gauge")):
            output = ResultCompositor((processor,)).compose(packet, (result,))
        self.assertEqual(predictor.inferences, 3)
        self.assertEqual(len(set(tracks)), 1)
        self.assertEqual([label for label, _ in predictor.labels], ["gauge | 2.3 bar", "gauge | pressure unavailable", "openValve"])
        np.testing.assert_array_equal(output.image[top + 160, left + 173], (0, 180, 0))
        np.testing.assert_array_equal(output.image[top + 148, left + 130], (0, 180, 0))
        self.assertTrue(output.image[116, 780].any())
        self.assertTrue(output.image[360, 780].any())
        np.testing.assert_array_equal(output.image[360, 840], 0)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(crop, original_crop)
        self.assertFalse(np.shares_memory(output.image, image))
        self.assertEqual(FrameIdentity.from_packet(output), FrameIdentity.from_packet(packet))

    def test_empty_following_frame_has_no_stale_targets_annotations_or_gcs_logs(self):
        image = np.zeros((80, 160, 3), np.uint8)
        predictor = ControlledPredictor([
            prediction("gauge", (12, 4, 36, 28), image[4:28, 12:36].copy(), image),
        ])
        processor = TargetDetectionProcessor(predictor = predictor)
        first, = process_frame(FramePacket(image, 1, 0.1, 0.12), (processor,))
        self.assertEqual(len(first.data["results"]), 1)
        predictor.detections = []
        packet = FramePacket(image, 2, 0.2, 0.22)
        result, = process_frame(packet, (processor,))
        output = ResultCompositor((processor,)).compose(packet, (result,))

        self.assertIsNone(result.failure_reason, result.error)
        self.assertEqual(result.data["results"], [])
        self.assertEqual(result.annotations, ())
        self.assertEqual(gcs_log_payloads(frame_record(packet, result)), [])
        self.assertEqual(predictor.inferences, 2)
        np.testing.assert_array_equal(output.image, image)
        self.assertFalse(np.shares_memory(output.image, image))


if __name__ == "__main__":
    unittest.main()
