"""Check target drawing through the shared stream compositor."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stream_target_detection import TargetDetectionProcessor
from annotater import Annotater
from gauge_types import GaugeReadResult, FailureReason
from gcs_output import gcs_log_payloads
from stream_processing import FrameIdentity, ResultCompositor, process_frame
from stream_types import FramePacket
from tests.test_gcs_output import native_record


class ControlledPredictor:
    # Use Aidan's real drawing without loading a model for these tests.
    annotate_image = Annotater.annotate_image

    def __init__(self, detections):
        self.detections = detections
        self.inference_count = 0

    def predict_detections(self, image):
        self.inference_count += 1
        return self.detections


class TargetOverlayTests(unittest.TestCase):
    def setUp(self):
        self.source = np.zeros((320, 640, 3), dtype=np.uint8)
        self.packet = FramePacket(self.source, 10, 1.0, 1.02)
        self.accepted = GaugeReadResult.success(
            value_bar=5.0,
            angle_degrees=0.0,
            dial_quality=0.9,
            needle_quality=0.9,
            centre=(80.0, 80.0),
            tip=(80.0, 40.0),
            radius=None,
        )
        self.rejected = GaugeReadResult.failure(FailureReason.DIAL_TOO_SMALL)

        # Keep each gauge crop paired with its original frame position.
        self.predictor = ControlledPredictor([
            {
                "class_name": "gauge",
                "box_xywhn": (0.1875, 0.5, 0.25, 0.5),
                "bbox_xyxy_px": (40, 80, 200, 240),
                "gauge_crop": self.source[80:240, 40:200].copy(),
            },
            {
                "class_name": "gauge",
                "box_xywhn": (0.6875, 0.5, 0.25, 0.5),
                "bbox_xyxy_px": (360, 80, 520, 240),
                "gauge_crop": self.source[80:240, 360:520].copy(),
            },
            {
                "class_name": "openValve",
                "box_xywhn": (0.875, 0.875, 0.125, 0.125),
                "bbox_xyxy_px": (520, 260, 600, 300),
                "gauge_crop": None,
            },
        ])
        self.processor = TargetDetectionProcessor(predictor=self.predictor)

    def process_targets(self):
        # Give the accepted gauge three current readings before drawing its value.
        with patch(
            "stream_target_detection.read_gauge",
            side_effect=[self.accepted, self.rejected] * 3,
        ):
            for sequence in (8, 9, 10):
                packet = FramePacket(
                    self.source,
                    sequence,
                    sequence / 10,
                    sequence / 10 + 0.02,
                )
                result, = process_frame(packet, [self.processor])

        self.assertIsNone(result.failure_reason, result.error)
        self.assertTrue(
            callable(getattr(self.processor, "draw", None)),
            "TargetDetectionProcessor needs a draw() method",
        )
        return result

    def test_draws_both_gauges_without_changing_source(self):
        original = self.source.copy()
        result = self.process_targets()

        # Drawing must use saved readings without running the reader again.
        with patch(
            "stream_target_detection.read_gauge",
            side_effect=AssertionError("Drawing tried to read the gauge again"),
        ):
            output = ResultCompositor([self.processor]).compose(
                self.packet,
                [result],
            )

        self.assertEqual(self.predictor.inference_count, 3)
        self.assertTrue(np.any(output.image[80:240, 40:200]))
        self.assertTrue(np.any(output.image[80:240, 360:520]))
        self.assertTrue(np.any(output.image[280, 520]))

        # Crop centre (80, 80) and its needle must be shifted by the box origin.
        self.assertTrue(np.any(output.image[160, 120]))
        self.assertTrue(np.any(output.image[140, 120]))
        np.testing.assert_array_equal(output.image[160, 440], (0, 0, 0))

        # A valve must not inherit the preceding gauge's reading or geometry.
        self.assertIsNone(result.data["results"][2]["gauge_reading"])
        np.testing.assert_array_equal(output.image[280, 560], (0, 0, 0))
        np.testing.assert_array_equal(self.source, original)
        self.assertFalse(np.shares_memory(output.image, self.source))
        self.assertEqual(
            FrameIdentity.from_packet(output),
            FrameIdentity.from_packet(self.packet),
        )

        # Drawing objects must stay out of the JSON sent to the stream.
        saved = json.loads(json.dumps(result.to_record(), allow_nan=False))
        readings = saved["data"]["results"]
        self.assertEqual(readings[0]["gauge_reading"]["value_bar"], 5.0)
        self.assertEqual(
            readings[1]["gauge_reading"]["failure_reason"],
            "DIAL_TOO_SMALL",
        )

    def test_labels_show_pressure_or_unavailable(self):
        result = self.process_targets()

        # Draw real text while inspecting what the operator is told.
        with patch("annotater.cv2.putText", wraps=cv2.putText) as put_text:
            self.processor.draw(self.source, result)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(len(labels), 3)
        self.assertTrue(
            any("gauge" in label and "5.0 bar" in label for label in labels)
        )
        self.assertTrue(any("pressure unavailable" in label for label in labels))
        self.assertIn("openValve", labels)

    def test_preserves_previous_annotations_inside_gauge_region(self):
        result = self.process_targets()

        # Stand in for markings already drawn by another stream processor.
        overlay = self.source.copy()
        overlay[200:205, 115:125] = (255, 0, 255)
        overlay[5:10, 5:10] = (255, 0, 255)
        before = overlay.copy()

        drawn = self.processor.draw(overlay, result)

        np.testing.assert_array_equal(
            drawn[200:205, 115:125],
            before[200:205, 115:125],
        )
        np.testing.assert_array_equal(drawn[5:10, 5:10], before[5:10, 5:10])
        np.testing.assert_array_equal(overlay, before)
        self.assertFalse(np.shares_memory(drawn, overlay))
        self.assertTrue(np.any(drawn[160, 120]))

    def test_empty_following_frame_draws_no_previous_targets(self):
        first = self.process_targets()
        compositor = ResultCompositor([self.processor])
        compositor.compose(self.packet, [first])

        self.predictor.detections = []
        packet = FramePacket(self.source, 11, 1.1, 1.12)
        result, = process_frame(packet, [self.processor])
        output = compositor.compose(packet, [result])

        self.assertEqual(result.data["results"], [])
        self.assertEqual(result.annotations, ())
        np.testing.assert_array_equal(output.image, self.source)
        self.assertFalse(np.shares_memory(output.image, self.source))

    def test_empty_gauge_crop_retains_failure_without_crashing_drawing(self):
        # A box with no width has a result but no drawable gauge region.
        self.predictor.detections = [{
            "class_name": "gauge",
            "box_xywhn": (0.5, 0.5, 0.0, 0.25),
            "bbox_xyxy_px": (320, 120, 320, 200),
            "gauge_crop": self.source[120:200, 320:320].copy(),
        }]
        result, = process_frame(self.packet, [self.processor])
        self.assertIsNone(result.failure_reason, result.error)
        self.assertEqual(
            result.data["results"][0]["gauge_reading"]["failure_reason"],
            "INVALID_INPUT",
        )
        self.assertTrue(callable(getattr(self.processor, "draw", None)))

        output = ResultCompositor([self.processor]).compose(self.packet, [result])

        self.assertEqual(output.image.shape, self.source.shape)
        self.assertTrue(np.any(output.image))
        np.testing.assert_array_equal(self.source, 0)


class GaugeReaderIntegrationTests(unittest.TestCase):
    def test_real_crop_confirms_pressure_and_draws_measured_geometry_in_frame(self):
        """Exercise reading, confirmation, drawing and GCS text with one real crop."""

        path = (
            Path(__file__).resolve().parents[2]
            / "Gauge Reader/tests/fixtures/gauge_offcentre_frame_258.png"
        )
        crop = cv2.imread(str(path))
        self.assertIsNotNone(crop)
        original_crop = crop.copy()
        height, width = crop.shape[:2]

        # Offset the photograph so a missed crop-to-frame translation is obvious.
        left, top = 380, 100
        source = np.zeros((500, 800, 3), dtype=np.uint8)
        source[top:top + height, left:left + width] = crop
        original_source = source.copy()
        predictor = ControlledPredictor([{
            "class_name": "gauge",
            "bbox_xyxy_px": (left, top, left + width, top + height),
            "box_xywhn": (
                (left + width / 2) / 800,
                (top + height / 2) / 500,
                width / 800,
                height / 500,
            ),
            "gauge_crop": crop,
        }])
        processor = TargetDetectionProcessor(predictor=predictor)
        compositor = ResultCompositor([processor])
        track_ids = []

        # Only model inference is replaced; every reading comes from the real reader.
        for sequence, status in enumerate(("pending", "pending", "confirmed"), 1):
            packet = FramePacket(
                source,
                sequence,
                sequence / 10,
                sequence / 10 + 0.02,
            )
            result, = process_frame(packet, [processor])
            self.assertIsNone(result.failure_reason, result.error)
            target, = result.data["results"]
            raw = target["raw_gauge_reading"]
            accepted = target["gauge_reading"]
            track_ids.append(target["track_id"])

            self.assertTrue(raw["valid"], raw["failure_reason"])
            self.assertGreaterEqual(raw["value_bar"], 1.8)
            self.assertLessEqual(raw["value_bar"], 2.5)
            # Manual reference points are in original crop pixels, not the resize.
            np.testing.assert_allclose(raw["centre"], (173.0, 160.0), atol=10.0)
            np.testing.assert_allclose(raw["tip"], (87.0, 136.0), atol=12.0)
            self.assertIsNone(raw["radius"])
            self.assertEqual(raw["metadata"]["crop_origin_px"], [left, top])
            self.assertEqual(raw["metadata"]["coordinate_frame"], "crop_pixels")
            self.assertEqual(target["reading_status"], status)
            self.assertEqual(accepted["centre"], raw["centre"])
            self.assertEqual(accepted["tip"], raw["tip"])
            self.assertIsNone(accepted["radius"])

            # Feed the saved processor evidence into the actual GCS formatter.
            record = native_record()
            record.update(result.to_record()["identity"])
            record["processors"] = [result.to_record()]
            payload, = gcs_log_payloads(record)
            self.assertEqual(payload["action"], "Gauge detected")

            if status == "confirmed":
                self.assertTrue(accepted["valid"])
                self.assertEqual(accepted["value_bar"], raw["value_bar"])
                self.assertTrue(payload["content"].startswith("Pressure: "))
                pressure_text = payload["content"].split(";", 1)[0]
                pressure_text = pressure_text.removeprefix("Pressure: ")
                pressure_text = pressure_text.removesuffix(" bar")
                self.assertAlmostEqual(
                    float(pressure_text),
                    raw["value_bar"],
                    delta=0.005,
                )
                colour = (0, 180, 0)

            else:
                self.assertFalse(accepted["valid"])
                self.assertIsNone(accepted["value_bar"])
                self.assertIn("awaiting confirmation", payload["content"])
                self.assertNotIn("Pressure:", payload["content"])
                colour = (0, 140, 255)

            output = compositor.compose(packet, [result])
            # These independently chosen pixels lie on the hub and pointer shaft.
            np.testing.assert_array_equal(
                output.image[top + 160, left + 173],
                colour,
            )
            np.testing.assert_array_equal(
                output.image[top + 148, left + 130],
                colour,
            )
            np.testing.assert_array_equal(output.image[160, 173], (0, 0, 0))
            np.testing.assert_array_equal(source, original_source)
            np.testing.assert_array_equal(crop, original_crop)
            self.assertFalse(np.shares_memory(output.image, source))
            json.dumps(record, allow_nan=False)

        self.assertEqual(len(set(track_ids)), 1)
        self.assertEqual(predictor.inference_count, 3)

    def test_missing_pivot_keeps_gauge_failure_and_following_valve(self):
        """A pointer-shaped component without a central hub must not lose targets."""

        # A thick off-centre counterweight supplies no eligible central hub peak.
        crop = np.full((512, 512, 3), 230, dtype=np.uint8)
        cv2.line(crop, (90, 256), (410, 256), (20, 20, 20), 5)
        cv2.circle(crop, (370, 256), 25, (20, 20, 20), -1)
        source = np.zeros((600, 1000, 3), dtype=np.uint8)
        source[40:552, 40:552] = crop
        original = source.copy()
        predictor = ControlledPredictor([
            {
                "class_name": "gauge",
                "bbox_xyxy_px": (40, 40, 552, 552),
                "box_xywhn": (0.296, 296 / 600, 0.512, 512 / 600),
                "gauge_crop": crop,
            },
            {
                "class_name": "openValve",
                "bbox_xyxy_px": (700, 300, 800, 400),
                "box_xywhn": (0.75, 350 / 600, 0.1, 100 / 600),
                "gauge_crop": None,
            },
        ])
        processor = TargetDetectionProcessor(predictor=predictor)
        packet = FramePacket(source, 1, 0.1, 0.12)

        result, = process_frame(packet, [processor])

        self.assertIsNone(result.failure_reason, result.error)
        gauge, valve = result.data["results"]
        raw = gauge["raw_gauge_reading"]
        self.assertEqual(gauge["reading_status"], "unavailable")
        self.assertFalse(raw["valid"])
        self.assertEqual(raw["failure_reason"], "NO_PIVOT")
        self.assertIsNone(raw["centre"])
        self.assertIsNone(raw["tip"])
        self.assertIsNone(raw["radius"])
        self.assertIsNone(raw["value_bar"])
        self.assertEqual(raw, gauge["gauge_reading"])
        self.assertEqual(raw["metadata"]["crop_origin_px"], [40, 40])
        self.assertEqual(valve["class_name"], "openValve")
        self.assertIsNone(valve["gauge_reading"])

        # Neither the rejected gauge nor its drawing may prevent the valve overlay.
        output = ResultCompositor([processor]).compose(packet, [result])
        np.testing.assert_array_equal(output.image[350, 700], (0, 255, 0))
        np.testing.assert_array_equal(output.image[296, 296], source[296, 296])
        np.testing.assert_array_equal(source, original)


if __name__ == "__main__":
    unittest.main()
