"""Check how detector crops become readings for one camera frame."""

import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from stream_target_detection import TargetDetectionProcessor, read_gauge
from stream_processing import FrameIdentity, process_frame
from stream_types import FramePacket


class TargetProcessingTests(unittest.TestCase):
    def test_reads_each_supplied_crop_and_keeps_unreadable_gauges(self):
        # The detector supplies two separate small gauges and one valve.
        first_crop = np.full((24, 24, 3), 50, dtype=np.uint8)
        second_crop = np.full((32, 32, 3), 150, dtype=np.uint8)
        detections = [
            {
                "class_name": "gauge",
                "box_xywhn": (0.15, 0.2, 0.15, 0.3),
                "bbox_xyxy_px": (12, 4, 36, 28),
                "gauge_crop": first_crop,
            },
            {
                "class_name": "gauge",
                "box_xywhn": (0.5, 0.5, 0.2, 0.4),
                "bbox_xyxy_px": (64, 24, 96, 56),
                "gauge_crop": second_crop,
            },
            {
                "class_name": "openValve",
                "box_xywhn": (0.8, 0.5, 0.2, 0.4),
                "bbox_xyxy_px": (112, 24, 144, 56),
                "gauge_crop": None,
            },
        ]
        predictor = Mock(spec=["predict_detections"])
        predictor.predict_detections.return_value = detections
        processor = TargetDetectionProcessor(predictor=predictor)

        source = np.zeros((80, 160, 3), dtype=np.uint8)
        original = source.copy()
        packet = FramePacket(source, 10, 1.0, 1.02)

        # Run the real reader, while recording which images the adapter passes in.
        with patch("stream_target_detection.read_gauge", wraps=read_gauge) as reader:
            result, = process_frame(packet, [processor])

        self.assertIsNone(result.failure_reason, result.error)
        self.assertEqual(result.identity, FrameIdentity.from_packet(packet))
        results = result.data["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual(reader.call_count, 2)
        self.assertIs(reader.call_args_list[0].args[0], first_crop)
        self.assertIs(reader.call_args_list[1].args[0], second_crop)

        # A pressure failure must retain the target, its box and the reason.
        for detection, origin in zip(results[:2], ([12, 4], [64, 24])):
            reading = detection["gauge_reading"]
            self.assertFalse(reading["valid"])
            self.assertIsNone(reading["value_bar"])
            self.assertEqual(reading["failure_reason"], "DIAL_TOO_SMALL")
            self.assertEqual(reading["metadata"]["crop_origin_px"], origin)

        self.assertEqual(results[2]["class_name"], "openValve")
        self.assertIsNone(results[2]["gauge_reading"])
        self.assertEqual(results[0]["bbox_xyxy_px"], (12, 4, 36, 28))
        self.assertEqual(results[1]["bbox_xyxy_px"], (64, 24, 96, 56))

        # Saved results contain measurements, never the image arrays themselves.
        saved = json.loads(json.dumps(result.to_record(), allow_nan=False))
        self.assertEqual(len(saved["data"]["results"]), 3)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(first_crop, 50)
        np.testing.assert_array_equal(second_crop, 150)

    def test_empty_following_frame_has_no_previous_detections(self):
        crop = np.zeros((24, 24, 3), dtype=np.uint8)
        predictor = Mock(spec=["predict_detections"])
        predictor.predict_detections.side_effect = [
            [{
                "class_name": "gauge",
                "box_xywhn": (0.5, 0.5, 0.5, 0.5),
                "bbox_xyxy_px": (12, 12, 36, 36),
                "gauge_crop": crop,
            }],
            [],
        ]
        processor = TargetDetectionProcessor(predictor=predictor)
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        first_packet = FramePacket(image, 20, 2.0, 2.02)
        next_packet = FramePacket(image, 21, 2.1, 2.12)

        first, = process_frame(first_packet, [processor])
        following, = process_frame(next_packet, [processor])

        self.assertIsNone(first.failure_reason, first.error)
        self.assertIsNone(following.failure_reason, following.error)
        self.assertEqual(len(first.data["results"]), 1)
        self.assertEqual(following.data["results"], [])
        self.assertEqual(following.annotations, ())
        self.assertEqual(following.identity, FrameIdentity.from_packet(next_packet))


if __name__ == "__main__":
    unittest.main()
