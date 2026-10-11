"""Check pressure, rejection evidence and crop coordinates without a dial search."""

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from gauge_reader import read_gauge
from gauge_types import FailureReason, GaugeCalibration


def pointer_image(angle_degrees, *, shape=(320, 320)):
    """Draw a pointer, a short counterweight and a bright hub, with no rim."""

    image = np.full((*shape, 3), 230, dtype=np.uint8)
    pivot = np.array([shape[1] / 2, shape[0] / 2])
    radians = np.radians(angle_degrees)
    direction = np.array([np.sin(radians), -np.cos(radians)])
    across = np.array([-direction[1], direction[0]])

    pointer = np.array(
        [
            pivot + across * 4,
            pivot + direction * 106,
            pivot - across * 4,
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pointer], (20, 20, 20))
    cv2.line(
        image,
        tuple(pivot.astype(int)),
        tuple((pivot - direction * 40).astype(int)),
        (20, 20, 20),
        thickness=13,
    )
    cv2.circle(image, tuple(pivot.astype(int)), 12, (20, 20, 20), -1)
    cv2.circle(image, tuple(pivot.astype(int)), 5, (190, 180, 130), -1)
    return image


class GaugeReaderTests(unittest.TestCase):
    def test_reads_pressure_without_a_visible_rim(self):
        # These directions are 54, 135 and 216 degrees past the 225-degree zero.
        for angle, expected_bar in ((279.0, 2.0), (0.0, 5.0), (81.0, 8.0)):
            with self.subTest(angle=angle):
                result = read_gauge(pointer_image(angle))

                self.assertTrue(result.valid, result.failure_reason)
                self.assertAlmostEqual(result.value_bar, expected_bar, delta=0.08)
                self.assertIsNone(result.failure_reason)
                self.assertIsNone(result.radius)
                self.assertEqual(result.dial_quality, 0.0)
                self.assertGreater(result.needle_quality, 0.0)

    def test_saved_crop_uses_the_hub_as_pivot_and_the_long_pointer(self):
        path = (
            Path(__file__).parent
            / "fixtures"
            / "gauge_offcentre_frame_258.png"
        )
        crop = cv2.imread(str(path))
        self.assertIsNotNone(crop)

        result = read_gauge(crop)

        self.assertTrue(result.valid, result.failure_reason)
        # Manual reference on the original photograph, independent of the algorithm.
        np.testing.assert_allclose(result.centre, (173.0, 160.0), atol=10.0)
        self.assertGreaterEqual(result.value_bar, 1.8)
        self.assertLessEqual(result.value_bar, 2.5)
        self.assertLess(result.tip[0], result.centre[0])
        self.assertIsNone(result.radius)

    def test_rectangular_crop_returns_source_coordinates_without_changing_input(self):
        image = pointer_image(279.0, shape=(240, 400))
        original = image.copy()
        metadata = {'candidate_id': 'gauge-9'}

        result = read_gauge(image, metadata=metadata)

        self.assertTrue(result.valid, result.failure_reason)
        np.testing.assert_allclose(result.centre, (200.0, 120.0), atol=3.0)
        self.assertLess(result.tip[0], 110.0)
        self.assertGreater(result.tip[1], 95.0)
        self.assertLess(result.tip[1], 115.0)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(metadata, {'candidate_id': 'gauge-9'})
        self.assertEqual(result.metadata['candidate_id'], 'gauge-9')
        self.assertIsNot(result.metadata, metadata)
        self.assertIn('needle_detection', result.metadata)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_custom_calibration_changes_pressure_without_changing_direction(self):
        calibration = GaugeCalibration(minimum_bar=2.0, maximum_bar=14.0)

        result = read_gauge(pointer_image(0.0), calibration=calibration)

        self.assertTrue(result.valid, result.failure_reason)
        self.assertAlmostEqual(result.value_bar, 8.0, delta=0.08)

    def test_outside_scale_retains_pivot_and_direction_but_withholds_pressure(self):
        result = read_gauge(
            pointer_image(180.0),
            metadata={"candidate_id": "below-scale"},
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.ANGLE_OUTSIDE_SCALE)
        self.assertIsNone(result.value_bar)
        self.assertIsNotNone(result.centre)
        self.assertIsNotNone(result.tip)
        self.assertAlmostEqual(result.angle_degrees, 180.0, delta=2.0)
        self.assertIsNone(result.radius)
        self.assertEqual(result.metadata['candidate_id'], 'below-scale')

    def test_invalid_and_unreadable_crops_keep_failure_evidence(self):
        cases = (
            (None, FailureReason.INVALID_INPUT),
            (np.zeros((30, 90, 3), np.uint8), FailureReason.DIAL_TOO_SMALL),
            (np.full((320, 320, 3), 255, np.uint8), FailureReason.DIAL_TOO_BLURRED),
        )

        for image, reason in cases:
            with self.subTest(reason=reason):
                result = read_gauge(image, metadata={'frame': 123})

                self.assertFalse(result.valid)
                self.assertIsNone(result.value_bar)
                self.assertEqual(result.failure_reason, reason)
                self.assertEqual(result.metadata['frame'], 123)
                self.assertIsNone(result.radius)

    def test_sharp_tick_marks_without_a_needle_are_rejected(self):
        image = np.full((320, 320, 3), 230, np.uint8)
        cv2.circle(image, (160, 160), 12, (0, 0, 0), -1)
        for y in (40, 60, 80):
            cv2.line(image, (40, y), (65, y), (0, 0, 0), 3)

        result = read_gauge(image)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.NO_NEEDLE)
        self.assertIsNone(result.value_bar)


if __name__ == '__main__':
    unittest.main()
