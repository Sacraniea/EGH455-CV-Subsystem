"""Check that attached scale markings cannot steer the measured needle."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from gauge_needle import detect_needle
from gauge_reader import read_gauge
from gauge_types import FailureReason, ReaderConfig
from test_gauge_needle import _pointed_needle


class GaugeRadialTests(unittest.TestCase):
    def test_thin_pointers_work_around_the_full_sweep_and_zero_wrap(self):
        for angle in (0, 1, 60, 120, 180, 240, 300, 359):
            with self.subTest(angle=angle):
                result = detect_needle(_pointed_needle(angle, width=8), ReaderConfig())

                self.assertTrue(result.valid, result.failure_reason)
                difference = abs((result.angle_degrees - angle + 180) % 360 - 180)
                self.assertLessEqual(difference, 2.0)

    def test_pointer_touching_scale_arc_keeps_its_direction(self):
        image = _pointed_needle(60)
        # Join a long printed arc to the pointer tip without changing the hand.
        cv2.ellipse(image, (256, 256), (165, 165), 0, -35, 150, 20, 5)

        result = detect_needle(image, ReaderConfig())

        self.assertTrue(result.valid, result.failure_reason)
        self.assertAlmostEqual(result.angle_degrees, 60.0, delta=3.0)
        np.testing.assert_allclose(result.pivot, (256, 256), atol=3.0)

    def test_attached_complete_scale_ring_does_not_become_a_large_hub(self):
        image = _pointed_needle(60)
        cv2.circle(image, (256, 256), 165, 20, 5)

        result = detect_needle(image, ReaderConfig())

        self.assertTrue(result.valid, result.failure_reason)
        self.assertAlmostEqual(result.angle_degrees, 60.0, delta=3.0)
        np.testing.assert_allclose(result.pivot, (256, 256), atol=3.0)
        self.assertLess(result.metadata['pivot_half_thickness_px'], 25.0)

    def test_two_separate_long_pointer_directions_are_ambiguous(self):
        for second_tip in ((109, 171), (142, 382)):
            with self.subTest(second_tip=second_tip):
                image = _pointed_needle(60)
                # Check a second hand beside the pointer and near its opposite end.
                cv2.line(image, (256, 256), second_tip, 20, 7)

                result = detect_needle(image, ReaderConfig())

                self.assertFalse(result.valid)
                self.assertEqual(result.failure_reason, FailureReason.AMBIGUOUS_NEEDLE)
                self.assertIsNone(result.angle_degrees)

    def test_separate_outer_tick_does_not_extend_the_counterweight(self):
        image = _pointed_needle(90)
        cv2.line(image, (40, 256), (160, 256), 20, 7)

        result = detect_needle(image, ReaderConfig())

        self.assertTrue(result.valid, result.failure_reason)
        self.assertAlmostEqual(result.angle_degrees, 90.0, delta=2.0)
        self.assertGreater(result.tip[0], result.pivot[0])


    def test_saved_connected_scale_crops_keep_the_visible_pointer_direction(self):
        # References were marked on the unannotated crop before running either
        # reader. Allow a few degrees for uncertainty in that visual estimate.
        cases = (
            (85, 321.5, (157.5, 149.0)),
            (313, 309.3, (151.0, 153.5)),
        )

        for sequence, expected_angle, expected_pivot in cases:
            with self.subTest(sequence=sequence):
                path = Path(__file__).parent / 'fixtures' / f'gauge_scale_frame_{sequence}.png'
                crop = cv2.imread(str(path))
                self.assertIsNotNone(crop)
                original = crop.copy()

                result = read_gauge(crop)

                self.assertTrue(result.valid, result.failure_reason)
                self.assertAlmostEqual(result.angle_degrees, expected_angle, delta=3.0)
                np.testing.assert_allclose(result.centre, expected_pivot, atol=3.0)
                np.testing.assert_array_equal(crop, original)

    def test_saved_ambiguous_crop_keeps_real_hub_without_inventing_pressure(self):
        path = Path(__file__).parent / 'fixtures' / 'gauge_scale_frame_131.png'
        crop = cv2.imread(str(path))
        self.assertIsNotNone(crop)

        result = read_gauge(crop)

        self.assertFalse(result.valid)
        self.assertIsNone(result.value_bar)
        self.assertIsNone(result.tip)
        np.testing.assert_allclose(result.centre, (201.5, 190.5), atol=3.0)


if __name__ == '__main__':
    unittest.main()
