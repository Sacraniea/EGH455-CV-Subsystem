"""Exercise needle and pivot detection without supplying any dial geometry."""

import unittest

import cv2
import numpy as np

from gauge_needle import detect_needle
from gauge_types import FailureReason, ReaderConfig


def _pointed_needle(angle_degrees, pivot=(256, 256), width=12):
    """Draw a long pointed hand, shorter thick tail, and bright enclosed hub."""
    image = np.full((512, 512), 230, dtype=np.uint8)
    centre = np.array(pivot, dtype=float)
    angle = np.radians(angle_degrees)
    direction = np.array([np.sin(angle), -np.cos(angle)])
    perpendicular = np.array([-direction[1], direction[0]])
    outline = np.array(
        [
            centre + perpendicular * (width / 2),
            centre + direction * 170,
            centre - perpendicular * (width / 2),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [outline], 20)
    cv2.line(
        image,
        pivot,
        tuple(np.int32(centre - direction * 64)),
        20,
        19,
    )
    cv2.circle(image, pivot, 18, 20, -1)
    cv2.circle(image, pivot, 7, 210, -1)
    return image


class GaugeNeedleTests(unittest.TestCase):
    def setUp(self):
        self.config = ReaderConfig()

    def test_recovers_pointed_needle_and_bright_hub_without_dial(self):
        image = _pointed_needle(60)
        original = image.copy()
        try:
            result = detect_needle(image, self.config)
        except TypeError as error:
            self.fail(
                "Needle detection must accept grayscale pixels and config "
                f"without dial geometry: {error}"
            )

        self.assertTrue(result.valid)
        self.assertIsNone(result.failure_reason)
        self.assertAlmostEqual(result.angle_degrees, 60.0, delta=2.0)
        np.testing.assert_allclose(result.pivot, (256, 256), atol=2.0)
        self.assertGreater(result.tip[0], 380)
        self.assertLess(result.tip[1], 185)
        self.assertGreater(result.quality, 0.4)
        self.assertLess(result.quality, 0.7)
        self.assertGreater(result.metadata["component_area"], 1000)
        self.assertEqual(result.metadata["candidate_components"], 1)
        np.testing.assert_array_equal(image, original)

    def test_opposite_direction_selects_pointer_instead_of_counterweight(self):
        result = detect_needle(_pointed_needle(240), self.config)

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.angle_degrees, 240.0, delta=2.0)
        self.assertLess(result.tip[0], 130)
        self.assertGreater(result.tip[1], 325)
        np.testing.assert_allclose(result.pivot, (256, 256), atol=2.0)

    def test_wide_needle_can_use_a_lower_elongation_threshold(self):
        config = ReaderConfig(
            minimum_needle_elongation=2.0,
            minimum_needle_direction_ratio=1.2,
        )

        result = detect_needle(_pointed_needle(60, width=80), config)

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.angle_degrees, 60.0, delta=2.0)

    def test_pivot_can_be_displaced_from_crop_centre(self):
        result = detect_needle(
            _pointed_needle(270, pivot=(280, 240)),
            self.config,
        )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.angle_degrees, 270.0, delta=1.0)
        np.testing.assert_allclose(result.pivot, (280, 240), atol=2.0)

    def test_angle_outside_pressure_scale_is_still_a_detected_needle(self):
        result = detect_needle(_pointed_needle(180), self.config)

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.angle_degrees, 180.0, delta=1.0)
        self.assertIsNone(result.failure_reason)

    def test_uniform_images_have_no_needle(self):
        for brightness in (0, 127, 255):
            with self.subTest(brightness=brightness):
                image = np.full((512, 512), brightness, dtype=np.uint8)
                result = detect_needle(image, self.config)

                self.assertFalse(result.valid)
                self.assertEqual(result.failure_reason, FailureReason.NO_NEEDLE)
                self.assertIsNone(result.pivot)
                self.assertIsNone(result.tip)
                self.assertIsNone(result.angle_degrees)
                self.assertEqual(result.quality, 0.0)

    def test_hub_and_disconnected_scale_ticks_without_hand_are_rejected(self):
        image = np.full((512, 512), 230, dtype=np.uint8)
        cv2.circle(image, (256, 256), 18, 20, -1)
        for angle in range(0, 360, 30):
            radians = np.radians(angle)
            direction = np.array([np.sin(radians), -np.cos(radians)])
            start = tuple(np.int32([256, 256] + direction * 180))
            end = tuple(np.int32([256, 256] + direction * 210))
            cv2.line(image, start, end, 20, 5)

        result = detect_needle(image, self.config)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.NO_NEEDLE)
        self.assertIsNone(result.pivot)

    def test_equal_opposite_ends_reject_direction_but_retain_pivot(self):
        image = np.full((512, 512), 230, dtype=np.uint8)
        cv2.line(image, (106, 256), (406, 256), 20, 5)
        cv2.circle(image, (256, 256), 18, 20, -1)

        result = detect_needle(image, self.config)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.AMBIGUOUS_NEEDLE)
        np.testing.assert_allclose(result.pivot, (256, 256), atol=2.0)
        self.assertIsNone(result.tip)
        self.assertIsNone(result.angle_degrees)
        self.assertAlmostEqual(result.quality, 0.0, delta=0.02)
        self.assertAlmostEqual(result.metadata["direction_ratio"], 1.0, delta=0.02)

    def test_thin_central_component_without_hub_reports_no_pivot(self):
        image = np.full((512, 512), 230, dtype=np.uint8)
        cv2.line(image, (150, 256), (360, 256), 20, 3)

        result = detect_needle(image, self.config)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.NO_PIVOT)
        self.assertIsNone(result.pivot)
        self.assertIsNone(result.tip)
        self.assertIsNone(result.angle_degrees)
        self.assertEqual(result.metadata["candidate_components"], 1)
        self.assertGreater(result.metadata["max_central_half_thickness_px"], 0)

    def test_hub_beyond_central_search_without_eligible_peak_reports_no_pivot(self):
        image = np.full((512, 512), 230, dtype=np.uint8)
        cv2.line(image, (90, 256), (410, 256), 20, 5)
        cv2.circle(image, (370, 256), 25, 20, -1)

        result = detect_needle(image, self.config)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.NO_PIVOT)
        self.assertIsNone(result.pivot)
        self.assertIsNone(result.tip)
        self.assertIsNone(result.angle_degrees)
        self.assertEqual(result.metadata["candidate_components"], 1)
        self.assertGreater(result.metadata["max_central_half_thickness_px"], 0)

    def test_component_connected_to_image_border_is_rejected(self):
        image = _pointed_needle(270)
        cv2.line(image, (0, 256), (256, 256), 20, 6)

        result = detect_needle(image, self.config)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.NO_NEEDLE)
        self.assertIsNone(result.pivot)

    def test_component_far_from_crop_centre_is_rejected(self):
        image = _pointed_needle(270, pivot=(400, 110))

        result = detect_needle(image, self.config)

        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, FailureReason.NO_NEEDLE)
        self.assertIsNone(result.pivot)


if __name__ == "__main__":
    unittest.main()
