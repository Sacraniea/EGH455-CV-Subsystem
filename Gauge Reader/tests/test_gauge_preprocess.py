"""Test crop validation, letterboxing, coordinate recovery, and blur evidence."""

import unittest

import numpy as np

from gauge_preprocess import preprocess_candidate
from gauge_types import FailureReason, ReaderConfig


class GaugePreprocessTests(unittest.TestCase):
    """Describe the required behaviour of candidate-crop preprocessing."""

    def setUp(self) -> None:
        """Create an independent default configuration for every test."""

        self.config = ReaderConfig()

    def test_rejects_non_image_and_unsupported_inputs(self):
        """Missing, empty, and unsupported arrays must fail closed."""

        invalid_inputs = {
            "none": None,
            "empty": np.array([], dtype=np.uint8),
            "two-channel": np.zeros((100, 100, 2), dtype=np.uint8),
            "empty-bgr": np.zeros((0, 100, 3), dtype=np.uint8),
        }

        # Check that every missing, empty or unsupported input returns INVALID_INPUT.
        for case_name, image in invalid_inputs.items():
            with self.subTest(case=case_name):
                outcome = preprocess_candidate(image, self.config)

                self.assertFalse(outcome.valid)
                self.assertEqual(
                    outcome.failure_reason,
                    FailureReason.INVALID_INPUT,
                )

    def test_rejects_crop_smaller_than_minimum_side(self):
        """Either source dimension being below 64 pixels is insufficient."""

        # One short side is enough to reject the crop.
        image = np.zeros((63, 100, 3), dtype=np.uint8)

        outcome = preprocess_candidate(image, self.config)

        self.assertFalse(outcome.valid)
        self.assertEqual(
            outcome.failure_reason,
            FailureReason.DIAL_TOO_SMALL,
        )

    def test_letterbox_preserves_source_coordinates(self):
        """A rectangular image must be padded rather than distorted."""

        image = np.full(
            (100, 200, 3),
            255,
            dtype=np.uint8,
        )

        outcome = preprocess_candidate(image, self.config)

        self.assertTrue(outcome.valid)
        self.assertIsNotNone(outcome.transform)
        self.assertIsNotNone(outcome.working_colour)
        self.assertIsNotNone(outcome.working_gray)

        # Check that both working images use the configured 512-pixel square canvas.
        self.assertEqual(
            outcome.working_colour.shape,
            (512, 512, 3),
        )
        self.assertEqual(
            outcome.working_gray.shape,
            (512, 512),
        )

        # For this 200x100 source, scale=2.56 and vertical padding=128.
        working_point = outcome.transform.to_working_point((50.0, 25.0))
        self.assertAlmostEqual(working_point[0], 128.0)
        self.assertAlmostEqual(working_point[1], 192.0)

        # Applying the inverse transform must recover the original point.
        source_point = outcome.transform.to_source_point(working_point)
        self.assertAlmostEqual(source_point[0], 50.0)
        self.assertAlmostEqual(source_point[1], 25.0)

    def test_supported_channel_formats_become_bgr(self):
        """Greyscale and BGRA inputs should become three-channel BGR."""

        supported_inputs = {
            "greyscale": np.full(
                (100, 100),
                127,
                dtype=np.uint8,
            ),
            "bgra": np.full(
                (100, 100, 4),
                255,
                dtype=np.uint8,
            ),
        }

        for case_name, image in supported_inputs.items():
            with self.subTest(case=case_name):
                outcome = preprocess_candidate(image, self.config)

                self.assertTrue(outcome.valid)
                self.assertIsNotNone(outcome.working_colour)
                self.assertEqual(
                    outcome.working_colour.shape,
                    (512, 512, 3),
                )

    def test_blur_variance_responds_to_source_detail(self):
        """A sharp pattern should have more edge evidence than a flat crop."""

        # A repeatable checkerboard supplies edges without an external fixture.
        rows, cols = np.indices((128, 128))
        checkerboard = (((rows // 8 + cols // 8) % 2) * 255).astype(np.uint8)

        sharp_image = np.repeat(
            checkerboard[:, :, np.newaxis],
            3,
            axis=2,
        )

        flat_image = np.full(
            (128, 128, 3),
            127,
            dtype=np.uint8,
        )

        sharp_outcome = preprocess_candidate(sharp_image, self.config)
        flat_outcome = preprocess_candidate(flat_image, self.config)

        # Accept both inputs here and verify the checkerboard has greater blur variance.
        self.assertTrue(sharp_outcome.valid)
        self.assertTrue(flat_outcome.valid)
        self.assertGreater(
            sharp_outcome.blur_variance,
            flat_outcome.blur_variance,
        )


if __name__ == "__main__":
    unittest.main()
