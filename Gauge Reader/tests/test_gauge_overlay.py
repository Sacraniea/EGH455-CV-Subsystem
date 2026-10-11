"""Tests for the visual explanation drawn over a gauge candidate image."""

import unittest

import numpy as np

from gauge_overlay import render_overlay
from gauge_types import FailureReason, GaugeReadResult

from unittest.mock import patch


class GaugeOverlayTests(unittest.TestCase):
    """Describe the externally visible behaviour of the overlay renderer."""

    def setUp(self) -> None:
        """Create a simple image whose contents make comparison easy."""

        self.image = np.full(
            shape=(120, 120, 3),
            fill_value=255,
            dtype=np.uint8,
        )

    def test_valid_overlay_is_a_changed_copy(self) -> None:
        """Rendering must annotate a copy without changing the input image"""

        # Keep the original pixels to catch accidental drawing on the input.
        before = self.image.copy()

        result = GaugeReadResult.success(
            value_bar=2.5,
            angle_degrees=292.5,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(60.0, 60.0),
            tip=(30.0, 35.0),
            radius=None,
        )

        overlay = render_overlay(
            image=self.image,
            result=result,
        )

        np.testing.assert_array_equal(self.image, before)

        self.assertFalse(np.array_equal(self.image, overlay))

    @patch("gauge_overlay.cv2.putText")
    def test_invalid_overlay_names_failure_without_bar_value(
        self,
        put_text,
    ) -> None:
        """A rejected result must explain the failure without inventing a value."""

        # A missing needle should show a reason, with no pressure value.
        result = GaugeReadResult.failure(
            FailureReason.NO_NEEDLE,
        )

        render_overlay(
            image=self.image,
            result=result,
        )

        # Inspect the text sent to OpenCV without depending on font pixels.
        put_text.assert_called_once()

        label = put_text.call_args.args[1]

        self.assertIn("NO_NEEDLE", label)
        self.assertNotIn("bar", label.lower())

    @patch("gauge_overlay.cv2.putText")
    def test_valid_overlay_reports_pressure_without_a_confidence_claim(
        self,
        put_text,
    ) -> None:
        """Keep the crop label short; detailed direction evidence stays in JSON."""

        result = GaugeReadResult.success(
            value_bar=2.5,
            angle_degrees=292.5,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(60.0, 60.0),
            tip=(30.0, 35.0),
            radius=None,
        )

        render_overlay(
            image=self.image,
            result=result,
        )

        # Inspect the text sent to OpenCV without depending on font pixels.
        put_text.assert_called_once()

        label = put_text.call_args.args[1]

        self.assertEqual(label, "2.5 bar")

    def test_valid_overlay_draws_needle_without_dial_radius(self) -> None:
        """The measured pivot and tip must be visible without a fitted dial."""

        result = GaugeReadResult.success(
            value_bar=5.0,
            angle_degrees=270.0,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(60.0, 80.0),
            tip=(30.0, 80.0),
            radius=None,
        )

        overlay = render_overlay(
            self.image,
            result,
        )

        # Check that accepted geometry uses the green status colour.
        centre_pixel = overlay[80, 60]

        # This pixel is away from the pivot and label, so only the needle changes it.
        midpoint_pixel = overlay[80, 45]

        np.testing.assert_array_equal(centre_pixel, (0, 180, 0))
        np.testing.assert_array_equal(midpoint_pixel, (0, 180, 0))

    def test_detected_pivot_is_visible_without_needle_or_radius(self) -> None:
        """A measured pivot must remain visible when no tip was accepted."""

        result = GaugeReadResult.failure(
            FailureReason.NO_NEEDLE,
            dial_quality=0.8,
            centre=(60.0, 80.0),
            radius=None,
        )

        overlay = render_overlay(self.image, result)

        # The label cannot change this pixel; the pivot marker must do it.
        centre_pixel = overlay[80, 60]
        np.testing.assert_array_equal(centre_pixel, (0, 140, 255))

    def test_hiding_label_preserves_pivot_and_needle(self) -> None:
        before = self.image.copy()
        result = GaugeReadResult.success(
            value_bar=5.0,
            angle_degrees=0.0,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(60.0, 80.0),
            tip=(60.0, 55.0),
            radius=None,
        )

        overlay = render_overlay(self.image, result, show_label=False)

        # The dial sits below the text area, which must remain completely untouched.
        np.testing.assert_array_equal(overlay[:40], before[:40])

        # Check the pivot and needle still appear when the reading text is hidden.
        for row, column in ((80, 60), (65, 60)):
            with self.subTest(pixel=(row, column)):
                self.assertFalse(
                    np.array_equal(overlay[row, column], before[row, column])
                )

        np.testing.assert_array_equal(self.image, before)
        self.assertFalse(np.shares_memory(overlay, self.image))

    def test_hiding_rejection_label_keeps_geometry_without_inventing_a_needle(self) -> None:
        before = self.image.copy()
        result = GaugeReadResult.failure(
            FailureReason.NO_NEEDLE,
            dial_quality=0.8,
            centre=(60.0, 80.0),
            radius=None,
        )

        overlay = render_overlay(self.image, result, show_label=False)

        # Draw neither a rejection message nor a pressure value in the text area.
        np.testing.assert_array_equal(overlay[:40], before[:40])
        self.assertFalse(np.array_equal(overlay[80, 60], before[80, 60]))

        # A missing needle leaves pixels away from the pivot unchanged.
        np.testing.assert_array_equal(overlay[65, 60], before[65, 60])
        np.testing.assert_array_equal(self.image, before)

    def test_rejected_reading_retains_pointer_geometry_without_radius(self) -> None:
        """A rejected angle still shows the measured pointer in orange."""

        result = GaugeReadResult.failure(
            FailureReason.ANGLE_OUTSIDE_SCALE,
            centre=(60.0, 80.0),
            tip=(30.0, 80.0),
            radius=None,
        )

        overlay = render_overlay(
            self.image,
            result,
            show_label=False,
        )

        np.testing.assert_array_equal(overlay[80, 60], (0, 140, 255))
        np.testing.assert_array_equal(overlay[80, 45], (0, 140, 255))

    def test_legacy_radius_does_not_draw_an_outer_circle(self) -> None:
        """An old result's optional radius must not suggest a fitted dial."""

        result = GaugeReadResult.failure(
            FailureReason.NO_NEEDLE,
            centre=(60.0, 80.0),
            radius=30.0,
        )

        overlay = render_overlay(
            self.image,
            result,
            show_label=False,
        )

        # These pixels lie on the old circle, well outside the pivot marker.
        for row, column in ((50, 60), (80, 90), (110, 60), (80, 30)):
            with self.subTest(pixel=(row, column)):
                np.testing.assert_array_equal(
                    overlay[row, column],
                    self.image[row, column],
                )

    def test_grayscale_input_is_converted_to_bgr(self) -> None:
        """Status colours require a three-channel output image."""

        grayscale = np.full(
            shape=(120, 120),
            fill_value=255,
            dtype=np.uint8,
        )

        result = GaugeReadResult.failure(
            FailureReason.NO_DIAL,
        )

        overlay = render_overlay(grayscale, result)

        # Confirm rendering leaves the original greyscale array as a single channel.
        self.assertEqual(grayscale.shape, (120, 120))

        # Confirm the returned overlay has three channels for the status colours.
        self.assertEqual(overlay.shape, (120, 120, 3))

    def test_large_crop_uses_visible_pivot_marker(self) -> None:
        """The pivot marker must scale with the crop when radius is absent."""

        large_image = np.full(
            shape=(800, 800, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        result = GaugeReadResult.failure(
            FailureReason.NO_NEEDLE,
            dial_quality=0.8,
            centre=(400.0, 400.0),
            radius=None,
        )

        overlay = render_overlay(
            large_image,
            result,
        )

        # Sample seven pixels from the pivot to check that its marker scales up.
        marker_edge_pixel = overlay[400, 407]
        original_edge_pixel = large_image[400, 407]

        self.assertFalse(
            np.array_equal(
                marker_edge_pixel,
                original_edge_pixel,
            )
        )

    def test_large_crop_uses_scaled_needle_strokes(self) -> None:
        """The pointer stroke must remain visible on larger images."""

        large_image = np.full(
            shape=(800, 800, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        result = GaugeReadResult.success(
            value_bar=5.0,
            angle_degrees=270.0,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(400.0, 400.0),
            tip=(100.0, 400.0),
            radius=None,
        )

        overlay = render_overlay(
            large_image,
            result,
        )

        # Sample beside the ray to check that its stroke scales with the crop.
        ray_edge_pixel = overlay[404, 275]
        original_ray_pixel = large_image[404, 275]

        self.assertFalse(
            np.array_equal(
                ray_edge_pixel,
                original_ray_pixel,
            )
        )


if __name__ == "__main__":
    unittest.main()
