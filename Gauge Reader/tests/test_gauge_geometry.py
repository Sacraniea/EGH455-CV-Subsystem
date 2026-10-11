"""Test gauge geometry and calibrated pressure conversion."""

import unittest

from gauge_geometry import (
    angle_from_points,
    normalise_angle_degrees,
    point_at_angle,
    pressure_from_angle,
)

from gauge_types import GaugeCalibration


class GaugeGeometryTests(unittest.TestCase):
    """Verify the gauge reader's clockwise clock-face convention."""

    def setUp(self):
        """Create the default 0-10 bar calibration before every test"""

        self.calibration = GaugeCalibration()

    def test_angles_are_normalised_to_one_revolution(self):
        """Angles must always be represented in the range [0, 360) degrees."""

        self.assertAlmostEqual(normalise_angle_degrees(360.0), 0.0)
        self.assertAlmostEqual(normalise_angle_degrees(-45.0), 315.0)
        self.assertAlmostEqual(normalise_angle_degrees(725.0), 5.0)

    def test_clock_face_angle_convention(self):
        """Image-space directions should produce clockwise gauge angles"""

        centre = (10.0, 10.0)

        # Check that points above, right, below and left map to 0, 90, 180 and 270.
        self.assertAlmostEqual(
            angle_from_points(centre, (10.0, 0.0)),
            0.0,
        )
        self.assertAlmostEqual(
            angle_from_points(centre, (20.0, 10.0)),
            90.0,
        )
        self.assertAlmostEqual(
            angle_from_points(centre, (10.0, 20.0)),
            180.0,
        )
        self.assertAlmostEqual(
            angle_from_points(centre, (0.0, 10.0)),
            270.0,
        )

    def test_identical_centre_and_tip_are_rejected(self):
        """A zero length needle has no direction and therefore no angle."""

        with self.assertRaises(ValueError):
            angle_from_points((10.0, 10.0), (10.0, 10.0))

    def test_default_scale_maps_zero_midpoint_and_full_scale(self):
        """Known scale positions should map to hand calculated pressures"""

        self.assertAlmostEqual(
            pressure_from_angle(225.0, self.calibration),
            0.0,
        )
        self.assertAlmostEqual(
            pressure_from_angle(0.0, self.calibration),
            5.0,
        )
        self.assertAlmostEqual(
            pressure_from_angle(135.0, self.calibration),
            10.0,
        )

    def test_lower_dead_zone_is_outside_scale(self):
        """Angles within the 90-degree dead zone must be rejected."""

        self.assertIsNone(
            pressure_from_angle(180.0, self.calibration),
        )

    def test_angles_just_outside_scale_use_the_endpoint_pressure(self):
        # Include the measured near-zero angles from the camera trial.
        for angle, expected in (
            (222.7, 0.0),
            (217.1, 0.0),
            (215.0, 0.0),
            (140.0, 10.0),
            (145.0, 10.0),
        ):
            with self.subTest(angle=angle):
                self.assertEqual(pressure_from_angle(angle, self.calibration), expected)

    def test_angles_beyond_endpoint_tolerance_remain_unavailable(self):
        for angle in (214.99, 145.01, 180.0):
            with self.subTest(angle=angle):
                self.assertIsNone(pressure_from_angle(angle, self.calibration))

    def test_angles_inside_scale_keep_their_calculated_pressure(self):
        # Readings inside the scale must not be rounded down or up to its ends.
        self.assertAlmostEqual(pressure_from_angle(234.0, self.calibration), 1 / 3)
        self.assertAlmostEqual(pressure_from_angle(126.0, self.calibration), 29 / 3)

    def test_endpoint_tolerance_follows_a_rotated_custom_scale(self):
        calibration = GaugeCalibration(
            minimum_bar=2.0,
            maximum_bar=14.0,
            zero_angle_degrees=5.0,
            full_scale_angle_degrees=275.0,
        )

        # The zero mark crosses 360 degrees; the full-scale mark is also rotated.
        self.assertEqual(pressure_from_angle(358.0, calibration), 2.0)
        self.assertEqual(pressure_from_angle(282.0, calibration), 14.0)
        self.assertIsNone(pressure_from_angle(354.0, calibration))
        self.assertIsNone(pressure_from_angle(286.0, calibration))

    def test_endpoint_tolerance_can_be_tightened_or_disabled(self):
        tighter = GaugeCalibration(endpoint_tolerance_degrees=3.0)
        strict = GaugeCalibration(endpoint_tolerance_degrees=0.0)

        self.assertEqual(pressure_from_angle(222.0, tighter), 0.0)
        self.assertIsNone(pressure_from_angle(221.9, tighter))
        self.assertIsNone(pressure_from_angle(224.9, strict))
        self.assertEqual(pressure_from_angle(225.0, strict), 0.0)

    def test_point_round_trip_preserves_angles(self):
        """Generating a point and measuring it again should preserve direction."""

        # Place a tip 80 pixels away, then recover its original 315-degree direction.
        centre = (200.0, 150.0)
        tip = point_at_angle(
            centre=centre,
            distance=80.0,
            angle_degrees=315.0,
        )

        self.assertAlmostEqual(
            angle_from_points(centre, tip),
            315.0,
        )


if __name__ == "__main__":
    unittest.main()
