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
