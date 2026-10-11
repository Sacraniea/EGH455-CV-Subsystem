import unittest

from gauge_types import (
    FailureReason,
    GaugeReadResult,
    GaugeCalibration,
    ReaderConfig,
)


class GaugeReadResultTest(unittest.TestCase):

    def test_success_contains_value_and_no_failure(self):
        # Build an accepted reading and check that its JSON fields contain the pressure.
        result = GaugeReadResult.success(
            value_bar=2.5,
            angle_degrees=292.5,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(100.0, 101.0),
            tip=(70.0, 70.0),
            radius=80.0,
            metadata={"candidate_id": "gauge-1"},
        )

        self.assertTrue(result.valid)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.to_dict()["value_bar"], 2.5)

    def test_failure_has_no_pressure_value(self):
        # Retain the detected circle even when no pressure can be reported.
        result = GaugeReadResult.failure(
            FailureReason.NO_NEEDLE,
            dial_quality=0.75,
            centre=(100.0, 101.0),
            radius=80.0,
        )

        self.assertFalse(result.valid)
        self.assertIsNone(result.value_bar)
        self.assertEqual(
            result.to_dict()["failure_reason"],
            "NO_NEEDLE",
        )

    def test_quality_outside_unit_interval_is_rejected(self):
        # Reject a dial quality above 1.0 even when constructing a failure result.
        with self.assertRaises(ValueError):
            GaugeReadResult.failure(
                FailureReason.NO_DIAL,
                dial_quality=1.01,
            )

    def test_default_calibration_matches_assumed_scale(self):
        # The assumed 0-10 bar scale sweeps clockwise through 270 degrees.
        calibration = GaugeCalibration()

        self.assertEqual(calibration.minimum_bar, 0.0)
        self.assertEqual(calibration.maximum_bar, 10.0)
        self.assertEqual(calibration.zero_angle_degrees, 225.0)
        self.assertEqual(calibration.full_scale_angle_degrees, 135.0)
        self.assertEqual(calibration.clockwise_sweep_degrees, 270.0)

    def test_reader_config_exposes_blur_variance_threshold(self):
        config = ReaderConfig()

        self.assertEqual(
            getattr(config, "minimum_blur_variance", None),
            20.0,
        )


if __name__ == "__main__":
    unittest.main()
