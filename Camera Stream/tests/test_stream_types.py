"""Tests for shared camera-stream data contracts."""

import unittest

from stream_types import get_stream_profile


class StreamProfileTests(unittest.TestCase):
    """Define the supported and reproducible camera configurations."""

    def test_required_profiles_expose_expected_capture_settings(self):
        """The approved profile names must resolve to exact configurations."""

        expected_profiles = {
            "720p30": {
                "width": 1280,
                "height": 720,
                "fps": 30.0,
            },
            "1080p30": {
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
            },
        }

        # Look up each supported name and compare its dimensions and frame rate.
        for profile_name, expected in expected_profiles.items():
            with self.subTest(profile=profile_name):
                profile = get_stream_profile(profile_name)

                self.assertEqual(profile.name, profile_name)
                self.assertEqual(profile.width, expected["width"])
                self.assertEqual(profile.height, expected["height"])
                self.assertEqual(profile.fps, expected["fps"])

    def test_unknown_profile_is_rejected(self):
        """An unrecognised name must not silently select another profile."""

        with self.assertRaisesRegex(
            ValueError,
            "unsupported stream profile",
        ):
            get_stream_profile("4k120")


if __name__ == "__main__":
    unittest.main()
