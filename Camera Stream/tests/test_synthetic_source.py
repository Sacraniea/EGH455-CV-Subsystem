import unittest
from unittest.mock import patch
import numpy as np
from synthetic_source import SyntheticFrameSource
from stream_types import get_stream_profile
from stream_aruco import ArUcoProcessor
from aruco_types import ArUcoConfig


class SyntheticSourceTests(unittest.TestCase):
    def test_generated_frames_carry_increasing_identity_and_real_detectable_marker(
        self,
    ):
        # Generate two frames and check that their images and identities advance.
        source = SyntheticFrameSource(get_stream_profile("720p30"))

        try:
            first = source.read()
            second = source.read()
            self.assertEqual(second.sequence_number, first.sequence_number + 1)
            self.assertGreater(second.device_timestamp_s, first.device_timestamp_s)
            self.assertFalse(np.shares_memory(first.image, second.image))
            self.assertEqual(source.device_details.mxid, "SYNTHETIC")
            # Run the ArUco detector and check that it returns valid marker 7 data.
            processor = ArUcoProcessor(
                calibration=source.calibration,
                config=ArUcoConfig(0.2, "DICT_5X5_100", frozenset({7})),
                max_reprojection_error_px=1.0,
            )
            result = processor.process(first).data["results"][0]
            self.assertEqual(result["marker_id"], 7)
            self.assertTrue(result["valid"])

        finally:
            source.close()

        # Check that closing the generator prevents another frame from being read.
        with self.assertRaises(RuntimeError):
            source.read()
