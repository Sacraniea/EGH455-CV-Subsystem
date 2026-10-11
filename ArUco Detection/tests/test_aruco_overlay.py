import unittest

import numpy as np

from aruco_overlay import draw_aruco_overlay, format_marker_label
from aruco_types import ArUcoFailureReason, ArUcoReadResult


class ArUcoOverlayTests(unittest.TestCase):
    def test_draws_rejected_marker_without_changing_source(self):
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        original = image.copy()

        result = ArUcoReadResult(
            marker_id=7,
            corners_px=(
                (50.0, 50.0),
                (249.0, 50.0),
                (249.0, 249.0),
                (50.0, 249.0),
            ),
            pose=None,
            failure_reason=ArUcoFailureReason.MISSING_CALIBRATION,
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
        )

        self.assertEqual(
            format_marker_label(result),
            "ID 7 | pose withheld: MISSING_CALIBRATION",
        )

        overlay = draw_aruco_overlay(image, (result,))

        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(overlay, image))
        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(overlay.dtype, image.dtype)

        # Compare the top marker edge with the source to confirm that an outline was drawn.
        self.assertTrue(np.any(overlay[50, 80:220] != original[50, 80:220]))

        # Check for changed pixels above the marker where its label should have been drawn.
        self.assertTrue(np.any(overlay[10:45, :] != original[10:45, :]))


if __name__ == "__main__":
    unittest.main()
