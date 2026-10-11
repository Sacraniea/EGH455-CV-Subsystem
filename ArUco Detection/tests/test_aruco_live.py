import unittest

import cv2
import numpy as np

from stream_types import FramePacket
from stream_controller import run_stream
from aruco_live import ArUcoDisplay
from aruco_types import ArUcoConfig


class RecordingDisplay:
    """Capture displayed packets without opening a window."""

    def __init__(self):
        self.packets = []
        self.closed = False

    def show(self, packet):
        # Save the displayed packet for inspection and request an immediate stream stop.
        self.packets.append(packet)
        return False

    def close(self):
        self.closed = True


class OneFrameSource:
    """Supply one frame without opening a physical camera."""

    def __init__(self, packet):
        self._packets = iter((packet,))
        self.closed = False

    def read(self):
        return next(self._packets)

    def close(self):
        self.closed = True


class ArUcoDisplayTests(unittest.TestCase):

    def test_displays_overlay_preserving_source_and_frame_metadata(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)

        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        image[50:250, 50:250] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        original = image.copy()

        packet = FramePacket(
            image=image,
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
        )
        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )
        display = RecordingDisplay()
        adapter = ArUcoDisplay(
            display,
            calibration=None,
            config=config,
            max_reprojection_error_px=1.0,
        )

        should_continue = adapter.show(packet)

        # Check that the adapter returned the fake display's stop request.
        self.assertFalse(should_continue)
        self.assertEqual(len(display.packets), 1)
        shown = display.packets[0]

        self.assertIsNot(shown, packet)
        self.assertEqual(shown.sequence_number, 42)
        self.assertEqual(shown.device_timestamp_s, 12.5)
        self.assertEqual(shown.host_received_timestamp_s, 12.6)

        np.testing.assert_array_equal(packet.image, original)
        self.assertFalse(np.shares_memory(shown.image, packet.image))
        self.assertEqual(shown.image.shape, original.shape)

        # Compare pixels on the top marker edge to confirm that the outline was drawn.
        self.assertTrue(np.any(shown.image[50, 80:220] != original[50, 80:220]))

    def test_stream_stops_and_closes_source_and_display(self):
        packet = FramePacket(
            image=np.full((300, 400, 3), 255, dtype=np.uint8),
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
        )
        source = OneFrameSource(packet)
        display = RecordingDisplay()

        adapter = ArUcoDisplay(
            display,
            calibration=None,
            config=ArUcoConfig(
                marker_side_length_m=0.2,
                dictionary_name="DICT_5X5_100",
                permitted_ids=frozenset({7}),
            ),
            max_reprojection_error_px=1.0,
        )

        summary = run_stream(
            source,
            adapter,
            requested_fps=30.0,
        )

        self.assertEqual(summary.frames_received, 1)
        self.assertEqual(len(display.packets), 1)
        self.assertTrue(display.closed)
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
