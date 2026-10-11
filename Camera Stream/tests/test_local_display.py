"""Tests for the OpenCV local-display boundary."""

import unittest

import numpy as np

from local_display import OpenCVDisplay
from stream_types import FramePacket


class RecordingCv2:
    """Small OpenCV substitute that records externally visible operations."""

    def __init__(self, keys):
        # Return the prepared key presses and record images and closed windows.
        self._keys = iter(keys)
        self.shown_frames = []
        self.destroyed_windows = []

    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4
    visible = 1

    def namedWindow(self, name, flags):
        self.window_created = name

    def resizeWindow(self, name, width, height):
        self.window_size = (width, height)

    def getWindowProperty(self, name, prop):
        return self.visible

    def imshow(self, window_name, image):
        """Record the window name and image passed to OpenCV."""

        self.shown_frames.append((window_name, image.copy()))

    def waitKey(self, delay_ms):
        """Return the next prepared keyboard value."""

        self.last_wait_delay_ms = delay_ms
        return next(self._keys)

    def destroyWindow(self, window_name):
        """Record which named window was released."""

        self.destroyed_windows.append(window_name)


class OpenCVDisplayTests(unittest.TestCase):
    """Define the behaviour of the local-screen output adapter."""

    def test_q_key_stops_stream_and_close_destroys_window(self):
        """Frames must display until q requests a controlled shutdown."""

        cv2_boundary = RecordingCv2(
            [
                -1,  # No key was pressed for the first frame.
                ord("q"),  # The operator stops after the second frame.
            ]
        )
        display = OpenCVDisplay(
            window_name="OAK-D Lite Preview",
            cv2_module=cv2_boundary,
        )

        image = np.full(
            (4, 6, 3),
            fill_value=80,
            dtype=np.uint8,
        )
        packet = FramePacket(
            image=image,
            sequence_number=1,
            device_timestamp_s=0.980,
            host_received_timestamp_s=1.000,
        )

        # Show the same frame twice; only the second call receives the quit key.
        self.assertTrue(display.show(packet))
        self.assertFalse(display.show(packet))

        # Check the displayed pixels, window name and one-millisecond key wait.
        self.assertEqual(len(cv2_boundary.shown_frames), 2)
        self.assertEqual(
            cv2_boundary.shown_frames[0][0],
            "OAK-D Lite Preview",
        )
        np.testing.assert_array_equal(
            cv2_boundary.shown_frames[0][1],
            image,
        )
        self.assertEqual(cv2_boundary.last_wait_delay_ms, 1)

        # Close the display and check that it released the named window.
        display.close()

        self.assertEqual(
            cv2_boundary.destroyed_windows,
            ["OAK-D Lite Preview"],
        )


if __name__ == "__main__":
    unittest.main()
