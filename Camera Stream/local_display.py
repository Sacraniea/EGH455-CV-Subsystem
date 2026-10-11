"""OpenCV preview sink; no processing and no source-image modifications."""

import cv2
from stream_types import FramePacket


class OpenCVDisplay:
    def __init__(self, *, window_name, cv2_module=cv2):
        # Keep the window name and drawing library until the first frame arrives.
        self._window_name = window_name
        self._cv2 = cv2 if cv2_module is None else cv2_module
        self._opened = False

    def show(self, packet: FramePacket) -> bool:
        # Show this frame and report whether the caller should keep streaming.
        if not self._opened:
            # Create the preview window when showing the first image.
            self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
            self._opened = True
            # Set the preview window to 960 by 540 without resizing the image array.
            self._cv2.resizeWindow(self._window_name, 960, 540)

        self._cv2.imshow(self._window_name, packet.image)
        # Refresh the window, read a key and check whether the window is still open.
        key = self._cv2.waitKey(1) & 0xFF
        visible = self._cv2.getWindowProperty(
            self._window_name, self._cv2.WND_PROP_VISIBLE
        )
        # Stop the stream for q, Escape or a closed window.
        return key not in (ord("q"), 27) and visible >= 1

    def close(self):
        # Destroy the preview window if this display opened one.
        if self._opened:
            self._opened = False
            self._cv2.destroyWindow(self._window_name)
