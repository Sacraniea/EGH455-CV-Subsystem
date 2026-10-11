"""Explicit generated-marker source for learning transport without a camera."""

import math
import time
import cv2
import numpy as np
import stream_aruco  # Add the sibling ArUco folder before importing its types.
from aruco_types import CameraCalibration
from stream_types import DeviceDetails, FramePacket


class SyntheticFrameSource:
    device_details = DeviceDetails(
        "SYNTHETIC DEMO — no camera", "SYNTHETIC", "NOT_APPLICABLE"
    )

    def __init__(self, profile):
        # Prepare a moving marker image and a made-up calibration for this profile.
        self.profile = profile
        self.sequence = 0
        self.closed = False
        self._next = time.monotonic()
        self._side = min(profile.width, profile.height) // 3
        # Generate marker 7, then convert its grey pixels to a three-channel image.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self._marker = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, 7, self._side),
            cv2.COLOR_GRAY2BGR,
        )
        # Set the focal length to the image width and the centre to the image centre.
        # Set distortion to zero for this generated image's pose calculations.
        f = float(profile.width)
        self.calibration = CameraCalibration(
            camera_matrix=(
                (f, 0.0, profile.width / 2),
                (0.0, f, profile.height / 2),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients=(0.0,) * 5,
            image_size_px=(profile.width, profile.height),
            source="synthetic-demo: invented intrinsics, not hardware calibration",
        )

    def read(self):
        # Generate the next labelled marker frame with its number and timing.
        if self.closed:
            raise RuntimeError("Synthetic source is closed")

        # Wait for this frame's turn, then schedule the next one at the chosen FPS.
        time.sleep(max(0.0, self._next - time.monotonic()))
        timestamp = time.monotonic()
        self._next = timestamp + 1.0 / self.profile.fps
        self.sequence += 1
        # Fill a white image and move the marker left or right as the frame advances.
        image = np.full((self.profile.height, self.profile.width, 3), 255, np.uint8)
        x = (self.profile.width - self._side) // 2 + int(
            math.sin(self.sequence / 45.0) * self.profile.width * 0.1
        )
        y = (self.profile.height - self._side) // 2
        image[y : y + self._side, x : x + self._side] = self._marker
        # Label the generated image and display its current frame number.
        cv2.putText(
            image,
            "SYNTHETIC DEMO - no camera",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 180),
            2,
        )
        cv2.putText(
            image,
            f"Frame {self.sequence}",
            (20, self.profile.height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 50, 50),
            2,
        )
        # Attach the times just before and after generating this image.
        return FramePacket(image, self.sequence, timestamp, time.monotonic())

    def close(self):
        self.closed = True
