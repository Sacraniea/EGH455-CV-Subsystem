"""Generate labelled marker frames for an explicit camera-free test."""

import math
import time

import cv2
import numpy as np

import detection_aruco  # Add the shared ArUco directory before importing its types.
from aruco_types import CameraCalibration
from stream_types import DeviceDetails, FramePacket


class SyntheticFrameSource:
    """Reuse the generated marker and invented intrinsics from the existing demo."""

    device_details = DeviceDetails(
        "SYNTHETIC DEMO — no camera", "SYNTHETIC", "NOT_APPLICABLE"
    )

    def __init__(self, profile):
        self.profile = profile
        self.sequence = 0
        self.closed = False
        self._next = time.monotonic()
        self._side = min(profile.width, profile.height) // 3

        # Generate marker 7 and convert its grey pixels to a BGR image.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self._marker = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, 7, self._side),
            cv2.COLOR_GRAY2BGR,
        )

        # These invented intrinsics belong only to the generated demonstration.
        focal_length = float(profile.width)
        self.calibration = CameraCalibration(
            camera_matrix = (
                (focal_length, 0.0, profile.width / 2),
                (0.0, focal_length, profile.height / 2),
                (0.0, 0.0, 1.0),
            ),
            distortion_coefficients = (0.0,) * 5,
            image_size_px = (profile.width, profile.height),
            source = "synthetic-demo: invented intrinsics, not hardware calibration",
        )

    def read(self):
        """Return the next independently generated image with its frame identity."""

        if self.closed:
            raise RuntimeError("Synthetic source is closed")

        # Schedule generated frames at the selected profile's requested frame rate.
        time.sleep(max(0.0, self._next - time.monotonic()))
        timestamp = time.monotonic()
        self._next = timestamp + 1.0 / self.profile.fps
        self.sequence += 1

        image = np.full((self.profile.height, self.profile.width, 3), 255, np.uint8)
        x = (self.profile.width - self._side) // 2 + int(
            math.sin(self.sequence / 45.0) * self.profile.width * 0.1
        )
        y = (self.profile.height - self._side) // 2
        image[y : y + self._side, x : x + self._side] = self._marker

        # Keep the generated source apparent in every transmitted image.
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
        return FramePacket(image, self.sequence, timestamp, time.monotonic())

    def close(self):
        self.closed = True
