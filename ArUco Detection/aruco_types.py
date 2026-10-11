from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Protocol

import numpy as np


class ArUcoFailureReason(str, Enum):
    REPROJECTION_ERROR_TOO_HIGH = "REPROJECTION_ERROR_TOO_HIGH"
    NONFINITE_POSE = "NONFINITE_POSE"
    NONPOSITIVE_DEPTH = "NONPOSITIVE_DEPTH"
    MISSING_CALIBRATION = "MISSING_CALIBRATION"
    MARKER_ID_NOT_ALLOWED = "MARKER_ID_NOT_ALLOWED"
    CALIBRATION_SIZE_MISMATCH = "CALIBRATION_SIZE_MISMATCH"
    DUPLICATE_MARKER_ID = "DUPLICATE_MARKER_ID"
    POSE_SOLVER_FAILED = "POSE_SOLVER_FAILED"


@dataclass(frozen=True)
class ArUcoConfig:
    marker_side_length_m: float  # Outer black square, excluding the white border.
    dictionary_name: str
    permitted_ids: frozenset[int]  # Other IDs are still detected, but get no pose.

    def __post_init__(self):
        # Reject settings that cannot describe a positive-size DICT_5X5_100 marker.
        if (
            not isfinite(self.marker_side_length_m)
            or self.marker_side_length_m <= 0
        ):
            raise ValueError(
                f"Marker side length must be a finite positive "
                f"number, got {self.marker_side_length_m}"
            )

        if self.dictionary_name not in {"DICT_5X5_100"}:
            raise ValueError(f"Unsupported ArUco dictionary: {self.dictionary_name}")

        for marker_id in self.permitted_ids:
            # Raise TypeError for a marker ID supplied as a boolean or non-integer.
            if (
                isinstance(marker_id, bool)
                or not isinstance(marker_id, int)
            ):
                raise TypeError(f"Marker ID must be an integer, got {marker_id}")

            # Reject integers outside this dictionary's IDs, which run from 0 to 99.
            if not (0 <= marker_id < 100):
                raise ValueError(
                    f"Permitted marker ID must be between 0 and 99, got {marker_id}"
                )


@dataclass(frozen=True)
class MarkerObservation:
    marker_id: int
    corners_px: tuple[tuple[float, float], ...]  # Keep the detector's corner order.

    def __post_init__(self):
        # Require exactly four image corners, each with finite x and y coordinates.
        if len(self.corners_px) != 4:
            raise ValueError(
                f"Marker observation must have exactly 4 corners, got {len(self.corners_px)}"
            )

        for x, y in self.corners_px:
            if (
                not isfinite(x)
                or not isfinite(y)
            ):
                raise ValueError(
                    f"Marker corner coordinates must be finite, got ({x}, {y})"
                )


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: tuple[tuple[float, ...], ...]
    distortion_coefficients: tuple[float, ...]
    image_size_px: tuple[int, int]  # width, height
    source: str

    def __post_init__(self):
        # Validate the matrix, image dimensions and lens coefficients before use.
        if len(self.camera_matrix) != 3:
            raise ValueError("Camera matrix must have exactly 3 rows")

        # Each of the three matrix rows must contain three finite numbers.
        for row in self.camera_matrix:
            if len(row) != 3:
                raise ValueError("Each camera matrix row must have exactly 3 values")

            for value in row:
                if not isfinite(value):
                    raise ValueError("Camera matrix values must be finite")

        # Read both focal lengths from the diagonal and reject zero or negative values.
        fx = self.camera_matrix[0][0]
        fy = self.camera_matrix[1][1]

        if (
            fx <= 0
            or fy <= 0
        ):
            raise ValueError("Camera focal lengths must be positive")

        width, height = self.image_size_px

        if (
            width <= 0
            or height <= 0
        ):
            raise ValueError("Calibration image width and height must be positive")

        # Reject coefficient lists whose length OpenCV's perspective model cannot use.
        if len(self.distortion_coefficients) not in {4, 5, 8, 12, 14}:
            raise ValueError(
                "Distortion coefficients must contain 4, 5, 8, 12 or 14 values"
            )

        for coefficient in self.distortion_coefficients:
            if not isfinite(coefficient):
                raise ValueError("Distortion coefficients must be finite")


@dataclass(frozen=True)
class MarkerPose:
    # Camera axes: X right, Y down and Z forward.
    translation_m: tuple[float, float, float]
    # Direction gives the rotation axis; length gives its angle in radians.
    rotation_vector_rad: tuple[float, float, float]
    reprojection_error_px: float
    coordinate_frame: str = field(
        default="camera_optical",
        init=False,
    )


@dataclass(frozen=True)
class ArUcoReadResult:
    marker_id: int
    corners_px: tuple[tuple[float, float], ...]
    pose: MarkerPose | None
    failure_reason: ArUcoFailureReason | None
    sequence_number: int | None
    device_timestamp_s: float | None
    host_received_timestamp_s: float | None
    reprojection_error_px: float | None = None

    @property
    def valid(self) -> bool:
        # Accept the result only when a pose is present and no rejection was recorded.
        return self.pose is not None and self.failure_reason is None


class FramePacketLike(Protocol):
    """Fields the reader needs; saved images may have no capture timestamps."""

    image: np.ndarray  # uint8 BGR pixels, shaped (height, width, 3).
    sequence_number: int | None
    device_timestamp_s: float | None
    host_received_timestamp_s: float | None
