"""Validate and normalise candidate crops for later detection stages."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gauge_types import FailureReason, Point, ReaderConfig


@dataclass(frozen=True)
class LetterboxTransform:
    """Map geometry between the source crop and square working image.
    Attributes:
        scale: Enlargement or reduction applied to source coordinates.
        pad_x: Horizontal padding added after scaling.
        pad_y: Vertical padding added after scaling.
    """

    scale: float
    pad_x: float
    pad_y: float

    def to_working_point(self, point: Point) -> Point:
        """Transform a source-image point into working-image coordinates."""
        return (
            point[0] * self.scale + self.pad_x,
            point[1] * self.scale + self.pad_y,
        )

    def to_source_point(self, point: Point) -> Point:
        """Transform a working-image point back into source coordinates."""
        return (
            (point[0] - self.pad_x) / self.scale,
            (point[1] - self.pad_y) / self.scale,
        )


@dataclass(frozen=True)
class PreprocessOutcome:
    """Structured result produced by candidate-crop preprocessing.
    Rejected outcomes contain a failure reason and no working images.
    Accepted outcomes contain the normalised images, coordinate transform,
    and measured blur evidence.
    """

    valid: bool
    failure_reason: Optional[FailureReason]
    working_colour: Optional[np.ndarray]
    working_gray: Optional[np.ndarray]
    transform: Optional[LetterboxTransform]
    blur_variance: float


def _rejected(reason: FailureReason) -> PreprocessOutcome:
    """Construct one consistent failed preprocessing result."""
    return PreprocessOutcome(
        valid=False,
        failure_reason=reason,
        working_colour=None,
        working_gray=None,
        transform=None,
        blur_variance=0.0,
    )


def preprocess_candidate(
    image: Optional[np.ndarray],
    config: ReaderConfig,
) -> PreprocessOutcome:
    """Validate and normalise one candidate gauge crop.
    The crop is converted to three-channel BGR, resized without changing
    its aspect ratio, centred on a square white canvas, and converted into
    an enhanced grayscale working image.
    Args:
        image: Candidate crop supplied as a NumPy image array.
        config: Reader dimensions and acceptance thresholds.
    Returns:
        A valid preprocessing outcome containing working images and their
        coordinate transform, or a rejected outcome with a stable reason.
    """

    # Reject missing, non-array or empty input before reading its dimensions.
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        return _rejected(FailureReason.INVALID_INPUT)

    # Convert supported greyscale, BGR and BGRA inputs to three-channel BGR.
    if image.ndim == 2:

        colour = cv2.cvtColor(
            image.astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):

        # Convert BGR values to 8-bit pixels without copying an already-matching array.
        colour = image.astype(np.uint8, copy=False)

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        # Convert four-channel BGRA input to BGR, discarding the alpha channel.
        colour = cv2.cvtColor(
            image.astype(np.uint8),
            cv2.COLOR_BGRA2BGR,
        )

    else:
        return _rejected(FailureReason.INVALID_INPUT)

    source_height, source_width = colour.shape[:2]

    # Reject the crop as DIAL_TOO_SMALL if either source dimension is below the limit.
    if min(source_height, source_width) < config.minimum_source_side_px:
        return _rejected(FailureReason.DIAL_TOO_SMALL)

    # Fit the longest source side to working_size and apply that scale to both axes.
    scale = config.working_size / float(
        max(source_height, source_width)
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        colour,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    # Calculate the padding and place the resized crop in the centre of a white square.
    pad_x = (config.working_size - resized_width) // 2
    pad_y = (config.working_size - resized_height) // 2

    canvas = np.full(
        (config.working_size, config.working_size, 3),
        255,
        dtype=np.uint8,
    )
    canvas[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized

    # Measure greyscale edge variation before smoothing; the reader uses it for blur
    # rejection.
    raw_gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY,
    )
    blur_variance = float(cv2.Laplacian(raw_gray, cv2.CV_64F).var())

    # Apply a 3x3 Gaussian blur to reduce isolated pixel noise before detection.
    denoised = cv2.GaussianBlur(
        raw_gray,
        (3, 3),
        0,
    )

    # Enhance contrast within local image tiles before searching for the pointer.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    # Keep smoothed greyscale image for the local bightness comparison.
    working_gray = denoised

    return PreprocessOutcome(
        valid=True,
        failure_reason=None,
        working_colour=canvas,
        working_gray=working_gray,
        transform=LetterboxTransform(
            scale=scale,
            pad_x=float(pad_x),
            pad_y=float(pad_y),
        ),
        blur_variance=blur_variance,
    )
