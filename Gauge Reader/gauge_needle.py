"""Locate a central needle component, its pivot, and its longer pointer end."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from gauge_geometry import angle_from_points
from gauge_types import FailureReason, Point, ReaderConfig


@dataclass(frozen=True)
class NeedleDetection:
    """Return working-image geometry and the evidence behind a detection.

    Quality measures directional separation: one minus short/long end length.
    It is not a probability that the detected object is a gauge needle.
    """

    valid: bool
    angle_degrees: Optional[float]
    pivot: Optional[Point]
    tip: Optional[Point]
    quality: float
    failure_reason: Optional[FailureReason]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _NeedleComponent:
    """Keep the pixels and principal axis of an eligible dark component."""

    label: int
    points: np.ndarray
    axis: np.ndarray
    elongation: float
    distance_from_centre: float
    area: int


def _select_component(
    mask: np.ndarray,
    centre: np.ndarray,
    side: float,
    minimum_elongation: float,
) -> Tuple[Optional[_NeedleComponent], np.ndarray, int]:
    """Choose the closest sufficiently large, elongated central component."""
    count, labels, bounds, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    candidates = []

    for label in range(1, count):
        x, y, width, height, area = bounds[label]

        # Ignore small marks and objects clipped by the working-image border.
        if (
            area < 0.001 * side ** 2
            or max(width, height) < 0.20 * side
        ):
            continue

        if (
            min(x, y) <= 0
            or x + width >= side
            or y + height >= side
        ):
            continue

        ys, xs = np.nonzero(labels == label)
        points = np.column_stack([xs, ys]).astype(float)
        nearest = float(np.min(np.linalg.norm(points - centre, axis=1)))
        if nearest > 0.15 * side:
            continue

        # Covariance measures the pixels' spread in different directions; its
        # eigenvectors identify the axes, with the greatest spread along the hand.
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        elongation = float(eigenvalues[-1] / max(eigenvalues[0], 1.0))
        if elongation < minimum_elongation:
            continue

        candidates.append(
            _NeedleComponent(
                label=label,
                points=points,
                axis=eigenvectors[:, -1],
                elongation=elongation,
                distance_from_centre=nearest,
                area=int(area),
            )
        )

    if not candidates:
        return None, labels, 0

    # Larger area breaks an equal-distance tie without depending on label order.
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.distance_from_centre,
            -candidate.area,
        ),
    )
    return selected, labels, len(candidates)


def _component_thickness(labels: np.ndarray, label: int) -> np.ndarray:
    """Fill bright holes inside the selected hand before measuring thickness."""
    component = (labels == label).astype(np.uint8) * 255

    # Fill only this component's enclosed holes, such as a bright brass hub.
    padded = np.pad(component, 1, constant_values=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    filled = cv2.bitwise_or(padded, cv2.bitwise_not(flooded))[1:-1, 1:-1]
    return cv2.distanceTransform(filled, cv2.DIST_L2, 5)


def _find_pivot(
    thickness: np.ndarray,
    centre: np.ndarray,
    side: float,
) -> Tuple[Optional[np.ndarray], float]:
    """Choose a substantial local thickness peak near the crop centre."""
    ys, xs = np.indices(thickness.shape)
    central = np.hypot(xs - centre[0], ys - centre[1]) <= 0.20 * side
    maximum = float(np.max(thickness[central]))
    if maximum < 0.008 * side:
        return None, maximum

    # Retain similarly thick peaks, then prefer the one nearest the crop centre.
    # This helps distinguish a central hub from a thick, displaced counterweight.
    local_maximum = cv2.dilate(thickness, np.ones((3, 3), np.uint8))
    peaks = (
        (thickness == local_maximum)
        & central
        & (thickness >= 0.70 * maximum)
    )
    peak_ys, peak_xs = np.nonzero(peaks)
    if peak_xs.size == 0:
        # A thick feature may extend inside the search area while its peak is outside.
        return None, maximum

    points = np.column_stack([peak_xs, peak_ys]).astype(float)
    nearest = np.argmin(np.linalg.norm(points - centre, axis=1))
    return points[nearest], maximum


def detect_needle(gray: np.ndarray, config: ReaderConfig) -> NeedleDetection:
    """Find a pivot and directed needle using only the preprocessed crop.

    The reader validates and preprocesses the input before this stage. Distances
    here refer to that square working image; pressure calibration happens later.
    """
    metadata: Dict[str, Any] = {}
    pivot: Optional[Point] = None
    quality = 0.0

    def rejected(reason: FailureReason) -> NeedleDetection:
        # Preserve evidence already found, but never invent a directed endpoint.
        return NeedleDetection(
            valid=False,
            angle_degrees=None,
            pivot=pivot,
            tip=None,
            quality=quality,
            failure_reason=reason,
            metadata=metadata,
        )

    side = float(config.working_size)
    centre = np.array([gray.shape[1] / 2, gray.shape[0] / 2])

    # Otsu chooses an automatic brightness cutoff and marks darker pixels foreground.
    threshold, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    metadata["otsu_threshold"] = float(threshold)

    # Closing joins tiny gaps in the dark hand before connected-component detection.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    component, labels, candidate_count = _select_component(
        mask,
        centre,
        side,
        config.minimum_needle_elongation,
    )
    metadata["candidate_components"] = candidate_count
    if component is None:
        return rejected(FailureReason.NO_NEEDLE)

    thickness = _component_thickness(labels, component.label)
    pivot_array, maximum = _find_pivot(thickness, centre, side)
    metadata.update(
        component_area=int(len(component.points)),
        elongation=component.elongation,
        max_central_half_thickness_px=maximum,
    )
    if pivot_array is None:
        return rejected(FailureReason.NO_PIVOT)

    pivot = (float(pivot_array[0]), float(pivot_array[1]))

    # Compare supported reach on both sides of the pivot along the principal axis.
    projections = (component.points - pivot_array) @ component.axis

    # Quantiles ignore a few extreme pixels at either end of the component.
    low, high = np.quantile(projections, [0.01, 0.99])
    lengths = np.array([-low, high])
    long_index = int(np.argmax(lengths))
    long_length = float(lengths[long_index])
    short_length = float(lengths[1 - long_index])
    direction_ratio = long_length / max(short_length, 1.0)
    metadata.update(
        long_side_px=long_length,
        short_side_px=short_length,
        direction_ratio=direction_ratio,
        pivot_half_thickness_px=float(thickness[int(pivot[1]), int(pivot[0])]),
    )
    if (
        long_length > 0.0
        and short_length >= 0.0
    ):
        quality = max(0.0, min(1.0, 1.0 - short_length / long_length))

    if (
        long_length < 0.13 * side
        or short_length < 0.0
    ):
        return rejected(FailureReason.NO_NEEDLE)

    if direction_ratio < config.minimum_needle_direction_ratio:
        return rejected(FailureReason.AMBIGUOUS_NEEDLE)

    # The longer supported side is the reading pointer, not its counterweight.
    # This axis endpoint measures its angle; it may lie inside the physical tip.
    direction = component.axis if long_index == 1 else -component.axis
    tip_array = pivot_array + direction * long_length
    tip = (float(tip_array[0]), float(tip_array[1]))
    return NeedleDetection(
        valid=True,
        angle_degrees=float(angle_from_points(pivot, tip)),
        pivot=pivot,
        tip=tip,
        quality=quality,
        failure_reason=None,
        metadata=metadata,
    )
