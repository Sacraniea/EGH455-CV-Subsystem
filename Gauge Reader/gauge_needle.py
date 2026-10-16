"""Locate the needle pivot and follow the pointer outward through dark pixels."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

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
    """Keep the identity and position of a possible needle shape."""

    label: int
    distance_from_centre: float
    area: int


def _select_component(
    mask: np.ndarray,
    centre: np.ndarray,
    side: float,
) -> Tuple[Optional[_NeedleComponent], np.ndarray, int]:
    """Choose a substantial dark shape that reaches near the crop centre."""

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

        # A scale marking may touch the hand, so judge its direction later.
        candidates.append(
            _NeedleComponent(
                label=label,
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
    """Fill small holes in the hand, then measure distance from its edges."""

    component = (labels == label).astype(np.uint8) * 255

    # Fill only this component's enclosed holes, such as a bright brass hub.
    padded = np.pad(component, 1, constant_values=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    count, hole_labels, bounds, _ = cv2.connectedComponentsWithStats(holes)
    filled = component.copy()

    # Fill a small reflection inside the hub, but never fill the whole dial
    # when its printed scale has become connected to the needle.
    for hole in range(1, count):
        if bounds[hole, cv2.CC_STAT_AREA] <= 0.01 * component.size:
            filled[hole_labels == hole] = 255

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


def _radial_reach(
    mask: np.ndarray,
    pivot: np.ndarray,
    hub_radius: float,
) -> np.ndarray:
    """Measure the continuous dark reach at each clockwise, whole-degree angle."""

    # Start just outside the round hub so its width cannot favour a direction.
    angles = np.deg2rad(np.arange(360, dtype=np.float32))[:, None, None]
    radii = np.arange(
        np.ceil(hub_radius * 1.2),
        min(mask.shape) * 0.45,
        dtype=np.float32,
    )[None, :, None]
    offsets = np.array([-1, 0, 1], dtype=np.float32)[None, None, :]

    if radii.size < 3:
        return np.zeros(360)

    # Sample a three-pixel-wide strip along each ray to reduce pixel noise.
    xs = pivot[0] + np.sin(angles) * radii + np.cos(angles) * offsets
    ys = pivot[1] - np.cos(angles) * radii + np.sin(angles) * offsets
    samples = cv2.remap(
        mask,
        xs.reshape(360, -1).astype(np.float32),
        ys.reshape(360, -1).astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(360, -1, 3)
    dark = np.count_nonzero(samples, axis=2) >= 2

    # Allow a one- or two-pixel break. Three background pixels end the hand;
    # a separate printed tick further out must not extend the measured reach.
    gaps = (
        ~dark[:, :-2]
        & ~dark[:, 1:-1]
        & ~dark[:, 2:]
    )
    first_gap = np.where(
        gaps.any(axis=1),
        gaps.argmax(axis=1),
        dark.shape[1],
    )

    # Report distance from the pivot, or zero if no hand leaves the hub.
    return np.where(
        first_gap > 0,
        radii[0, 0, 0] + first_gap,
        0.0,
    )


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

    # Mark pixels darker than their nearby surroundings as possible needle pixels.
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=10,
    )

    metadata["threshhold_method"] = "adaptive_gaussian"

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
    )
    metadata["candidate_components"] = candidate_count

    if component is None:
        return rejected(FailureReason.NO_NEEDLE)

    thickness = _component_thickness(labels, component.label)
    pivot_array, maximum = _find_pivot(thickness, centre, side)
    metadata.update(
        component_area=component.area,
        max_central_half_thickness_px=maximum,
    )

    if pivot_array is None:
        return rejected(FailureReason.NO_PIVOT)

    pivot = (float(pivot_array[0]), float(pivot_array[1]))

    # Find which direction has the longest continuous hand leaving this pivot.
    hub_radius = float(thickness[int(pivot[1]), int(pivot[0])])

    # Reuse the filled hand so a reflection inside a wide pointer cannot break
    # the ray. Keep disconnected text and scale markings outside this mask.
    hand_mask = (thickness > 0).astype(np.uint8) * 255
    reaches = _radial_reach(hand_mask, pivot_array, hub_radius)
    angle = int(np.argmax(reaches))
    long_length = float(reaches[angle])

    # Nearby rays describe the same hand. Look around the opposite direction
    # for its counterweight, allowing a small offset from an exact straight line.
    angles = np.arange(360)
    separation = np.abs((angles - angle + 180) % 360 - 180)
    short_length = float(np.max(reaches[separation >= 165]))
    direction_ratio = long_length / max(short_length, 1.0)
    # Ignore rays within 20 degrees of the pointer and 15 of its opposite end.
    # Any long ray elsewhere is a competing direction, so the reading is uncertain.
    other_directions = (
        (separation > 20)
        & (separation < 165)
    )
    competing_length = float(np.max(reaches[other_directions]))
    quality = max(0.0, 1.0 - short_length / max(long_length, 1.0))
    metadata.update(
        direction_method="radial_search",
        long_side_px=long_length,
        short_side_px=short_length,
        direction_ratio=direction_ratio,
        competing_side_px=competing_length,
        pivot_half_thickness_px=hub_radius,
    )

    # A short protrusion from the hub is not enough evidence of a pointer.
    if (
        long_length < 0.13 * side
        or long_length < 3 * hub_radius
    ):
        return rejected(FailureReason.NO_NEEDLE)

    # Withhold pressure when the two ends or two separate directions compete.
    if (
        direction_ratio < config.minimum_needle_direction_ratio
        or competing_length * config.minimum_needle_direction_ratio > long_length
    ):
        return rejected(FailureReason.AMBIGUOUS_NEEDLE)

    radians = np.deg2rad(angle)
    tip = (
        float(pivot[0] + np.sin(radians) * long_length),
        float(pivot[1] - np.cos(radians) * long_length),
    )
    return NeedleDetection(
        valid=True,
        angle_degrees=float(angle),
        pivot=pivot,
        tip=tip,
        quality=quality,
        failure_reason=None,
        metadata=metadata,
    )
