"""Read pressure directly from the pivot and pointer in a detected gauge crop."""

from typing import Any, Dict, Optional

import numpy as np

from gauge_geometry import pressure_from_angle
from gauge_needle import detect_needle
from gauge_preprocess import preprocess_candidate
from gauge_types import (
    FailureReason,
    GaugeCalibration,
    GaugeReadResult,
    ReaderConfig,
)


def read_gauge(
    image: Optional[np.ndarray],
    calibration: GaugeCalibration = GaugeCalibration(),
    config: ReaderConfig = ReaderConfig(),
    metadata: Optional[Dict[str, Any]] = None,
) -> GaugeReadResult:
    """Accept one gauge crop and retain evidence even when pressure is unavailable."""

    # Keep the caller's frame or target details without changing its dictionary.
    evidence_metadata = dict(metadata or {})

    # Prepare Aidan's crop for the needle search without finding or cropping a face.
    prepared = preprocess_candidate(image, config)

    if not prepared.valid:
        return GaugeReadResult.failure(
            prepared.failure_reason or FailureReason.INVALID_INPUT,
            metadata=evidence_metadata,
        )

    evidence_metadata["blur_variance"] = prepared.blur_variance

    # Withhold pressure if the image does not have enough detail to read a pointer.
    if prepared.blur_variance < config.minimum_blur_variance:
        return GaugeReadResult.failure(
            FailureReason.DIAL_TOO_BLURRED,
            metadata=evidence_metadata,
        )

    assert prepared.working_gray is not None
    assert prepared.transform is not None

    needle = detect_needle(prepared.working_gray, config)
    evidence_metadata["needle_detection"] = dict(needle.metadata)

    # Undo the working-image resize and padding for both measured points.
    source_pivot = (
        prepared.transform.to_source_point(needle.pivot)
        if needle.pivot is not None
        else None
    )
    source_tip = (
        prepared.transform.to_source_point(needle.tip)
        if needle.tip is not None
        else None
    )

    # Accepted and rejected results retain the same available geometry and evidence.
    evidence = {
        "needle_quality": needle.quality,
        "angle_degrees": needle.angle_degrees,
        "centre": source_pivot,
        "tip": source_tip,
        "metadata": evidence_metadata,
    }

    if (
        not needle.valid
        or needle.angle_degrees is None
        or source_pivot is None
        or source_tip is None
    ):
        return GaugeReadResult.failure(
            needle.failure_reason or FailureReason.NO_NEEDLE,
            **evidence,
        )

    # Use the existing scale calibration to convert the pointer direction to bar.
    pressure_bar = pressure_from_angle(
        needle.angle_degrees,
        calibration,
    )

    if pressure_bar is None:
        return GaugeReadResult.failure(
            FailureReason.ANGLE_OUTSIDE_SCALE,
            **evidence,
        )

    return GaugeReadResult.success(
        value_bar=pressure_bar,
        **evidence,
    )
