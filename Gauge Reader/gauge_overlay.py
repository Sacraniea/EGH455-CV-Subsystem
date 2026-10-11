"""Render an explainable visual overlay for a gauge-reading result."""

from __future__ import annotations

import cv2
import numpy as np

from gauge_types import GaugeReadResult


def render_overlay(
    image: np.ndarray,
    result: GaugeReadResult,
    *,
    show_label: bool = True,
) -> np.ndarray:
    """Return an annotated copy, optionally without the reading or rejection label."""

    # The measured pivot and needle tip use pixels within this crop.
    # Make a colour copy before drawing so the supplied image stays unchanged.
    if image.ndim == 2:
        overlay = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    else:
        overlay = image.copy()

    # Choose green for an accepted reading and orange for a rejected reading.
    colour = (0, 180, 0) if result.valid else (0, 140, 255)

    # Keep the measured pivot visible even when the reading was rejected.
    if result.centre is not None:
        # The public centre field stores the needle pivot, not a fitted dial centre.
        centre = (
            int(round(result.centre[0])),
            int(round(result.centre[1])),
        )

        # Scale the marker and line with the crop; no dial circle is needed.
        crop_size = min(overlay.shape[:2])
        centre_marker_radius = max(
            5,
            int(round(0.01 * crop_size)),
        )
        geometry_stroke_width = max(
            3,
            int(round(0.005 * crop_size)),
        )

        # Draw a filled pivot marker whether or not a needle tip was detected.
        cv2.circle(
            overlay,
            centre,
            radius=centre_marker_radius,
            color=colour,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if result.tip is not None:
            tip = (
                int(round(result.tip[0])),
                int(round(result.tip[1])),
            )

            cv2.line(
                overlay,
                centre,
                tip,
                colour,
                thickness=(geometry_stroke_width * 2),
                lineType=cv2.LINE_AA,
            )

    # Leave the crop's text area alone when the caller supplies a full-frame label.
    if show_label:
        if result.valid:
            # Keep the visible label short; the JSON retains the detection evidence.
            label = f"{result.value_bar:.1f} bar"

        else:
            # Label a rejected reading with its failure reason instead of a pressure value.
            reason = (
                result.failure_reason.value
                if result.failure_reason is not None
                else "UNKNOWN"
            )
            label = f"Rejected: {reason}"

        # Draw the reading or rejection label at the top left of the overlay.
        cv2.putText(
            overlay,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return overlay
