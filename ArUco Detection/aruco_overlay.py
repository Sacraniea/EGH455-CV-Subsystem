"""Draw marker results on a copy of the camera image."""

import math

import cv2
import numpy as np

from aruco_types import ArUcoReadResult


def format_marker_label(result: ArUcoReadResult) -> str:
    # Label the marker with its rejection first, otherwise its available pose status.
    if result.failure_reason is not None:
        return f"ID {result.marker_id} | pose withheld: {result.failure_reason.value}"

    if result.valid:
        return f"ID {result.marker_id} | pose accepted"

    return f"ID {result.marker_id} | pose unavailable"


def marker_annotation_lines(result: ArUcoReadResult) -> list[str]:
    # Build display lines for the marker status and any usable pose measurements.
    lines = [format_marker_label(result)]

    if result.valid:
        x, y, z = result.pose.translation_m
        # Display X/Y/Z and the distance from the camera to the marker centre, in metres.
        lines += [
            f"X={x:+.3f} Y={y:+.3f} Z={z:.3f} m",
            f"range={math.hypot(x, y, z):.3f} m | RMS={result.pose.reprojection_error_px:.2f} px",
        ]

    elif (
        result.reprojection_error_px is not None
        and math.isfinite(result.reprojection_error_px)
    ):
        # If no pose is accepted, add any finite pixel error to the status label.
        lines.append(f"RMS={result.reprojection_error_px:.2f} px")

    return lines


def draw_aruco_overlay(
    image: np.ndarray,
    results: tuple[ArUcoReadResult, ...],
) -> np.ndarray:
    # Draw every marker and label on a new image, leaving the source pixels unchanged.
    overlay = image.copy()
    height, width = image.shape[:2]

    for result in results:
        # Round the detected corners to integer pixels for OpenCV's drawing functions.
        corners = (
            np.rint(np.asarray(result.corners_px))
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
        # Outline accepted markers in green and rejected markers in orange.
        colour = (0, 180, 0) if result.valid else (0, 165, 255)
        cv2.polylines(
            overlay,
            [corners],
            isClosed=True,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        lines = marker_annotation_lines(result)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55

        # Shrink long labels enough to fit within the image width.
        longest = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines)
        scale *= min(1.0, max(1, width - 12) / max(1, longest))
        sizes = [cv2.getTextSize(line, font, scale, 1) for line in lines]
        line_height = max(size[0][1] + size[1] for size in sizes) + 6
        block_height = line_height * len(lines)
        text_width = max(size[0][0] for size in sizes)

        left = int(np.min(corners[:, 0, 0]))
        top = int(np.min(corners[:, 0, 1]))
        x = max(3, min(left, width - text_width - 4))

        # Position the label above the marker, then limit its vertical offset to fit.
        first_y = top - 10 - line_height * (len(lines) - 1)
        first_y = max(
            line_height,
            min(first_y, height - block_height + line_height - 4),
        )

        for index, line in enumerate(lines):
            origin = (x, first_y + index * line_height)

            # Draw each line in black first, then white, to give the text an outline.
            for text_colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(
                    overlay,
                    line,
                    origin,
                    font,
                    scale,
                    text_colour,
                    thickness,
                    cv2.LINE_AA,
                )

    return overlay
