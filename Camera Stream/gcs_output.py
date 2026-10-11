"""Build the log messages and photo records that the GCS expects."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

_POSE_REASONS = {
    "MISSING_CALIBRATION": "calibration missing",
    "CALIBRATION_SIZE_MISMATCH": "calibration does not match the frame size",
    "MARKER_ID_NOT_ALLOWED": "marker ID not permitted",
    "ID_NOT_PERMITTED": "marker ID not permitted",
    "DUPLICATE_MARKER_ID": "same marker ID appears more than once",
    "REPROJECTION_ERROR_TOO_HIGH": "corner fit error too high",
    "NONFINITE_POSE": "pose calculation returned invalid numbers",
    "NONPOSITIVE_DEPTH": "marker is not in front of the camera",
    "POSE_SOLVER_FAILED": "pose calculation failed",
}


def _finite_number(value):
    # Check that a position, pixel coordinate or timestamp is a finite number.
    # Return False for booleans, text, infinity and NaN before formatting them.
    return type(value) in (int, float) and isfinite(value)


def _camera_translation(result):
    """Return camera-relative position only when the pose can be used."""
    pose = result.get("pose")

    # Return no translation if the marker was rejected or has no pose record.
    if not result.get("valid") or not isinstance(pose, dict):
        return None

    translation = pose.get("translation_m")

    # Reject positions labelled with another coordinate system, incomplete or invalid
    # x/y/z values, and a forward distance of zero or less.
    if (
        pose.get("coordinate_frame") != "camera_optical"
        or not isinstance(translation, (list, tuple))
        or len(translation) != 3
        or not all(_finite_number(value) for value in translation)
        or translation[2] <= 0
    ):
        return None

    return translation


def _marker_summary(result, processor_failed):
    # Build one marker's log line, withholding its position if processing failed.
    translation = None if processor_failed else _camera_translation(result)

    if translation is not None:
        x, y, z = translation
        # The camera uses metres: X points right, Y down and Z forward.
        # Convert to centimetres here so the log is easier to read.
        pose_text = (
            f'{abs(x)*100:.1f} cm {"right" if x >= 0 else "left"}, '
            f'{abs(y)*100:.1f} cm {"down" if y >= 0 else "up"}, '
            f"{z*100:.1f} cm forward"
        )

    else:
        # Replace the unavailable position with the reason it was rejected.
        code = result.get("failure_reason")

        if processor_failed:
            # A branch error takes priority over any individual pose result.
            reason = "detection processing failed"

        elif code:
            # Turn the stored failure code into a readable explanation.
            reason = _POSE_REASONS.get(code, str(code).replace("_", " ").lower())

        else:
            reason = "pose did not pass validation"

        error = result.get("reprojection_error_px")

        if (
            not processor_failed
            and code == "REPROJECTION_ERROR_TOO_HIGH"
            and _finite_number(error)
        ):
            # Include the measured pixel error when it caused the pose rejection.
            reason += f": {error:.2f} px"

        pose_text = f"pose unavailable ({reason})"

    corners = result.get("corners_px")

    # Calculate image position only when all four corners have numeric x/y values.
    if (
        isinstance(corners, (list, tuple))
        and len(corners) == 4
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(_finite_number(value) for value in point)
            for point in corners
        )
    ):
        # Average the four corners to give a position in the original image.
        u, v = (sum(point[axis] for point in corners) / 4 for axis in (0, 1))
        image_text = f"image position ({u:.0f}, {v:.0f}) px"

    else:
        # Do not invent an image position when the corner coordinates are missing.
        image_text = "image position unavailable"

    return f'Marker {result["marker_id"]}: {pose_text}; {image_text}.'


def _find_processor(record, name):
    """Return this frame's result for the named processor, when it is present."""
    for processor in record.get("processors", []):
        if processor.get("processor") == name:
            return processor

    return None


def gcs_log_payload(record: dict) -> dict | None:
    """Turn one frame's ArUco results into a readable GCS log message."""

    if record.get("producer") != "camera_stream":
        raise ValueError("GCS ArUco logs require a native camera_stream record")

    # Find the ArUco entry in the frame's processor results; ignore other models.
    processor = _find_processor(record, "aruco")

    if processor is None:
        # There is no ArUco result to turn into a GCS log for this frame.
        return None

    failure = processor.get("failure_reason")

    # Skip frames with neither an ArUco detection nor an ArUco processing error.
    if not failure and not processor.get("data", {}).get("results"):
        return None

    # Work on a copy so building the log doesn't change our saved evidence.
    content = {
        key: deepcopy(value) for key, value in record.items() if key != "processors"
    }
    processor = deepcopy(processor)
    content["processors"] = [processor]
    results = processor.get("data", {}).get("results", [])

    for result in results:
        # Accept a pose only if the branch and marker succeeded and a pose exists.
        accepted = (
            not failure
            and result.get("valid") is True
            and result.get("failure_reason") is None
            and isinstance(result.get("pose"), dict)
        )

        if not accepted:
            # Mark this copied result invalid and remove its rejected pose.
            result["valid"] = False
            result["pose"] = None

    # Check the copied evidence can be encoded as JSON; raise on NaN or infinity.
    json.dumps(content, allow_nan=False)
    # Choose the action label from the branch status and usable marker positions.
    action = (
        "ArUco processing error"
        if failure
        else (
            "ArUco detected"
            if any(_camera_translation(result) is not None for result in results)
            else "ArUco detected - pose unavailable"
        )
    )
    lines = ["Synthetic demo."] if record.get("source_kind") == "synthetic" else []

    if failure:
        lines.append("Detection processing failed; no pose reported.")

    lines.extend(_marker_summary(result, bool(failure)) for result in results)
    # Add the frame number, elapsed device time and run ID to identify the reading.
    device_time = record.get("device_timestamp_s")
    time_text = f"{device_time:.3f} s" if _finite_number(device_time) else "unavailable"
    lines.append(
        f'Frame {record["sequence_number"]} | Device time {time_text} | Run {record["run_id"]}'
    )

    return {"action": action, "content": "\n".join(lines)}


def _target_image_position(box):
    """Describe the centre of a valid target box in original-frame pixels."""

    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not all(_finite_number(value) for value in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        # Keep the detection but withhold its position when the box is unusable.
        return "image position unavailable"

    # Average each pair of edges without summing large coordinates first.
    u = box[0] / 2 + box[2] / 2
    v = box[1] / 2 + box[3] / 2
    return f"image position ({u:g}, {v:g}) px"


def gcs_log_payloads(record: dict) -> list[dict]:
    """Keep the ArUco log and add one readable log for each gauge or valve."""

    # Reuse the existing ArUco formatter so photo captions and old logs stay the same.
    aruco = gcs_log_payload(record)
    payloads = [] if aruco is None else [aruco]
    processor = _find_processor(record, "targets")

    if processor is None:
        return payloads

    prefix = "Synthetic demo. " if record.get("source_kind") == "synthetic" else ""
    frame = f'Frame {record["sequence_number"]}'

    if processor.get("failure_reason") is not None:
        # Report the failed branch without publishing any candidate or stale readings.
        payloads.append(
            {
                "action": "Target processing error",
                "content": f"{prefix}Target processing failed; {frame}.",
            }
        )
        return payloads

    for result in processor.get("data", {}).get("results", []):
        class_name = result.get("class_name")

        if class_name == "gauge":
            action = "Gauge detected"
            reading = result.get("gauge_reading")
            summary = "Pressure unavailable"

            if result.get("reading_status") == "pending":
                # Keep a sudden candidate reading out of logs until later frames agree.
                summary = "Pressure unavailable (awaiting confirmation)"

            if (
                isinstance(reading, dict)
                and reading.get("valid") is True
                and reading.get("failure_reason") is None
                and _finite_number(reading.get("value_bar"))
                and result.get("reading_status") in (None, "confirmed")
            ):
                # Show only accepted pressure, rounded without unnecessary zeros.
                pressure = f'{reading["value_bar"]:.2f}'.rstrip("0").rstrip(".")
                summary = f"Pressure: {pressure} bar"

        elif class_name in ("openValve", "closedValve"):
            action = "Valve detected"
            summary = "Open valve" if class_name == "openValve" else "Closed valve"

        else:
            # Skip unrecognised classes rather than inventing a gauge or valve state.
            continue

        position = _target_image_position(result.get("bbox_xyxy_px"))
        # Keep remote text compact; the native record retains the full run identity.
        payloads.append(
            {
                "action": action,
                "content": f"{prefix}{summary}; {position}; {frame}.",
            }
        )

    return payloads


def gcs_detection_payload(
    path: str | Path,
    target_type: str,
    description: str,
    *,
    timestamp: str | None = None,
) -> dict:
    """Package an original photo for MQTT without changing its image bytes."""

    if (
        not isinstance(target_type, str)
        or not isinstance(description, str)
    ):
        raise ValueError("Detection type and description must be strings")

    if timestamp is not None:
        # Parse the supplied capture time and reject text without a timezone.
        if not isinstance(timestamp, str):
            raise ValueError(
                "Detection timestamp must be an ISO 8601 string with a timezone"
            )

        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        except ValueError as error:
            raise ValueError(
                "Detection timestamp must be an ISO 8601 string with a timezone"
            ) from error

        if parsed.utcoffset() is None:
            raise ValueError("Detection timestamp must include a timezone")

    path = Path(path)
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        path.suffix.lower()
    )

    if mime is None:
        raise ValueError("Detection images must be .jpg, .jpeg or .png files")

    original = path.read_bytes()
    # Check the file begins with the JPEG or PNG signature required by its suffix.
    signature = b"\xff\xd8\xff" if mime == "image/jpeg" else b"\x89PNG\r\n\x1a\n"

    if not original.startswith(signature):
        raise ValueError("Image bytes do not match the file extension")

    # Decode the bytes for validation without replacing the original image file.
    image = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if image is None or image.size == 0:
        raise ValueError("Detection image could not be decoded")

    # Return the five fields expected by the GCS detection-image receiver.
    return {
        "type": target_type,
        "description": description,
        "mime": mime,
        # Encode the original file bytes as Base64 text without changing the image.
        "image": base64.b64encode(original).decode("ascii"),
        # With no capture time supplied, this is when we prepared the message.
        "timestamp": (
            timestamp
            if timestamp is not None
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
    }
