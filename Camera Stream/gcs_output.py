"""Prepare readable GCS logs and photo messages from completed detections.

These functions build the message contents. MQTT publishing happens separately.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path

import cv2
import numpy as np


# Replace the stored ArUco rejection codes with explanations for the GCS log.
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
    """Check that a measurement can be included as a number in the GCS text."""

    # Require an ordinary number with a finite value before formatting a measurement.
    return (
        type(value) in (int, float)
        and isfinite(value)
    )


def _camera_translation(result):
    """Return the marker's camera-relative x, y and z when its pose is usable."""

    pose = result.get("pose")

    # Withhold position when the marker was rejected or no pose was supplied.
    if (
        not result.get("valid")
        or not isinstance(pose, dict)
    ):
        return None

    translation = pose.get("translation_m")

    # Require three finite camera coordinates and a marker in front of the camera.
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
    """Describe one marker's position, or explain why its pose is unavailable."""

    # A failed detector cannot supply a position, even if a candidate pose remains.
    if processor_failed:
        translation = None

    else:
        translation = _camera_translation(result)

    if translation is not None:
        x, y, z = translation

        # Convert the signs of the camera coordinates into readable directions.
        if x >= 0:
            horizontal_direction = "right"

        else:
            horizontal_direction = "left"

        if y >= 0:
            vertical_direction = "down"

        else:
            vertical_direction = "up"

        # Report centimetres; z is forward depth along the camera's viewing axis.
        pose_text = (
            f"{abs(x) * 100:.1f} cm {horizontal_direction}, "
            f"{abs(y) * 100:.1f} cm {vertical_direction}, "
            f"{z * 100:.1f} cm forward"
        )

    else:
        failure_code = result.get("failure_reason")

        # Explain a detector failure before considering individual pose rejections.
        if processor_failed:
            reason = "detection processing failed"

        elif failure_code:
            # Use the known explanation, or make an unfamiliar code readable.
            fallback_reason = str(failure_code).replace("_", " ").lower()
            reason = _POSE_REASONS.get(failure_code, fallback_reason)

        else:
            reason = "pose did not pass validation"

        error_px = result.get("reprojection_error_px")

        # Include the corner-fit error when that measurement caused the rejection.
        if (
            not processor_failed
            and failure_code == "REPROJECTION_ERROR_TOO_HIGH"
            and _finite_number(error_px)
        ):
            reason += f": {error_px:.2f} px"

        pose_text = f"pose unavailable ({reason})"

    corners = result.get("corners_px")

    # Require all four marker corners before calculating its image position.
    corners_valid = (
        isinstance(corners, (list, tuple))
        and len(corners) == 4
    )

    if corners_valid:
        for point in corners:
            # Each corner must contain a usable horizontal and vertical coordinate.
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or not all(_finite_number(value) for value in point)
            ):
                corners_valid = False
                break

    if corners_valid:
        # Average the four corners to locate the marker in the original image.
        centre_x = sum(point[0] for point in corners) / 4
        centre_y = sum(point[1] for point in corners) / 4
        image_text = f"image position ({centre_x:.0f}, {centre_y:.0f}) px"

    else:
        image_text = "image position unavailable"

    return f'Marker {result["marker_id"]}: {pose_text}; {image_text}.'


def _find_processor(record, name):
    """Find the saved result for a named detector in this frame."""

    # Each frame can contain several detectors, such as targets and ArUco.
    for processor in record.get("processors", []):
        if processor.get("processor") == name:
            return processor

    return None


def gcs_log_payload(record: dict) -> dict | None:
    """Combine this frame's ArUco observations into one GCS log message."""

    # Require our frame-record format before reading its detector results.
    if record.get("producer") != "camera_stream":
        raise ValueError("GCS ArUco logs require a native camera_stream record")

    processor = _find_processor(record, "aruco")

    # A frame without ArUco processing has no ArUco message to prepare.
    if processor is None:
        return None

    failure = processor.get("failure_reason")

    # An empty successful detection produces no log; a processing failure still does.
    if (
        not failure
        and not processor.get("data", {}).get("results")
    ):
        return None

    # Copy the frame details and just the ArUco result for validation.
    # Removing a rejected pose here must not change the original frame evidence.
    aruco_record = {}

    for key, value in record.items():
        if key != "processors":
            aruco_record[key] = deepcopy(value)

    processor = deepcopy(processor)
    aruco_record["processors"] = [processor]
    marker_results = processor.get("data", {}).get("results", [])

    for result in marker_results:
        # Keep a pose only when both processing and this marker's checks succeeded.
        accepted = (
            not failure
            and result.get("valid") is True
            and result.get("failure_reason") is None
            and isinstance(result.get("pose"), dict)
        )

        if not accepted:
            result["valid"] = False
            result["pose"] = None

    # Reject remaining evidence that cannot be represented as standard JSON.
    json.dumps(aruco_record, allow_nan = False)

    # Distinguish a detector failure from detected markers with unavailable poses.
    if failure:
        action = "ArUco processing error"

    else:
        action = "ArUco detected - pose unavailable"

        # One accepted camera position is enough to use the normal detection label.
        for result in marker_results:
            if _camera_translation(result) is not None:
                action = "ArUco detected"
                break

    lines = []

    # Mark generated test frames so their messages cannot be mistaken for camera data.
    if record.get("source_kind") == "synthetic":
        lines.append("Synthetic demo.")

    if failure:
        lines.append("Detection processing failed; no pose reported.")

    # Keep every decoded marker in the message, including rejected poses.
    for result in marker_results:
        lines.append(
            _marker_summary(result, bool(failure))
        )

    # Include the source clock value, frame number and run ID for later verification.
    device_time = record.get("device_timestamp_s")

    if _finite_number(device_time):
        time_text = f"{device_time:.3f} s"

    else:
        time_text = "unavailable"

    lines.append(
        f'Frame {record["sequence_number"]} | '
        f'Device time {time_text} | Run {record["run_id"]}'
    )

    return {
        "action": action,
        "content": "\n".join(lines),
    }


def _target_image_position(box):
    """Describe the centre of a gauge or valve box in the original frame."""

    # Require four finite edges that form a box with positive width and height.
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not all(_finite_number(value) for value in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        return "image position unavailable"

    # Halve each edge before adding to avoid overflowing with large coordinates.
    centre_x = box[0] / 2 + box[2] / 2
    centre_y = box[1] / 2 + box[3] / 2
    return f"image position ({centre_x:g}, {centre_y:g}) px"


def gcs_log_payloads(record: dict) -> list[dict]:
    """Prepare the ArUco log, then one log for each detected gauge or valve."""

    payloads = []

    # Add the combined ArUco message first when this frame has one.
    aruco_message = gcs_log_payload(record)

    if aruco_message is not None:
        payloads.append(aruco_message)

    processor = _find_processor(record, "targets")

    # Keep any ArUco message even when target detection was not run.
    if processor is None:
        return payloads

    prefix = ""

    if record.get("source_kind") == "synthetic":
        prefix = "Synthetic demo. "

    frame = f'Frame {record["sequence_number"]}'

    # Report a failed target detector once, without using its candidate readings.
    if processor.get("failure_reason") is not None:
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

            # Use the reading after confirmation, rather than the raw candidate value.
            reading = result.get("gauge_reading")
            summary = "Pressure unavailable"

            if result.get("reading_status") == "pending":
                summary = "Pressure unavailable (awaiting confirmation)"

            # Include pressure only when the reading passed validation and is not pending.
            # Older records without a confirmation status retain their accepted reading.
            if (
                isinstance(reading, dict)
                and reading.get("valid") is True
                and reading.get("failure_reason") is None
                and _finite_number(reading.get("value_bar"))
                and result.get("reading_status") in (None, "confirmed")
            ):
                # Keep up to two decimal places, removing unnecessary trailing zeros.
                pressure = f'{reading["value_bar"]:.2f}'.rstrip("0").rstrip(".")
                summary = f"Pressure: {pressure} bar"

        elif class_name in ("openValve", "closedValve"):
            action = "Valve detected"

            # Use the valve state already identified by the target model.
            if class_name == "openValve":
                summary = "Open valve"

            else:
                summary = "Closed valve"

        else:
            # Only the recognised gauge and valve classes have GCS target messages.
            continue

        position = _target_image_position(result.get("bbox_xyxy_px"))

        # Give each target its own short log with its image position and frame number.
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
    """Package a selected photo for the GCS, preserving the file's image bytes."""

    # Require text for the photo's category and caption in the GCS history.
    if (
        not isinstance(target_type, str)
        or not isinstance(description, str)
    ):
        raise ValueError("Detection type and description must be strings")

    if timestamp is not None:
        # Validate a supplied capture time, keeping its original text for the message.
        if not isinstance(timestamp, str):
            raise ValueError(
                "Detection timestamp must be an ISO 8601 string with a timezone"
            )

        try:
            parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        except ValueError as error:
            raise ValueError(
                "Detection timestamp must be an ISO 8601 string with a timezone"
            ) from error

        # A timezone is required so the GCS can interpret the capture time correctly.
        if parsed_time.utcoffset() is None:
            raise ValueError("Detection timestamp must include a timezone")

    path = Path(path)

    # Tell the GCS which image format to expect from the selected file.
    image_formats = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    mime = image_formats.get(path.suffix.lower())

    if mime is None:
        raise ValueError("Detection images must be .jpg, .jpeg or .png files")

    image_bytes = path.read_bytes()

    # Check the file's opening bytes agree with its JPEG or PNG extension.
    if mime == "image/jpeg":
        signature = b"\xff\xd8\xff"

    else:
        signature = b"\x89PNG\r\n\x1a\n"

    if not image_bytes.startswith(signature):
        raise ValueError("Image bytes do not match the file extension")

    # Decode once to check the photo can be opened; keep the original bytes for sending.
    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype = np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if (
        image is None
        or image.size == 0
    ):
        raise ValueError("Detection image could not be decoded")

    # Convert the selected file bytes to text for JSON without re-encoding the image.
    encoded_image = base64.b64encode(image_bytes).decode("ascii")

    # Without a supplied capture time, report when this message was prepared in UTC.
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "type": target_type,
        "description": description,
        "mime": mime,
        "image": encoded_image,
        "timestamp": timestamp,
    }
