from math import isfinite

from aruco_types import ArUcoReadResult


def result_to_record(result: ArUcoReadResult) -> dict:
    # Convert one marker result into a JSON-ready record while retaining its rejection.
    pose_record = None

    # Only export a pose when it passed the reader's checks.
    if result.valid:
        pose_record = {
            "translation_m": list(result.pose.translation_m),
            "rotation_vector_rad": list(result.pose.rotation_vector_rad),
            "reprojection_error_px": result.pose.reprojection_error_px,
            "coordinate_frame": result.pose.coordinate_frame,
        }

    error_px = result.reprojection_error_px

    # Replace an infinite or NaN pixel error with None so JSON writes it as null.
    if (
        error_px is not None
        and not isfinite(error_px)
    ):
        error_px = None

    # Include the source corners and timestamps whether or not the pose was accepted.
    return {
        "schema_version": 1,
        "marker_id": result.marker_id,
        "corners_px": [list(corner) for corner in result.corners_px],
        "valid": result.valid,
        "pose": pose_record,
        "failure_reason": (
            result.failure_reason.value
            if result.failure_reason is not None
            else None
        ),
        "sequence_number": result.sequence_number,
        "device_timestamp_s": result.device_timestamp_s,
        "host_received_timestamp_s": result.host_received_timestamp_s,
        "reprojection_error_px": error_px,
    }
