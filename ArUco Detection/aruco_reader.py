from collections import Counter

import cv2

from aruco_detection import detect_markers
from aruco_types import (
    ArUcoConfig,
    ArUcoFailureReason,
    ArUcoReadResult,
    CameraCalibration,
    FramePacketLike,
)
from aruco_pose import (
    estimate_marker_pose,
    pose_rejection_reason,
    validate_reprojection_limit,
)


def read_aruco(
    packet: FramePacketLike,
    calibration: CameraCalibration | None,
    config: ArUcoConfig,
    *,
    max_reprojection_error_px: float,
) -> tuple[ArUcoReadResult, ...]:
    # Read this frame's markers and return each detection with a pose or rejection.
    validate_reprojection_limit(max_reprojection_error_px)
    observations = detect_markers(packet.image, config)

    # Count each ID so a pose can be withheld when the same ID appears twice.
    id_counts = Counter(observation.marker_id for observation in observations)
    results = []

    # Handle each detection separately so one rejected marker cannot discard the rest.
    for observation in observations:
        pose = None
        error_px = None

        # Keep every detection; the first failed check sets its rejection reason.
        if observation.marker_id not in config.permitted_ids:
            reason = ArUcoFailureReason.MARKER_ID_NOT_ALLOWED

        elif id_counts[observation.marker_id] > 1:
            reason = ArUcoFailureReason.DUPLICATE_MARKER_ID

        elif calibration is None:
            reason = ArUcoFailureReason.MISSING_CALIBRATION

        elif calibration.image_size_px != (
            packet.image.shape[1],
            packet.image.shape[0],
        ):
            reason = ArUcoFailureReason.CALIBRATION_SIZE_MISMATCH

        else:
            # Estimate a pose only after the ID and calibration checks have passed.
            try:
                candidate_pose = estimate_marker_pose(
                    observation,
                    calibration,
                    marker_side_length_m=config.marker_side_length_m,
                )

            except (ValueError, cv2.error):
                # Record a solver failure for this marker and continue building results.
                reason = ArUcoFailureReason.POSE_SOLVER_FAILED

            else:
                # Keep the measured pixel error even if the pose is rejected next.
                error_px = candidate_pose.reprojection_error_px

                reason = pose_rejection_reason(
                    candidate_pose,
                    max_reprojection_error_px=max_reprojection_error_px,
                )

                # Only report the pose when it passes every quality check.
                if reason is None:
                    pose = candidate_pose

        # Attach the original corners and capture timestamps to every marker result.
        results.append(
            ArUcoReadResult(
                marker_id=observation.marker_id,
                corners_px=observation.corners_px,
                pose=pose,
                failure_reason=reason,
                sequence_number=packet.sequence_number,
                device_timestamp_s=packet.device_timestamp_s,
                host_received_timestamp_s=packet.host_received_timestamp_s,
                reprojection_error_px=error_px,
            )
        )

    return tuple(results)
