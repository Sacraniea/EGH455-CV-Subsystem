from math import isfinite

import cv2
import numpy as np

from aruco_types import (
    ArUcoFailureReason,
    CameraCalibration,
    MarkerObservation,
    MarkerPose,
)


def marker_object_points(
    marker_side_length_m: float,
) -> np.ndarray:
    # Build the marker's four 3D corners in metres from its measured side length.
    half_side = marker_side_length_m / 2.0

    # Place all corners at Z=0 around the centre, in the square solver's required order.
    return np.array(
        [
            [-half_side, half_side, 0.0],
            [half_side, half_side, 0.0],
            [half_side, -half_side, 0.0],
            [-half_side, -half_side, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_marker_pose(
    observation: MarkerObservation,
    calibration: CameraCalibration,
    *,
    marker_side_length_m: float,
) -> MarkerPose:
    # Estimate the marker's position and rotation from its corners and calibration.
    object_points = marker_object_points(marker_side_length_m)

    # Convert the saved tuples to the floating-point arrays expected by OpenCV.
    image_points = np.asarray(
        observation.corners_px,
        dtype=np.float64,
    )
    camera_matrix = np.asarray(
        calibration.camera_matrix,
        dtype=np.float64,
    )
    distortion = np.asarray(
        calibration.distortion_coefficients,
        dtype=np.float64,
    )

    # Match the known square in metres to its detected corners in pixels.
    success, rotation, translation = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    # Stop here if OpenCV could not find an initial pose for the four corners.
    if not success:
        raise ValueError("Marker pose estimation failed")

    # Refine the first estimate to better fit the observed corners.
    rotation, translation = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        rotation,
        translation,
    )

    # Project the estimate back into the image to measure its corner error.
    projected_corners, _ = cv2.projectPoints(
        object_points,
        rotation,
        translation,
        camera_matrix,
        distortion,
    )

    error_px = reprojection_rmse_px(
        image_points,
        projected_corners.reshape(4, 2),
    )

    # Flatten OpenCV's vectors into three-value tuples for the returned pose.
    return MarkerPose(
        translation_m=tuple(float(value) for value in translation.reshape(3)),
        rotation_vector_rad=tuple(float(value) for value in rotation.reshape(3)),
        reprojection_error_px=error_px,
    )


def reprojection_rmse_px(
    observed_corners: np.ndarray,
    projected_corners: np.ndarray,
) -> float:
    # Return the root-mean-square distance between observed and predicted corners.
    differences = projected_corners - observed_corners
    # Square each corner's pixel distance, then average before taking the root.
    squared_distances = np.sum(differences**2, axis=1)

    return float(np.sqrt(np.mean(squared_distances)))


def pose_rejection_reason(
    pose: MarkerPose,
    *,
    max_reprojection_error_px: float,
) -> ArUcoFailureReason | None:
    # Return the first reason to reject this pose, or None when every check passes.
    validate_reprojection_limit(max_reprojection_error_px)

    # Reject any position or rotation component containing infinity or NaN.
    for vector in (pose.translation_m, pose.rotation_vector_rad):
        for value in vector:
            if not isfinite(value):
                return ArUcoFailureReason.NONFINITE_POSE

    # Reject a marker positioned at the camera plane or behind the camera.
    if pose.translation_m[2] <= 0:
        return ArUcoFailureReason.NONPOSITIVE_DEPTH

    # Reject an unusable pixel error before comparing it with the allowed limit.
    if not isfinite(pose.reprojection_error_px):
        return ArUcoFailureReason.NONFINITE_POSE

    # Reject the pose if its corner-fitting error exceeds the supplied pixel limit.
    if pose.reprojection_error_px > max_reprojection_error_px:
        return ArUcoFailureReason.REPROJECTION_ERROR_TOO_HIGH

    return None


def validate_reprojection_limit(limit: float) -> None:
    # Raise ValueError unless the allowed pixel error is finite and greater than zero.
    if (
        not isfinite(limit)
        or limit <= 0
    ):
        raise ValueError("Maximum reprojection error must be finite and positive")
