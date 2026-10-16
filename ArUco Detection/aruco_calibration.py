from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from aruco_types import CameraCalibration


def resize_calibration(
    calibration: CameraCalibration,
    *,
    image_size_px: tuple[int, int],
) -> CameraCalibration:
    # Return calibration for an image resized with the same width-to-height ratio.
    original_width, original_height = calibration.image_size_px
    new_width, new_height = image_size_px

    # Raise ValueError if the resize would stretch width and height differently.
    if new_width * original_height != original_width * new_height:
        raise ValueError("Calibration resizing must preserve the aspect ratio.")

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    matrix = calibration.camera_matrix

    # Scale the first two matrix rows by the width and height changes; keep the last row.
    resized_matrix = (
        tuple(value * scale_x for value in matrix[0]),
        tuple(value * scale_y for value in matrix[1]),
        matrix[2],
    )

    return CameraCalibration(
        camera_matrix=resized_matrix,
        distortion_coefficients=calibration.distortion_coefficients,
        image_size_px=image_size_px,
        source=calibration.source,
    )


def crop_calibration(
    calibration: CameraCalibration,
    *,
    origin_px: tuple[int, int],
    image_size_px: tuple[int, int],
) -> CameraCalibration:
    # Return calibration for the requested crop, using the original camera matrix.
    offset_x, offset_y = origin_px

    original_width, original_height = calibration.image_size_px
    crop_width, crop_height = image_size_px

    # If any crop edge falls outside the original image, raise ValueError.
    if (
        offset_x < 0
        or offset_y < 0
        or offset_x + crop_width > original_width
        or offset_y + crop_height > original_height
    ):
        raise ValueError("Crop must lie within the original image dimensions.")

    matrix = calibration.camera_matrix

    # Subtract the crop origin from the optical centre; keep focal lengths unchanged.
    cropped_matrix = (
        (matrix[0][0], matrix[0][1], matrix[0][2] - offset_x),
        (matrix[1][0], matrix[1][1], matrix[1][2] - offset_y),
        matrix[2],
    )

    # Carry over the lens distortion values and record the cropped image size.
    return CameraCalibration(
        camera_matrix=cropped_matrix,
        distortion_coefficients=calibration.distortion_coefficients,
        image_size_px=image_size_px,
        source=calibration.source,
    )


def load_calibration(
    path: str | Path,
    *,
    expected_device_mxid: str | None = None,
) -> CameraCalibration:
    # Load a CAM_A perspective-camera JSON file and return its calibration values.
    calibration_path = Path(path).resolve()

    with calibration_path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    # Reject a lens model or camera socket that this pose solver does not support.
    if record.get("distortion_model") != "CameraModel.Perspective":
        raise ValueError("Calibration must use the perspective camera model")

    if record.get("camera_socket") != "CAM_A":
        raise ValueError("Calibration must belong to the CAM_A colour camera")

    # If an expected camera ID was supplied, reject a file from another camera.
    if (
        expected_device_mxid is not None
        and record.get("device_mxid") != expected_device_mxid
    ):
        raise ValueError("Calibration device ID does not match the expected device")

    # Store JSON arrays as tuples and keep the resolved file path as their source.
    return CameraCalibration(
        camera_matrix=tuple(tuple(row) for row in record["camera_matrix"]),
        distortion_coefficients=tuple(record["distortion_coefficients"]),
        image_size_px=tuple(record["image_size_px"]),
        source=str(calibration_path),
    )


def prepare_stream_calibration(
    calibration: CameraCalibration,
    *,
    image_size_px: tuple[int, int],
    mode: str = "exact",
) -> CameraCalibration:
    """Adjust the calibration only when a camera mode was selected."""

    if mode == "exact":
        # Return the supplied calibration unchanged when no transform was requested.
        return calibration

    if mode != "oak-1080":
        raise ValueError(f"Unknown calibration mode: {mode}")

    # Reject input sizes that cannot use the supported OAK resize-and-crop geometry.
    if calibration.image_size_px not in {(3840, 2160), (1920, 1080)}:
        raise ValueError(
            "oak-1080 requires a 3840x2160 or 1920x1080 native calibration"
        )

    width, height = image_size_px

    # The OAK video output is a centre crop of the 1080p sensor frame.
    if not (0 < width <= 1920 and 0 < height <= 1080):
        raise ValueError(
            "oak-1080 supports only profiles that fit inside the 1920x1080 frame"
        )

    # First express the calibration in the full 1080p frame's pixel coordinates.
    full = resize_calibration(calibration, image_size_px=(1920, 1080))

    if image_size_px == (1920, 1080):
        return full

    # Remove the same margin from each side, e.g. 320x180 px for 720p.
    return crop_calibration(
        full,
        origin_px=((1920 - width) // 2, (1080 - height) // 2),
        image_size_px=image_size_px,
    )


def load_stream_calibration(
    path: str | Path,
    *,
    image_size_px: tuple[int, int],
    mode: str = "exact",
) -> tuple[CameraCalibration, dict]:
    """Load stream calibration and the original-file evidence saved by both runners."""
    calibration_path = Path(path)
    native = load_calibration(calibration_path)
    calibration = prepare_stream_calibration(
        native,
        image_size_px=image_size_px,
        mode=mode,
    )

    # Keep the file's original values and fingerprint beside the transformed values.
    raw = calibration_path.read_bytes()
    evidence = dict(
        file=str(calibration_path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        original_record=json.loads(raw),
        effective=asdict(calibration),
        transform=mode,
    )
    return calibration, evidence
