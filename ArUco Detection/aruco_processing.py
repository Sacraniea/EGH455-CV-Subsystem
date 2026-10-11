"""Read one frame and keep its results, settings and capture details together."""

from dataclasses import asdict
from time import perf_counter

from aruco_output import result_to_record
from aruco_reader import read_aruco
from aruco_types import ArUcoConfig, CameraCalibration, FramePacketLike


def process_aruco_frame(
    packet: FramePacketLike,
    calibration: CameraCalibration | None,
    config: ArUcoConfig,
    *,
    max_reprojection_error_px: float,
):
    """Return marker objects for drawing and a JSON-ready record for saving."""
    # Run detection and pose estimation, then convert their elapsed time to milliseconds.
    start = perf_counter()
    results = read_aruco(
        packet,
        calibration,
        config,
        max_reprojection_error_px=max_reprojection_error_px,
    )
    elapsed_ms = (perf_counter() - start) * 1000

    calibration_record = None

    # Convert supplied calibration tuples to lists for the JSON-ready record.
    if calibration is not None:
        calibration_record = asdict(calibration)
        calibration_record["camera_matrix"] = [
            list(row) for row in calibration.camera_matrix
        ]
        calibration_record["distortion_coefficients"] = list(
            calibration.distortion_coefficients
        )
        calibration_record["image_size_px"] = list(calibration.image_size_px)

    # Combine frame identity, settings and all marker results, including empty frames.
    record = {
        "schema_version": 1,
        "producer": "aruco",
        "sequence_number": packet.sequence_number,
        "device_timestamp_s": packet.device_timestamp_s,
        "host_received_timestamp_s": packet.host_received_timestamp_s,
        "image_size_px": [packet.image.shape[1], packet.image.shape[0]],
        "corners_coordinate_frame": "source_px",
        "processing_time_ms": elapsed_ms,
        "config": {
            "dictionary_name": config.dictionary_name,
            "marker_side_length_m": config.marker_side_length_m,
            "permitted_ids": sorted(config.permitted_ids),
            "max_reprojection_error_px": max_reprojection_error_px,
        },
        "calibration": calibration_record,
        "results": [result_to_record(result) for result in results],
    }

    return results, record
