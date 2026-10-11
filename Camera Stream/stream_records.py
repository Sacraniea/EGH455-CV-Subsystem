"""Create persistent, machine-readable camera-stream diagnostic records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import cv2
import depthai as dai
import numpy as np

from stream_diagnostics import StreamSummary
from stream_types import DeviceDetails, StreamProfile


def _format_utc(moment: datetime) -> str:
    """Represent a timestamp unambiguously in UTC."""

    return (
        # Convert to UTC text with milliseconds and replace the offset with Z.
        moment.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def build_complete_run_record(
    *,
    profile: StreamProfile,
    device: DeviceDetails,
    summary: StreamSummary,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    source_kind: str = "camera",
) -> dict[str, object]:
    """Combine configuration, observations and environment into one record."""

    return {
        # Record the format version, source and completed run's start and end times.
        "schema_version": 1,
        "source_kind": source_kind,
        "status": "complete",
        "started_at_utc": _format_utc(started_at_utc),
        "finished_at_utc": _format_utc(finished_at_utc),
        # Copy the reason the frame loop stopped into the saved record.
        "stop_reason": summary.stop_reason,
        # Read this computer's platform, processor architecture and Python version.
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        # Read the versions of the libraries used to capture and process frames.
        "software": {
            "depthai_version": dai.__version__,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        },
        # Copy the connected device's identity and USB speed.
        "device": {
            "product_name": device.product_name,
            "mxid": device.mxid,
            "usb_speed": device.usb_speed,
        },
        # Copy the requested image dimensions and frame rate from the profile.
        "profile": {
            "name": profile.name,
            "requested_width": profile.width,
            "requested_height": profile.height,
            "requested_fps": profile.fps,
        },
        # Copy the frame counts, missing frames and timings measured during the run.
        "observed": {
            "frame_width": summary.frame_width,
            "frame_height": summary.frame_height,
            "frames_received": summary.frames_received,
            "duration_s": summary.duration_s,
            "received_fps": summary.received_fps,
            "sequence_gaps": summary.sequence_gaps,
            "latency_mean_ms": summary.latency_mean_ms,
            "latency_p95_ms": summary.latency_p95_ms,
        },
        # Describe which part of capture was timed and which stages were excluded.
        "measurement": {
            "latency_p95_window_size": summary.latency_p95_window_size,
            "latency_p95_sample_count": summary.latency_p95_sample_count,
            "latency_p95_scope": (
                "recent_window" if summary.latency_p95_window_size else "all_samples"
            ),
            "latency_boundary": (
                "synthetic_image_generation_start_to_finish"
                if source_kind == "synthetic"
                else "depthai_frame_timestamp_to_application_dequeue"
            ),
            "latency_excludes": (
                [] if source_kind == "synthetic" else ["bgr_conversion"]
            )
            + [
                "local_display",
                "computer_vision",
                "network_transport",
            ],
        },
    }


def write_complete_run_record(
    *,
    profile: StreamProfile,
    device: DeviceDetails,
    summary: StreamSummary,
    started_at_utc: datetime,
    finished_at_utc: datetime,
    directory: str | Path = "diagnostics/runs",
    source_kind: str = "camera",
) -> Path:
    """Write one completed diagnostic run and return its resulting path."""

    record = build_complete_run_record(
        profile=profile,
        device=device,
        summary=summary,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        source_kind=source_kind,
    )

    # Convert the record to JSON, rejecting invalid numbers before creating a file.
    serialized_record = (
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    output_directory = Path(directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build the output filename from the UTC start time and chosen stream profile.
    filename_timestamp = started_at_utc.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )

    output_path = output_directory / (f"{filename_timestamp}_{profile.name}.json")

    # Write the JSON to a temporary file beside the intended output file.
    temporary_path = output_path.with_suffix(".json.tmp")

    try:
        temporary_path.write_text(
            serialized_record,
            encoding="utf-8",
        )

        # Rename the completed temporary file to the final record filename.
        temporary_path.replace(output_path)

    finally:
        # Delete a temporary file left behind if writing or renaming failed.
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path
