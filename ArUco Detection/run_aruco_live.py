"""Run local OAK ArUco processing with a preview and durable JSONL evidence."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from uuid import uuid4

# Add the sibling camera-code directory to the import search path for this script.
STREAM_DIRECTORY = Path(__file__).resolve().parent.parent / "Camera Stream"

if str(STREAM_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STREAM_DIRECTORY))

import cv2
import numpy as np
from stream_types import get_stream_profile

from aruco_calibration import load_calibration, load_stream_calibration
from aruco_cli import add_processing_arguments, config_from_args
from aruco_live import ArUcoDisplay, OpenCVPreview, run_live
from aruco_records import ArUcoRunRecorder


def positive_int(value):
    # Convert the frame limit to an integer and reject zero or negative limits.
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number


def default_display_factory(*, window_name):
    # The standalone preview needs no video sender or network receiver.
    return OpenCVPreview(window_name=window_name)


def main(argv=None, *, source_factory=None, display_factory=None, printer=print):
    # Configure the camera, record every processed frame and display ArUco results.
    parser = argparse.ArgumentParser(description=__doc__)
    add_processing_arguments(parser)
    parser.add_argument(
        "--profile",
        choices=("1080p30", "720p30"),
        default="1080p30",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("exact", "oak-1080"),
        default="exact",
        help="exact uses supplied geometry; oak-1080 explicitly resizes/crops factory intrinsics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New evidence directory; defaults to runs/<unique run>.",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Stop after this many processed frames.",
    )
    args = parser.parse_args(argv)
    recorder = None
    status, reason, error_text, summary = "failed", "exception", None, None
    exit_code = 1

    try:
        config = config_from_args(args)
        profile = get_stream_profile(args.profile)

        # Reject a requested calibration transform when no calibration file was given.
        if (
            args.calibration_mode != "exact"
            and args.calibration is None
        ):
            raise ValueError("--calibration-mode requires --calibration")

        calibration = None
        calibration_evidence = None

        if args.calibration:
            # Load this profile's calibration and the file evidence saved with the run.
            calibration, calibration_evidence = load_stream_calibration(
                args.calibration,
                image_size_px=(profile.width, profile.height),
                mode=args.calibration_mode,
            )

        # Build a unique default folder name, then prefer a supplied output directory.
        name = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
        )
        output_dir = (
            args.output_dir
            or Path(__file__).resolve().parent / "runs" / name
        )
        # Start the frame log with camera settings, calibration and software versions.
        recorder = ArUcoRunRecorder(
            output_dir,
            dict(
                profile=asdict(profile),
                marker_side_length_m=config.marker_side_length_m,
                dictionary_name=config.dictionary_name,
                permitted_ids=sorted(config.permitted_ids),
                max_reprojection_error_px=args.max_reprojection_error_px,
                calibration=calibration_evidence,
                max_frames=args.max_frames,
                software=dict(
                    python=platform.python_version(),
                    opencv=cv2.__version__,
                    numpy=np.__version__,
                ),
                measurement=dict(
                    processing_time_ms="detection and pose only; excludes overlay, disk and display",
                    sampling="every processed frame; camera queue may drop frames",
                    physical_accuracy="not established by software tests",
                ),
            ),
        )

        # Use the real OAK source unless a caller supplied a replacement, such as a test camera.
        if source_factory is None:
            import depthai
            from depthai_source import DepthAIFrameSource

            source_factory = DepthAIFrameSource
            recorder.update_metadata(depthai_version=depthai.__version__)

        if display_factory is None:
            display_factory = default_display_factory

        # Register camera and display cleanup while the remaining startup steps run.
        with ExitStack() as resources:
            source = source_factory(profile)
            resources.callback(source.close)
            recorder.update_metadata(device=asdict(source.device_details))

            # Reject a calibration file whose device ID differs from the connected camera.
            if args.calibration:
                load_calibration(
                    args.calibration,
                    expected_device_mxid=source.device_details.mxid,
                )

            printer(
                f"Camera: {source.device_details.mxid} | USB: {source.device_details.usb_speed}"
            )
            printer(
                f"Profile: {profile.name} | marker side: {config.marker_side_length_m:.3f} m"
            )

            # Warn about a size mismatch; marker detection can still run without valid poses.
            if (
                calibration
                and calibration.image_size_px != (profile.width, profile.height)
            ):
                printer(
                    "Calibration dimensions differ: detected markers will retain CALIBRATION_SIZE_MISMATCH."
                )

            if args.calibration_mode == "oak-1080":
                printer(
                    "Using explicit OAK 1080p resize/centre-crop geometry; physical validation remains required."
                )

            printer(
                f"Evidence: {recorder.output_dir} | press q in the preview to stop."
            )
            display = display_factory(window_name=f"ArUco — {profile.name}")
            resources.callback(display.close)
            # Send every frame record to the recorder and its annotated image to the display.
            adapter = ArUcoDisplay(
                display,
                calibration=calibration,
                config=config,
                max_reprojection_error_px=args.max_reprojection_error_px,
                on_result=recorder.write,
                max_frames=args.max_frames,
            )
            # This ExitStack owns camera/display cleanup on every exit path.
            summary = run_live(source, adapter, requested_fps=profile.fps)
            status, reason, exit_code = "complete", adapter.stop_reason, 0

    except KeyboardInterrupt:
        # Record Ctrl+C as an interrupted run and return its usual command exit code.
        status, reason, exit_code = "interrupted", "keyboard_interrupt", 130

    except Exception as error:
        # Keep the default failed status and save the exception text in the run record.
        error_text = f"{type(error).__name__}: {error}"
        printer(f"ArUco live error: {error_text}")

    finally:
        if recorder is not None:
            # Close the recorder with the outcome, stop reason and any captured error.
            try:
                recorder.finish(
                    status=status,
                    stop_reason=reason,
                    stream_summary=summary,
                    error=error_text,
                )

            except Exception as error:
                # A failure to save the final record makes the command fail as well.
                printer(f"Could not finalise run evidence: {error}")
                status = "failed"
                exit_code = 1

    if recorder is not None:
        printer(f"Run {status}: {recorder.output_dir}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
