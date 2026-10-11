"""One OAK source, same-frame CV processing, annotated preview and JSONL evidence."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import platform
import sys
from uuid import uuid4

import cv2
import depthai
import numpy as np
from depthai_source import DepthAIFrameSource
from local_display import OpenCVDisplay
from stream_controller import run_stream
from stream_types import get_stream_profile
from stream_records import write_complete_run_record
from stream_session import StreamSession


def positive_int(value):
    # Convert the command-line value to an integer and reject zero or negatives.
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")

    return number


def positive_finite(value):
    # Convert the value to a number and reject zero, negatives or invalid numbers.
    number = float(value)

    if (
        not isfinite(number)
        or number <= 0
    ):
        raise argparse.ArgumentTypeError("must be finite and positive")

    return number


def build_parser():
    # Define the camera, processor, capture and output options for this command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("720p30", "1080p30"), default="720p30")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generated ID 7 demo, no camera; invented calibration.",
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Publish result JSON to the local test broker.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Send annotated frames to the local NetGear receiver.",
    )
    parser.add_argument("--mqtt-port", type=positive_int, default=18883)
    parser.add_argument("--video-port", type=positive_int, default=5555)
    parser.add_argument(
        "--aruco",
        action="store_true",
        help="Enable the independent ArUco branch.",
    )
    parser.add_argument(
        "--targets",
        action="store_true",
        help="Enable Aidan's gauge/valve model and read detected gauge crops.",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        help="Local model weights; defaults to Target Detection/models/50_epoch_best.pt.",
    )
    parser.add_argument(
        "--capture-detections",
        action="store_true",
        help="Save original ArUco sightings and periodic gauge/valve photos for GCS upload.",
    )
    parser.add_argument(
        "--capture-interval-s",
        type=positive_finite,
        default=3.0,
        help="Seconds between gauge/valve photo attempts (default: 3).",
    )
    parser.add_argument("--capture-max-pending", type=positive_int, default=3)
    parser.add_argument(
        "--capture-storage-mib",
        type=positive_finite,
        default=256.0,
        help="Session image-storage limit in MiB (default: 256); never deletes originals.",
    )
    parser.add_argument(
        "--permitted-ids",
        type=int,
        nargs="+",
        help="Restrict accepted marker IDs; omission accepts all DICT_5X5_100 IDs (0-99).",
    )
    parser.add_argument(
        "--marker-side-length-m",
        type=positive_finite,
        default=0.2,
        help="Measured outer black square side, metres (default: 0.2).",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--calibration-mode",
        choices=("exact", "oak-1080"),
        default="exact",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=positive_finite,
        default=1.0,
    )
    parser.add_argument(
        "--max-processing-ms",
        type=positive_finite,
        help="Optional per-branch elapsed budget; checked after return, not cancellation.",
    )
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No preview; requires --max-frames.",
    )
    parser.add_argument(
        "--source-timeout-s",
        type=positive_finite,
        default=5.0,
        help="Maximum wait for the next frame after device setup (default: 5 s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New session directory; existing data is never overwritten.",
    )
    return parser


class HeadlessDisplay:
    def show(self, packet):
        # Keep the frame loop running without opening a preview window.
        return True

    def close(self):
        pass


def _validate_options(parser, args):
    """Reject conflicting settings before opening files or the camera."""

    # Reject incompatible options before creating the run folder or opening a camera.
    if (
        args.headless
        and args.max_frames is None
    ):
        parser.error("--headless requires --max-frames")

    if (
        args.target_model
        and not args.targets
    ):
        parser.error("--target-model requires --targets")

    if (
        args.capture_detections
        and not (args.aruco or args.targets)
    ):
        parser.error("--capture-detections requires --aruco or --targets")

    if (
        not args.aruco
        and (
            args.calibration
            or args.permitted_ids
            or args.calibration_mode != "exact"
        )
    ):
        parser.error("ArUco calibration and IDs require --aruco")

    if (
        args.calibration_mode != "exact"
        and not args.calibration
    ):
        parser.error("--calibration-mode requires --calibration")

    if (
        args.synthetic
        and args.calibration
    ):
        parser.error("--synthetic uses invented calibration; omit --calibration")

    if max(args.mqtt_port, args.video_port) > 65535:
        parser.error("Network ports must be between 1 and 65535")


def _session_metadata(
    args, profile, processors, *, aruco_metadata, targets_metadata, source_kind
):
    """Record the requested settings so this run can be checked later."""

    return dict(
        profile=asdict(profile),
        aruco=aruco_metadata,
        targets=targets_metadata,
        source_kind=source_kind,
        distribution=dict(
            mqtt_port=args.mqtt_port if args.mqtt else None,
            video_port=args.video_port if args.video else None,
            host="127.0.0.1",
            topic="taip/local/results",
            mqtt_qos=1,
            mqtt_retain=False,
        ),
        enabled_processors=[p.name for p in processors],
        headless=args.headless,
        detection_capture=(
            dict(
                directory="detections",
                target_interval_s=args.capture_interval_s,
                max_pending=args.capture_max_pending,
                max_storage_bytes=int(args.capture_storage_mib * 1024 * 1024),
            )
            if args.capture_detections
            else None
        ),
        max_frames=args.max_frames,
        max_processing_ms=args.max_processing_ms,
        source_timeout_s=args.source_timeout_s,
        software=dict(
            python=platform.python_version(),
            opencv=cv2.__version__,
            numpy=np.__version__,
            depthai=depthai.__version__,
        ),
        measurement=dict(
            latency=(
                "Synthetic image generation duration using the host monotonic clock; no camera latency measured"
                if args.synthetic
                else "DepthAI timestamp to application dequeue; excludes conversion and CV"
            ),
            timings=(
                "Host perf_counter stage durations; source_read includes waiting and image generation"
                if args.synthetic
                else "Host perf_counter stage durations; source_read includes waiting and BGR conversion"
            ),
            percentile="Latency P95 over most recent up to 2048 processed frames; mean and counts cover whole run",
            end_to_end="Display call duration does not measure physical screen presentation or network latency",
        ),
    )


def main(
    argv=None,
    *,
    source_factory=None,
    display_factory=OpenCVDisplay,
    printer=print,
):
    # Validate the options, run the selected source and save the run's outcome.
    parser = build_parser()
    args = parser.parse_args(argv)

    _validate_options(parser, args)

    profile = get_stream_profile(args.profile)
    # Prepare each selected processor once, before opening the camera.
    processors = []
    aruco_processor = None
    aruco_metadata = targets_metadata = None

    if args.aruco:
        from stream_aruco import prepare_aruco

        aruco_processor, aruco_metadata = prepare_aruco(args, profile)
        processors.append(aruco_processor)

    if args.targets:
        from stream_target_detection import prepare_targets

        printer("Loading target model...")
        target_processor, targets_metadata = prepare_targets(args.target_model)
        processors.append(target_processor)

    # Choose a new run folder from the supplied path or a timestamp and random ID.
    started = datetime.now(timezone.utc)
    name = started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    source_kind = "synthetic" if args.synthetic else "camera"
    output_dir = args.output_dir or Path("diagnostics/sessions") / name

    # Create the session and save the requested settings before opening resources.
    session = StreamSession(
        output_dir,
        _session_metadata(
            args,
            profile,
            processors,
            aruco_metadata=aruco_metadata,
            targets_metadata=targets_metadata,
            source_kind=source_kind,
        ),
    )
    status, reason, error_text, summary = "failed", "exception", None, None

    try:
        # Register resources to close if setup or the frame loop raises an error.
        with ExitStack() as resources, ExitStack() as transports:
            # Keep network and capture cleanup here; run_stream closes the camera and display.
            publisher = video_sender = None
            capture = None

            if args.capture_detections:
                # Create the original-image queue and convert its MiB limit to bytes.
                from stream_detection_capture import DetectionCapture

                capture = DetectionCapture(
                    session.directory / "detections",
                    target_interval_s=args.capture_interval_s,
                    max_pending=args.capture_max_pending,
                    max_storage_bytes=int(args.capture_storage_mib * 1024 * 1024),
                )
                transports.callback(capture.close)

            if args.mqtt:
                # Open the result publisher and arrange to close it after the run.
                from stream_mqtt import MqttResultPublisher

                publisher = MqttResultPublisher(port=args.mqtt_port)
                transports.callback(publisher.close)

            if args.video:
                # Open the annotated-video sender and register its close method.
                from stream_video import NetGearVideoSender

                video_sender = NetGearVideoSender(port=args.video_port)
                transports.callback(video_sender.close)

            if args.synthetic:
                # Generate marker frames locally when the demo source is selected.
                from synthetic_source import SyntheticFrameSource

                source = SyntheticFrameSource(profile)

            else:
                # Open the OAK camera or use the source supplied by the caller.
                source = (
                    DepthAIFrameSource(profile, read_timeout_s=args.source_timeout_s)
                    if source_factory is None
                    else source_factory(profile)
                )

            resources.callback(source.close)

            if (
                args.synthetic
                and aruco_processor is not None
            ):
                # Give only ArUco the generated marker's made-up camera calibration.
                aruco_processor.calibration = source.calibration
                aruco_metadata["calibration"] = dict(
                    effective=asdict(source.calibration),
                    transform="synthetic",
                )
                session.update_metadata(aruco=aruco_metadata)

            # Add the source's device identity and USB speed to the saved settings.
            details = source.device_details
            session.update_metadata(device=asdict(details))

            if args.calibration:
                from aruco_calibration import load_calibration

                # Reject a calibration file recorded for a different camera MXID.
                load_calibration(args.calibration, expected_device_mxid=details.mxid)

            # Print the active device, profile, processors and calibration conditions.
            printer(f"Device: {details.product_name} | MXID: {details.mxid}")
            printer(f"USB: {details.usb_speed}")

            if (
                not args.synthetic
                and details.usb_speed not in ("SUPER", "SUPER_PLUS")
            ):
                printer("USB is below USB 3 speed; check recorded frame rate and gaps.")

            printer(
                f"Profile: {profile.name} | {profile.width}x{profile.height} at {profile.fps:.1f} FPS"
            )
            printer(
                f'Processors: {", ".join(p.name for p in processors) or "camera only"} | Evidence: {output_dir}'
            )

            if (
                args.aruco
                and not args.calibration
                and not args.synthetic
            ):
                printer(
                    "No calibration: marker evidence is retained; metric pose is withheld."
                )

            if args.calibration_mode == "oak-1080":
                printer(
                    "Explicit OAK 1080p resize/centre-crop geometry selected; validate physically."
                )

            printer(
                "Bounded headless run."
                if args.headless
                else "Press q in the video window to stop."
            )
            # Choose a preview window or a display that continues without a window.
            display = (
                HeadlessDisplay()
                if args.headless
                else display_factory(window_name=f"OAK-D Lite — {profile.name}")
            )
            resources.callback(display.close)

            if capture is not None:
                printer(
                    f'Detection capture queue: {(session.directory/"detections").resolve()}'
                )

            def retain_capture(packet, output, record):
                # Save the original evidence and the same annotated image used by the feed.
                record["run_id"] = session.manifest["run_id"]
                record["source_kind"] = source_kind
                capture(packet, record, annotated_packet=output)

            def distribute(packet, record):
                # Add the run identity before sending this frame and its measurements.
                record["run_id"] = session.manifest["run_id"]
                record["source_kind"] = source_kind

                if video_sender is not None:
                    # Send the annotated image with its run and source identity.
                    video_sender.send(
                        packet,
                        run_id=record["run_id"],
                        source_kind=source_kind,
                    )

                if publisher is not None:
                    # Publish this frame's complete result record over MQTT.
                    publisher.publish(record)

            # Remove source/display callbacks here; run_stream closes both on exit.
            resources.pop_all()
            summary = run_stream(
                source,
                display,
                requested_fps=profile.fps,
                processors=processors,
                on_frame=session.write,
                max_frames=args.max_frames,
                max_processing_ms=args.max_processing_ms,
                on_output=distribute,
                on_capture=retain_capture if capture is not None else None,
            )

        # Save the completed run's frame counts and capture timings as a JSON record.
        record_path = write_complete_run_record(
            profile=profile,
            device=details,
            summary=summary,
            started_at_utc=started,
            finished_at_utc=datetime.now(timezone.utc),
            source_kind=source_kind,
            directory=(
                output_dir / "acquisition" if args.output_dir else "diagnostics/runs"
            ),
        )
        # Mark the run complete and print its measured frame rate and timing summary.
        status, reason = "complete", summary.stop_reason
        printer(
            f"Frames received: {summary.frames_received} | Sequence gaps: {summary.sequence_gaps}"
        )
        printer(
            "Received FPS: unavailable"
            if summary.received_fps is None
            else f"Received FPS: {summary.received_fps:.3f}"
        )

        if summary.latency_mean_ms is not None:
            label = (
                "synthetic image generation time"
                if args.synthetic
                else "capture-to-dequeue latency"
            )
            printer(f"Mean {label}: {summary.latency_mean_ms:.3f} ms")
            printer(
                f"Recent P95 latency: {summary.latency_p95_ms:.3f} ms ({summary.latency_p95_sample_count} samples)"
            )

        printer(f"Diagnostic record: {record_path}")

    except KeyboardInterrupt:
        # Record an operator interruption and return the usual Ctrl+C exit code.
        status, reason = "interrupted", "keyboard_interrupt"
        return 130

    except Exception as error:
        # Save the exception text for the session summary, then report the exception.
        error_text = f"{type(error).__name__}: {error}"
        raise

    finally:
        # Save the final session status on success, interruption or failure.
        active_error = sys.exception()

        try:
            session.finish(
                status=status,
                stop_reason=reason,
                summary=asdict(summary) if summary else None,
                error=error_text,
            )

        except Exception as finalisation_error:
            if active_error is None:
                # Report the save error if no other exception is already being raised.
                raise

            # Otherwise attach the save error to the existing exception.
            active_error.add_note(
                f"Session finalisation also failed: {finalisation_error}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
