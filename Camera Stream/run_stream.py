"""Acquire frames, run the detectors once, and send their video and GCS results."""

import argparse
from contextlib import ExitStack, closing
from dataclasses import asdict
from pathlib import Path
import signal
from time import monotonic
from uuid import uuid4

from NetGear_config import create_sender, send_frame, build_frame_record
from MQTT_config import create_mqtt_client, close_mqtt_client, publish_log
from detection_aruco import add_processing_arguments, load_calibration, prepare_aruco
from detection_targets import prepare_targets
from detection_pipeline import process_frame, ResultCompositor
from gcs_output import gcs_log_payloads
from session_records import SessionRecords, new_run_directory
from stream_types import get_stream_profile


def stream_frames(
    source,
    sender,
    processors,
    *,
    source_kind,
    mqtt_client = None,
    session = None,
    photos = None,
    max_frames = 0,
):
    """Reuse each frame's results for the overlay, evidence, text and photos."""

    run_id = session.run_id if session is not None else uuid4().hex
    compositor = ResultCompositor(processors)
    next_log_at = 0.0
    frames = 0

    while max_frames == 0 or frames < max_frames:
        # Every detector reads the same untouched source frame.
        packet = source.read()
        results = process_frame(packet, processors)

        for result in results:
            if result.failure_reason is not None:
                print(
                    f"Frame {packet.sequence_number}: {result.processor} failed — "
                    f"{result.failure_reason}; {result.error or 'no further details'}"
                )

        # Draw saved results on a separate image, without repeating detection.
        display_packet = compositor.compose(packet, results)
        record = build_frame_record(
            packet,
            results,
            run_id = run_id,
            source_kind = source_kind,
        )

        if session is not None:
            session.record(record)

        # Only selected photos enter the bounded worker queue; most frames skip it.
        if photos is not None:
            photos.submit(packet, display_packet, record)

        # Sample text logs while connected. Camera frames never wait for PUBACK.
        now = monotonic()
        if (
            mqtt_client is not None
            and mqtt_client.is_connected()
            and now >= next_log_at
        ):
            payloads = gcs_log_payloads(record)
            if payloads:
                next_log_at = now + 5.0
                for payload in payloads:
                    publish_log(mqtt_client, payload, event_id = record["event_id"])

        # Acquisition-only runs keep the same evidence without sending video.
        if sender is not None:
            # The transmitted annotations and measurements identify this exact frame.
            send_frame(sender, display_packet.image, record)
        frames += 1

    return frames


def build_parser():
    parser = argparse.ArgumentParser(
        description = "Run detection and video, with optional GCS logs and photos.",
    )
    parser.add_argument("gcs_broker", nargs = "?", help = "GCS broker; omit to keep results local.")
    parser.add_argument("--gcs-port", type = int, default = 1883)
    parser.add_argument(
        "--source",
        choices = ("oak", "synthetic"),
        default = "oak",
        help = "Capture source; defaults to the connected OAK camera.",
    )
    parser.add_argument(
        "--profile",
        choices = ("480p10", "480p15", "480p20", "720p15", "720p30", "1080p30"),
        default = "720p30",
    )
    parser.add_argument("--source-timeout-s", type = float, default = 10.0)
    parser.add_argument(
        "--no-targets",
        action = "store_true",
        help = "Skip gauge and valve detection.",
    )
    parser.add_argument("--no-aruco", action = "store_true", help = "Skip ArUco detection.")
    parser.add_argument("--target-model", type = Path)
    parser.add_argument(
        "--receiver-host",
        default = "127.0.0.1",
        help = "Computer running the NetGear receiver.",
    )
    parser.add_argument("--video-port", type = int, default = 5555)
    parser.add_argument(
        "--no-video",
        action = "store_true",
        help = "Process and save frames without opening a NetGear sender or requiring a receiver.",
    )
    parser.add_argument(
        "--max-frames",
        type = int,
        default = 0,
        help = "Stop after this many frames; zero runs until Ctrl+C.",
    )
    parser.add_argument(
        "--output-dir",
        type = Path,
        help = "New run folder; omission creates a unique folder.",
    )
    parser.add_argument(
        "--no-photos",
        action = "store_true",
        help = "Keep video and text results without saving detection images.",
    )
    parser.add_argument("--capture-interval-s", type = float, default = 3.0)
    parser.add_argument(
        "--photo-max-mbps",
        type = float,
        default = 1.0,
        help = "Average photo JSON budget; a packet can still burst.",
    )
    parser.add_argument("--capture-storage-mib", type = int, default = 256)
    add_processing_arguments(parser)
    parser.add_argument("--calibration-mode", choices = ("exact", "oak-1080"), default = "exact")
    return parser


def validate_options(parser, args):
    from math import isfinite

    # Reject impossible settings before opening files, hardware or connections.
    for name in ("gcs_port", "video_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")

    if args.max_frames < 0 or args.capture_storage_mib <= 0:
        parser.error("Frame limit must be nonnegative; storage must be positive")

    for name in ("source_timeout_s", "capture_interval_s", "photo_max_mbps"):
        value = getattr(args, name)
        if not isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")

    if args.capture_interval_s < 3:
        parser.error("--capture-interval-s must be at least 3 seconds")

    if args.source != "oak" and (args.calibration or args.calibration_mode != "exact"):
        parser.error("OAK calibration cannot be used with generated images")

    if args.no_aruco and args.calibration:
        parser.error("--calibration needs ArUco detection enabled")

    if args.calibration_mode != "exact" and not args.calibration:
        parser.error("--calibration-mode requires --calibration")


def open_source(args, profile):
    # Import the selected hardware adapter only when that source is requested.
    if args.source == "oak":
        from depthai_source import DepthAIFrameSource
        return DepthAIFrameSource(profile, read_timeout_s = args.source_timeout_s)

    if args.source == "synthetic":
        from synthetic_source import SyntheticFrameSource
        return SyntheticFrameSource(profile)

    raise ValueError(f"Unsupported capture source: {args.source}")


def main(argv = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_options(parser, args)
    profile = get_stream_profile(args.profile)
    directory = args.output_dir or new_run_directory()
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    session = None
    status, error_text = "complete", None

    # Let the launcher request the same cleanup as pressing Ctrl+C.
    def stop_stream(signum, frame):
        raise KeyboardInterrupt

    stop_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        stop_signals.append(signal.SIGBREAK)
    previous_signals = {number: signal.signal(number, stop_stream) for number in stop_signals}
    try:
        session = SessionRecords(directory, settings)
        print(f"Evidence: {session.directory}", flush = True)
        processors = []

        if not args.no_targets:
            print("Loading detection model...", flush = True)
            target_processor, metadata = prepare_targets(args.target_model)
            processors.append(target_processor)
            settings["targets"] = metadata

        aruco_processor = None
        if not args.no_aruco:
            aruco_processor, metadata = prepare_aruco(args, profile)
            processors.append(aruco_processor)
            settings["aruco"] = metadata

        with ExitStack() as resources:
            source = resources.enter_context(closing(open_source(args, profile)))
            details = getattr(source, "device_details", None)
            if details is not None and hasattr(details, "__dataclass_fields__"):
                settings["device"] = asdict(details)
                print(f"Source: {details.product_name} | Profile: {profile.name}")

            if args.calibration:
                # A correctly sized file can still belong to a different OAK camera.
                load_calibration(args.calibration, expected_device_mxid = details.mxid)

            if args.source == "synthetic" and aruco_processor is not None:
                # Generated intrinsics are only valid for these generated images.
                aruco_processor.calibration = source.calibration
                settings["aruco"]["calibration"] = asdict(source.calibration)

            sender = None
            if not args.no_video:
                sender = resources.enter_context(
                    closing(create_sender(args.receiver_host, args.video_port))
                )
            mqtt_client = None
            if args.gcs_broker:
                mqtt_client = create_mqtt_client(
                    args.gcs_broker,
                    args.gcs_port,
                    on_event = session.mqtt_event,
                )
                resources.callback(close_mqtt_client, mqtt_client)

            photos = None
            if not args.no_photos:
                from detection_photos import PhotoCapture
                photos = PhotoCapture(
                    session.directory / "detections",
                    broker = args.gcs_broker,
                    port = args.gcs_port,
                    interval_s = args.capture_interval_s,
                    max_storage_bytes = args.capture_storage_mib * 1024 * 1024,
                    max_mbps = args.photo_max_mbps,
                )
                resources.callback(photos.close)

            source_kind = "camera" if args.source == "oak" else args.source
            print(
                f"{args.source.capitalize()} stream started. "
                f"Detectors: {', '.join(p.name for p in processors) or 'none'}. "
                "Press Ctrl+C to stop.",
                flush = True,
            )
            stream_frames(
                source,
                sender,
                processors,
                source_kind = source_kind,
                mqtt_client = mqtt_client,
                session = session,
                photos = photos,
                max_frames = args.max_frames,
            )

    except KeyboardInterrupt:
        status = "stopped"
        print("\nStopping stream.")

    except Exception as error:
        status = "failed"
        error_text = f"{type(error).__name__}: {error}"
        print(f"Stream error: {error_text}")

    finally:
        for number, previous in previous_signals.items():
            signal.signal(number, previous)
        if session is not None:
            session.close(status, error_text)
            status = session.summary["status"]
            print(f"Frames recorded: {session.summary['frames_recorded']} | Status: {status}")

    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
