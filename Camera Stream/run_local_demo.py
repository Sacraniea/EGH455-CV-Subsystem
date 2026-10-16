"""Run the CV subsystem and optional GCS forwarders from one command."""

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.request import urlopen
from uuid import uuid4
import webbrowser
from run_video_receiver import http_bind_host
from stream_types import STREAM_PROFILE_NAMES


def check_ports(ports, *, http_host="127.0.0.1"):
    # Check that the three service ports differ and can accept a local listener.
    if len(set(ports)) != len(ports):
        raise ValueError("MQTT, video and HTTP ports must be different")

    for index, port in enumerate(ports):
        # Check MQTT and video on localhost, and HTTP on the chosen interface.
        host = http_host if index == 2 else "127.0.0.1"

        if not 1 <= port <= 65535:
            raise ValueError("Ports must be between 1 and 65535")

        with socket.socket() as probe:
            try:
                # Allow reuse after an old connection closes, then try listening.
                # A service already listening on this address makes the check fail.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, port))
                probe.listen(1)

            except OSError as error:
                raise OSError(
                    f"Local endpoint {host}:{port} is already in use or unavailable; "
                    "check the interface address and port. No existing process was stopped."
                ) from error


def records(path, *, allow_partial=False):
    # Read saved JSON lines, or return no records if the file has not appeared yet.
    if not path.exists():
        return []

    content = path.read_text()

    # Remove an unfinished final line when reading a file that is still being written.
    if (
        allow_partial
        and content
        and not content.endswith("\n")
    ):
        content = content.rpartition("\n")[0]

    return [json.loads(line) for line in content.splitlines() if line.strip()]


def verify_delivery(root, *, allow_partial=False):
    # Compare the sender's records with the received MQTT results and video frames.
    sent = records(root / "sender/frames.jsonl", allow_partial=allow_partial)
    mqtt = records(root / "mqtt.jsonl", allow_partial=allow_partial)
    video = records(root / "receiver/frames.jsonl", allow_partial=allow_partial)
    # Identify each frame by its run ID and sequence number across all three files.
    def frame_key(record):
        return record["run_id"], record["sequence_number"]

    sent_by_id = {frame_key(r): r for r in sent}
    mqtt_ids = {frame_key(r) for r in mqtt}
    video_ids = {frame_key(r) for r in video}
    identity_fields = (
        "run_id",
        "sequence_number",
        "device_timestamp_s",
        "host_received_timestamp_s",
        "image_size_px",
        "source_kind",
    )
    # Require MQTT to contain every sent frame with identical data; allow duplicates.
    mqtt_matches = (
        bool(sent)
        and mqtt_ids == set(sent_by_id)
        and all(r == sent_by_id.get(frame_key(r)) for r in mqtt)
    )
    # Require at least one video frame and check every received frame's identity.
    # Do not require video delivery of every frame, because this transport can drop it.
    video_matches = bool(video)

    for received in video:
        original = sent_by_id.get(frame_key(received))

        if (
            original is None
            or any(received[field] != original[field] for field in identity_fields)
        ):
            video_matches = False
            break

    # Return the match results and counts, including frames received by both routes.
    return dict(
        verified=mqtt_matches and video_matches,
        sender_frames=len(sent),
        mqtt_messages=len(mqtt),
        mqtt_unique_frames=len(mqtt_ids),
        video_frames=len(video),
        matched_video_and_mqtt_frames=len(video_ids & mqtt_ids),
        all_mqtt_results_match=mqtt_matches,
        all_received_video_identities_match=video_matches,
        note="QoS 1 may duplicate messages. PUB/SUB video may drop frames; every received identity is checked.",
    )


def complete_trial(root, children):
    """Stop writers before checking their complete files and final exit codes."""
    children.close()
    delivery = verify_delivery(root)
    # These consumers run until the launcher asks them to stop.
    interrupted_services = {
        "video-receiver",
        "mqtt-subscriber",
        "gcs-log-forwarder",
        "gcs-image-forwarder",
    }
    failed = {}

    for name, process, _ in children.running:
        accepted_codes = (0, 130) if name in interrupted_services else (0,)

        if process.returncode not in accepted_codes:
            failed[name] = process.returncode

    if failed:
        # Mark delivery unverified if any service failed during shutdown.
        delivery.update(
            verified=False,
            error=f"Child process failure during finalization: {failed}",
        )

    elif not delivery["verified"]:
        # Record that the final files did not match the sender's frames.
        delivery["error"] = (
            "Final delivery verification failed; inspect individual logs"
        )

    return delivery


class Children:
    """Own only this invocation's children; bounded shutdown, never kill by name."""

    def __init__(self, root):
        # Prepare a log folder and a list of the child processes started by this run.
        self.root = root
        self.running = []
        (root / "logs").mkdir()

    def start(self, name, script, arguments):
        # Start the sibling Python script and send its output and errors to one log.
        log = (self.root / "logs" / f"{name}.log").open("x", encoding="utf-8")

        # Give each child its own interrupt group on the current operating system.
        process_options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )

        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-B",
                    str(Path(__file__).with_name(script)),
                    *map(str, arguments),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                **process_options,
            )

        except BaseException:
            # Close the log if the child process could not be started.
            log.close()
            raise

        self.running.append((name, process, log))
        return process

    def ready(self, process, marker, seconds=10.0):
        # Wait for the child's ready file, failing if it exits or misses the deadline.
        deadline = monotonic() + seconds

        while monotonic() < deadline:
            exit_code = process.poll()

            # A short finished sender can still leave a complete capture directory.
            if (
                marker.exists()
                and (
                    exit_code is None
                    or (marker.is_dir() and exit_code == 0)
                )
            ):
                return

            if exit_code is not None:
                raise RuntimeError(
                    f'Child exited before readiness; inspect {self.root / "logs"}'
                )

            sleep(0.05)

        raise TimeoutError(f'Child startup timed out; inspect {self.root / "logs"}')

    def close(self):
        stop_signal = (
            signal.CTRL_BREAK_EVENT
            if sys.platform == "win32"
            else signal.SIGINT
        )
        # Ask the sender and consumers to stop first, leaving the broker running.
        for name, process, _ in reversed(self.running):
            if (
                name != "broker"
                and process.poll() is None
            ):
                process.send_signal(stop_signal)

        # Wait for each child to finish, then stop the broker after its consumers.
        for name, process, log in reversed(self.running):
            try:
                if (
                    name == "broker"
                    and process.poll() is None
                ):
                    process.send_signal(stop_signal)

                try:
                    process.wait(timeout=5.0)

                except subprocess.TimeoutExpired:
                    # Kill this child if it has not stopped after five seconds.
                    process.kill()
                    process.wait(timeout=3.0)

            finally:
                # Close the child's log even if waiting for its exit fails.
                log.close()


def build_parser():
    # Define the source, service addresses and capture options for the local trial.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use the attached OAK instead of the synthetic marker.",
    )
    parser.add_argument(
        "--targets",
        action="store_true",
        help="Run target detection and gauge reading in the sender process.",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        help="Override the sender's target model weights; requires --targets.",
    )
    parser.add_argument(
        "--profile",
        choices=STREAM_PROFILE_NAMES,
        default="1080p30",
        help="Capture resolution and requested frame rate (default: 1080p30).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=900,
        help="Processed frame limit; 0 runs until Ctrl+C (default: 900).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mqtt-port", type=int, default=18883)
    parser.add_argument("--video-port", type=int, default=5555)
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument(
        "--http-host",
        type=http_bind_host,
        default="127.0.0.1",
        help="HTTP viewing interface: localhost by default, or this Mac's LAN IPv4 address.",
    )
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--capture-detections",
        action="store_true",
        help="Save original ArUco sightings and periodic gauge/valve photos.",
    )
    parser.add_argument(
        "--capture-interval-s",
        type=float,
        default=3.0,
        help="Seconds between gauge/valve photo attempts (default: 3).",
    )
    parser.add_argument("--capture-max-pending", type=int, default=3)
    parser.add_argument("--capture-storage-mib", type=float, default=256.0)
    parser.add_argument("--marker-side-length-m", type=float, default=0.2)
    parser.add_argument(
        "--permitted-ids",
        type=int,
        nargs="+",
        help="Restrict accepted marker IDs; omission accepts all DICT_5X5_100 IDs (0-99).",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--calibration-mode",
        choices=("exact", "oak-1080"),
        default="exact",
    )

    # An address on its own sends both logs and photos to that GCS broker.
    parser.add_argument(
        "gcs_address",
        nargs="?",
        metavar="GCS_ADDRESS",
        help="Publish logs and photos to this broker unless --gcs overrides the mode.",
    )

    # Auto previews locally until a broker address is supplied.
    parser.add_argument(
        "--gcs",
        choices=("auto", "off", "preview", "publish"),
        help="GCS mode: auto publishes with an address, otherwise previews; omitted means off unless GCS_ADDRESS is supplied.",
    )

    # Use this destination when publishing logs and detection images.
    parser.add_argument(
        "--gcs-broker",
        help="GCS broker hostname or IP address.",
    )
    parser.add_argument(
        "--gcs-port",
        type=int,
        default=1883,
        help="GCS broker port; defaults to 1883.",
    )

    return parser


def _sender_arguments(args, root):
    """Pass the selected camera, detection and output settings to the sender."""

    # Tell the sender which profile, markers, frame count and ports to use.
    sender_args = [
        "--profile",
        args.profile,
        "--aruco",
        "--marker-side-length-m",
        args.marker_side_length_m,
        "--headless",
        "--max-frames",
        args.max_frames,
        "--mqtt",
        "--mqtt-port",
        args.mqtt_port,
        "--video",
        "--video-port",
        args.video_port,
        "--output-dir",
        root / "sender",
    ]

    if args.permitted_ids is not None:
        # Forward an explicit restriction; otherwise the sender accepts every ID.
        sender_args.extend(["--permitted-ids", *args.permitted_ids])

    if not args.camera:
        # Use generated marker images unless the camera was explicitly selected.
        sender_args.append("--synthetic")

    if args.targets:
        # Let the sender load the model once and process its own source frames.
        sender_args.append("--targets")

        if args.target_model:
            sender_args.extend(
                [
                    "--target-model",
                    args.target_model,
                ]
            )

    if args.capture_detections:
        # Pass the original-image capture limits to the sender.
        sender_args.extend(
            [
                "--capture-detections",
                "--capture-interval-s",
                args.capture_interval_s,
                "--capture-max-pending",
                args.capture_max_pending,
                "--capture-storage-mib",
                args.capture_storage_mib,
            ]
        )

    if args.calibration:
        # Pass the absolute camera calibration path and its geometry mode.
        sender_args.extend(
            [
                "--calibration",
                args.calibration.resolve(),
                "--calibration-mode",
                args.calibration_mode,
            ]
        )

    return sender_args


def main(argv=None):
    # Start the services together; verify delivery when a finite trial finishes.
    parser = build_parser()
    args = parser.parse_args(argv)

    # Remove accidental spaces around the broker address.
    if args.gcs_broker:
        args.gcs_broker = args.gcs_broker.strip()

    # Accept the short command while keeping the named broker option available.
    if args.gcs_address is not None:
        address = args.gcs_address.strip()

        if not address:
            parser.error("GCS_ADDRESS must not be empty")

        if (
            args.gcs_broker
            and args.gcs_broker != address
        ):
            parser.error("GCS_ADDRESS and --gcs-broker must name the same broker")

        args.gcs_broker = address

    # Keep an explicit preview or off choice, even when an address was supplied.
    if args.gcs is None:
        args.gcs = "publish" if args.gcs_address is not None else "off"

    elif args.gcs == "auto":
        args.gcs = "publish" if args.gcs_broker else "preview"

    # Publishing needs a destination; local previews do not.
    if (
        args.gcs == "publish"
        and not args.gcs_broker
    ):
        parser.error("--gcs publish requires --gcs-broker")

    if args.gcs != "off":
        # Check the destination port before starting any services.
        if not 1 <= args.gcs_port <= 65535:
            parser.error("--gcs-port must be between 1 and 65535")

        # Both GCS modes need saved photos for the image forwarder.
        args.capture_detections = True

    # Reject invalid frame limits, storage limits and camera-only options.
    if args.max_frames < 0:
        parser.error("--max-frames must be zero or positive")

    if (
        args.capture_max_pending <= 0
        or not isfinite(args.capture_interval_s)
        or args.capture_interval_s <= 0
        or not isfinite(args.capture_storage_mib)
        or args.capture_storage_mib <= 0
    ):
        parser.error("Capture interval, queue and storage limits must be positive and finite")

    if (
        not args.camera
        and (
            args.calibration
            or args.calibration_mode != "exact"
            or args.marker_side_length_m != 0.2
        )
    ):
        parser.error(
            "Synthetic mode uses its own 200 mm marker and invented calibration; camera options require --camera"
        )

    if (
        args.calibration_mode != "exact"
        and not args.calibration
    ):
        parser.error("--calibration-mode requires --calibration")

    if (
        args.target_model
        and not args.targets
    ):
        parser.error("--target-model requires --targets")

    if args.target_model:
        # Resolve the caller's model path before any child changes working directory.
        args.target_model = args.target_model.expanduser().resolve()

    # Check all listening addresses before creating files or starting services.
    try:
        check_ports(
            [args.mqtt_port, args.video_port, args.http_port],
            http_host=args.http_host,
        )

    except (OSError, ValueError) as error:
        parser.error(str(error))

    # Create a new trial folder and start its summary in the failed state.
    root = (
        args.output_dir
        or Path("diagnostics/local-network")
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=False)
    children = Children(root)
    url = f"http://{args.http_host}:{args.http_port}/stream.mjpeg"
    summary = dict(
        status="failed",
        source_kind="camera" if args.camera else "synthetic",
        preview_url=url,
        gcs_mode=args.gcs,
        gcs_broker=args.gcs_broker if args.gcs == "publish" else None,
        gcs_port=args.gcs_port if args.gcs == "publish" else None,
    )
    exit_code = 1

    try:
        # Start the broker and video receiver, waiting for each to finish setup.
        broker = children.start(
            "broker",
            "run_local_broker.py",
            ["--port", args.mqtt_port, "--ready-file", root / "broker.ready.json"],
        )
        children.ready(broker, root / "broker.ready.json")
        receiver_args = [
            "--video-port",
            args.video_port,
            "--http-port",
            args.http_port,
            "--http-host",
            args.http_host,
            "--output-dir",
            root / "receiver",
            "--ready-file",
            root / "receiver.ready.json",
        ]

        if args.targets:
            # Keep the receiver waiting while the sender imports and loads the model.
            receiver_args.extend(["--idle-timeout-s", 60])

        receiver = children.start(
            "video-receiver",
            "run_video_receiver.py",
            receiver_args,
        )
        children.ready(receiver, root / "receiver.ready.json")
        # Finite tests allow time for slower processors; the main run has no deadline.
        max_runtime = (
            max(60.0, args.max_frames / 30.0 * 12.0 + 60.0)
            if args.max_frames
            else None
        )
        subscriber_runtime = max_runtime + 15.0 if max_runtime is not None else 0
        # Start the MQTT subscriber and wait until it is ready to receive results.
        subscriber = children.start(
            "mqtt-subscriber",
            "run_mqtt_subscriber.py",
            [
                "--port",
                args.mqtt_port,
                "--timeout-s",
                subscriber_runtime,
                "--output",
                root / "mqtt.jsonl",
                "--ready-file",
                root / "subscriber.ready",
            ],
        )
        children.ready(subscriber, root / "subscriber.ready")

        # Reuse the existing forwarders with one destination and this run's files.
        gcs_processes = {}
        gcs_destination = []

        if args.gcs == "publish":
            gcs_destination = [
                "--publish",
                "--broker", args.gcs_broker,
                "--port", args.gcs_port,
            ]

        if args.gcs != "off":
            # Listen before the sender can produce its first detection result.
            log_forwarder = children.start(
                "gcs-log-forwarder",
                "run_gcs_forwarder.py",
                [
                    "--local-port", args.mqtt_port,
                    "--interval-s", 5,
                    "--timeout-s", subscriber_runtime,
                    "--output", root / "gcs-logs.jsonl",
                    "--ready-file", root / "gcs-logs.ready",
                    *gcs_destination,
                ],
            )
            gcs_processes["gcs-log-forwarder"] = log_forwarder
            children.ready(log_forwarder, root / "gcs-logs.ready")

        sender_args = _sender_arguments(args, root)

        # Start capture after the receiving services are ready and print their paths.
        sender = children.start("sender", "run_camera_stream.py", sender_args)
        deadline = monotonic() + max_runtime if max_runtime is not None else None

        if args.gcs != "off":
            capture_dir = root / "sender" / "detections"

            # The sender owns this directory; wait for it rather than creating it here.
            children.ready(sender, capture_dir, seconds=60)
            image_forwarder = children.start(
                "gcs-image-forwarder",
                "run_gcs_image_forwarder.py",
                [
                    "--capture-dir", capture_dir,
                    "--interval-s", 3,
                    "--max-mbps", 1,
                    "--publish-timeout-s", 60,
                    "--timeout-s", 0,
                    *gcs_destination,
                ],
            )
            gcs_processes["gcs-image-forwarder"] = image_forwarder
            print(
                f"GCS mode: {args.gcs}\n"
                f'GCS log records: {root / "gcs-logs.jsonl"}\n'
                f'Forwarder output: {root / "logs"}',
                flush=True,
            )

        print(
            f'{"CAMERA" if args.camera else "SYNTHETIC DEMO — no camera"}\n'
            f"Profile: {args.profile}\n"
            f"Video: {url}\nMQTT: 127.0.0.1:{args.mqtt_port}, topic taip/local/results\n"
            f"Evidence: {root}\nCtrl+C stops all processes started by this run.",
            flush=True,
        )

        if args.capture_detections:
            print(
                f'Detection capture queue: {root / "sender" / "detections"}',
                flush=True,
            )

        if args.open_browser:
            webbrowser.open(url)

        next_report = 0.0

        # Watch child exits and the time limit while the sender is still running.
        while sender.poll() is None:
            # Report a stopped forwarder rather than silently losing GCS output.
            for name, process in gcs_processes.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} stopped during capture; "
                        f"inspect {root / 'logs' / (name + '.log')}"
                    )

            if (
                broker.poll() is not None
                or subscriber.poll() is not None
            ):
                raise RuntimeError(
                    "Broker/subscriber exited during capture; inspect logs"
                )

            if receiver.poll() not in (None, 0):
                raise RuntimeError(
                    "Video receiver failed; inspect logs/video-receiver.log"
                )

            if deadline is not None and monotonic() > deadline:
                raise TimeoutError("Sender exceeded the finite trial time limit")

            if monotonic() >= next_report:
                # Read and print the receiver's latest frame count every two seconds.
                try:
                    with urlopen(
                        f"http://{args.http_host}:{args.http_port}/health",
                        timeout=0.5,
                    ) as response:
                        health = json.load(response)

                    latest = health.get("latest_identity") or {}
                    print(
                        f'Video {health["status"]}: received={health["frames_received"]}, '
                        f'latest sequence={latest.get("sequence_number", "—")}',
                        flush=True,
                    )

                except OSError:
                    # Continue if the health endpoint is unavailable during shutdown.
                    pass

                next_report = monotonic() + 2.0

            sleep(0.05)

        if sender.returncode != 0:
            raise RuntimeError(
                f"Sender exited {sender.returncode}; inspect logs/sender.log"
            )

        # Wait up to three seconds for received files to catch up with the sender.
        deadline = monotonic() + 3.0

        while monotonic() < deadline:
            delivery = verify_delivery(root, allow_partial=True)

            if delivery["verified"]:
                break

            sleep(0.1)

        # Allow final verification to run once the receiver and subscriber stop.
        exit_code = 0

    except KeyboardInterrupt:
        # Record Ctrl+C as an interrupted trial and return exit code 130.
        summary["status"] = "interrupted"
        exit_code = 130

    except Exception as error:
        # Save and print the failure while leaving the trial's exit code unsuccessful.
        summary["error"] = f"{type(error).__name__}: {error}"
        print(summary["error"], file=sys.stderr, flush=True)

    finally:
        if exit_code == 0:
            # Stop the child writers, verify their final files and set the outcome.
            try:
                summary.update(complete_trial(root, children))
                summary["status"] = "complete" if summary["verified"] else "failed"

                if not summary["verified"]:
                    exit_code = 1

            except Exception as error:
                # Record a shutdown or verification exception as a failed trial.
                summary.update(
                    status="failed",
                    verified=False,
                    error=f"{type(error).__name__}: {error}",
                )
                exit_code = 1

        else:
            # Stop all child processes after a failed or interrupted trial.
            children.close()

        # Save every child's exit code and the final trial outcome in summary.json.
        summary["child_exit_codes"] = {
            name: p.returncode for name, p, _ in children.running
        }
        if args.gcs != "off":
            # Local delivery checks do not establish whether the GCS stored a message.
            summary["gcs_delivery_note"] = (
                "Inspect gcs-logs.jsonl and sender/detections receipts. "
                "Shutdown does not drain pending photos; saved files are retained. "
                "A broker acknowledgement does not confirm GCS database ingestion."
            )

        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )

    if exit_code == 0:
        print(
            f'Verified {summary["mqtt_unique_frames"]} MQTT frame records and '
            f'{summary["video_frames"]} received video frames. All local services stopped.\n'
            f'Summary: {root / "summary.json"}',
            flush=True,
        )

    return exit_code


if __name__ == "__main__":
    # Windows sends Ctrl+Break to child groups; run the usual interrupt cleanup.
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    raise SystemExit(main())
