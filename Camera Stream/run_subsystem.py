"""Start the local receiver before the camera sender and own their lifetime."""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import webbrowser
from time import monotonic, sleep
from uuid import uuid4


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "diagnostics" / "runs"


def child_options():
    """Isolate child signals so terminal Ctrl+C is handled once by this launcher."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    return {"start_new_session": True}


def stop_child(child):
    """Stop only a process created here, allowing it to flush evidence first."""
    if child is None or child.poll() is not None:
        return

    try:
        stop_signal = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
        child.send_signal(stop_signal)
        # Photo shutdown can take five seconds; MQTT and evidence also need time.
        child.wait(timeout = 8)
        return

    except subprocess.TimeoutExpired:
        pass

    except ProcessLookupError:
        return

    try:
        child.terminate()
        child.wait(timeout = 2)
        return

    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout = 2)

    except ProcessLookupError:
        return


def exit_status(returncode):
    return returncode if returncode >= 0 else 128 - returncode


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv = None):
    parser = argparse.ArgumentParser(
        description = "Run the receiver and sender together. OAK is the default source.",
        epilog = (
            "Sender options are forwarded to run_stream.py:\n"
            "  GCS_ADDRESS                    Publish logs and photos to this broker.\n"
            "  --source {oak,synthetic}        Select the capture source.\n"
            "  --profile NAME                 Choose 480p20, 720p15, 720p30 or 1080p30.\n"
            "  --max-frames N                 Stop after N frames; zero runs until Ctrl+C.\n"
            "  --no-targets / --no-aruco       Skip a detector.\n"
            "  --no-photos                    Skip detection photo capture.\n"
            "  --calibration FILE             Use explicit OAK calibration.\n"
            "  --calibration-mode MODE        Select exact or oak-1080 geometry.\n"
            "  --marker-side-length-m SIZE    Measured marker side, in metres.\n"
            "  --gcs-port PORT                MQTT broker port (default: 1883).\n"
            "Use run_stream.py --help for all sender options. For acquisition only,\n"
            "run run_stream.py --no-video without this browser receiver launcher."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev = False,
    )
    parser.add_argument(
        "--http-host",
        default = "127.0.0.1",
        help = "Browser listener address.",
    )
    parser.add_argument(
        "--http-port",
        type = int,
        default = 8080,
        help = "Browser stream port.",
    )
    parser.add_argument(
        "--video-host", "--receiver-host",
        dest = "video_host",
        default = "127.0.0.1",
        help = "Local NetGear listener address.",
    )
    parser.add_argument(
        "--video-port",
        type = int,
        default = 5555,
        help = "Local NetGear port.",
    )
    parser.add_argument(
        "--startup-timeout",
        type = float,
        default = 30.0,
        help = "Seconds to wait for both receiver listeners (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        type = Path,
        help = "New run directory; defaults to diagnostics/runs/<UTC>-<UUID>.",
    )
    parser.add_argument(
        "--open-browser",
        action = "store_true",
        help = "Open the local viewing URL after the receiver is ready.",
    )
    args, sender_args = parser.parse_known_args(argv)

    for option in ("http_port", "video_port"):
        if not 1 <= getattr(args, option) <= 65535:
            parser.error(f"--{option.replace('_', '-')} must be between 1 and 65535")

    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be a positive finite number")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.output_dir or RUNS_DIR / f"{stamp}-{uuid4().hex[:12]}").resolve()

    try:
        run_dir.mkdir(parents = True, exist_ok = False)

    except OSError as error:
        print(
            f"Subsystem error: cannot create a new run directory {run_dir}: {error}",
            file = sys.stderr,
        )
        return 2

    log_path = run_dir / "receiver.log"
    print(f"Run evidence: {run_dir}", flush = True)
    sender = receiver = None
    old_handlers = {}

    try:
        for name in ("SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                old_handlers[signum] = signal.signal(signum, interrupt)

        with log_path.open("x", encoding = "utf-8") as log, tempfile.TemporaryDirectory(
            prefix = "cv-ready-",
        ) as ready_dir:
            ready_file = Path(ready_dir) / "receiver.ready"
            receiver_command = [
                sys.executable, "-B", "-u", str(SCRIPT_DIR / "run_receiver.py"),
                "--http-host", args.http_host,
                "--http-port", str(args.http_port),
                "--video-host", args.video_host,
                "--video-port", str(args.video_port),
                "--ready-file", str(ready_file),
            ]
            receiver = subprocess.Popen(
                receiver_command,
                cwd = SCRIPT_DIR,
                stdout = log,
                stderr = subprocess.STDOUT,
                **child_options(),
            )
            deadline = monotonic() + args.startup_timeout

            while True:
                receiver_status = receiver.poll()

                if receiver_status is not None:
                    print(
                        f"Receiver exited before ready ({receiver_status}); see {log_path}",
                        file = sys.stderr,
                    )
                    return exit_status(receiver_status) or 1

                if ready_file.is_file():
                    break

                if monotonic() >= deadline:
                    print(
                        f"Receiver was not ready within {args.startup_timeout:g}s; "
                        f"see {log_path}",
                        file = sys.stderr,
                    )
                    return 1

                sleep(0.05)

            browser_host = "<LAN-IP>" if args.http_host == "0.0.0.0" else args.http_host
            print(
                f"Browser stream: http://{browser_host}:{args.http_port}/stream.mjpeg",
                flush = True,
            )
            if args.open_browser:
                # A wildcard listens on every interface; use loopback for this browser.
                local_host = "127.0.0.1" if args.http_host == "0.0.0.0" else args.http_host
                local_url = f"http://{local_host}:{args.http_port}/stream.mjpeg"
                try:
                    browser_opened = webbrowser.open(local_url)

                except (OSError, webbrowser.Error):
                    browser_opened = False

                if not browser_opened:
                    # A Pi without a desktop can still serve the stream normally.
                    print(
                        f"Could not open a browser automatically. Open {local_url} manually.",
                        file = sys.stderr,
                    )

            # A wildcard is valid for binding but cannot be a sender destination.
            receiver_host = "127.0.0.1" if args.video_host == "0.0.0.0" else args.video_host
            sender_command = [
                sys.executable, "-B", "-u", str(SCRIPT_DIR / "run_stream.py"), *sender_args,
                "--receiver-host", receiver_host,
                "--video-port", str(args.video_port),
                "--output-dir", str(run_dir),
            ]
            sender = subprocess.Popen(
                sender_command,
                cwd = SCRIPT_DIR,
                **child_options(),
            )

            while True:
                sender_status = sender.poll()

                if sender_status is not None:
                    if sender_status:
                        return exit_status(sender_status)

                    # Give the receiver time to consume the final finite-run frames.
                    try:
                        receiver_status = receiver.wait(timeout = 5)

                    except subprocess.TimeoutExpired:
                        return 0

                    if receiver_status:
                        print(
                            f"Receiver failed ({receiver_status}); see {log_path}",
                            file = sys.stderr,
                        )

                    return exit_status(receiver_status)

                receiver_status = receiver.poll()

                if receiver_status is not None:
                    print(
                        f"Receiver stopped while sender was active ({receiver_status}); "
                        f"see {log_path}",
                        file = sys.stderr,
                    )
                    return exit_status(receiver_status) or 1

                sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping subsystem.", flush = True)
        return 130

    except OSError as error:
        print(
            f"Subsystem error: {error}; receiver log: {log_path}",
            file = sys.stderr,
        )
        return 1

    finally:
        # A repeated Ctrl+C must not abandon the children halfway through cleanup.
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                previous = signal.signal(signum, signal.SIG_IGN)
                old_handlers.setdefault(signum, previous)

        try:
            stop_child(sender)
            stop_child(receiver)

        finally:
            for signum, previous in old_handlers.items():
                signal.signal(signum, previous)


if __name__ == "__main__":
    sys.exit(main())
