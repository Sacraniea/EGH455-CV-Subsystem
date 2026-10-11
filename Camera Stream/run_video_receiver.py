"""Receive local NetGear frames and expose an HTTP MJPEG preview on a chosen IPv4 host."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address
import json
from pathlib import Path
import signal
import sys
import threading
from time import monotonic
from uuid import uuid4

import cv2


def http_bind_host(value: str) -> str:
    """Select one concrete IPv4 interface; the operating system checks locality."""

    if value == "localhost":
        value = "127.0.0.1"

    try:
        address = IPv4Address(value)

    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "HTTP host must be a concrete IPv4 address or localhost"
        ) from error

    if (
        address.is_unspecified
        or address.is_multicast
        or address == IPv4Address("255.255.255.255")
    ):
        # Reject wildcard, multicast and broadcast addresses for the HTTP listener.
        raise argparse.ArgumentTypeError(
            "HTTP host cannot be unspecified, multicast or broadcast"
        )

    return str(address)


class LatestFrame:
    """One JPEG only. No history and no repeated publication of stale images."""

    def __init__(self, *, stale_after_s=2.0, clock=monotonic):
        self.condition = threading.Condition()
        self.closed = False
        self.generation = 0
        self.jpeg = None
        self.metadata = None
        self.received_at = None
        self.stale_after_s = stale_after_s
        self.clock = clock

    def publish(self, metadata, image):
        # Encode the received image as JPEG before locking the shared frame store.
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if not success:
            raise ValueError("Could not encode received frame for HTTP preview")

        with self.condition:
            # Replace the latest JPEG, metadata and receipt time under the same lock.
            self.jpeg = encoded.tobytes()
            self.metadata = dict(metadata)
            self.received_at = self.clock()
            self.generation += 1
            # Wake browser requests waiting for a newer frame.
            self.condition.notify_all()

    def snapshot(self):
        # Return the latest frame and copied metadata only while that frame is fresh.
        with self.condition:
            if (
                self.closed
                or self.jpeg is None
                or self.clock() - self.received_at > self.stale_after_s
            ):
                # Return no snapshot after shutdown, before arrival or after expiry.
                return None

            return self.generation, self.jpeg, dict(self.metadata)

    def health(self):
        # Calculate frame age and report whether the stream is waiting, fresh or stale.
        with self.condition:
            age = None if self.received_at is None else self.clock() - self.received_at

            if self.closed:
                status = "closed"

            elif age is None:
                status = "waiting"

            elif age > self.stale_after_s:
                status = "stale"

            else:
                status = "receiving"

            return dict(
                status=status,
                frames_received=self.generation,
                age_s=age,
                latest_identity=self.metadata,
            )

    def close(self):
        with self.condition:
            self.closed = True
            # Wake streaming HTTP requests so shutdown does not leave them waiting.
            self.condition.notify_all()


def handler_for(store):
    # Build HTTP handlers that read pictures and status from this shared frame store.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Suppress the HTTP server's default per-request terminal messages.
            pass

        def respond(self, status, kind, body):
            # Send the response type, length and body with browser caching disabled.
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Choose between status JSON, a single JPEG and a continuous JPEG stream.
            try:
                if self.path in ("/health", "/frame.json"):
                    # Return stream status and the latest frame identity as JSON.
                    self.respond(
                        200,
                        "application/json",
                        json.dumps(store.health(), allow_nan=False).encode(),
                    )

                elif self.path == "/snapshot.jpg":
                    # Send the latest fresh picture, or HTTP 503 when none is available.
                    snapshot = store.snapshot()

                    if snapshot is None:
                        self.respond(
                            503,
                            "text/plain",
                            b"No fresh frame is available.\n",
                        )

                    else:
                        self.respond(200, "image/jpeg", snapshot[1])

                elif self.path in ("/", "/stream.mjpeg", "/stream.mjpg"):
                    # Serve the same video at the current address and the old .mjpg link.
                    # Keep the response open to send each fresh JPEG to the browser.
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    # Track which stored frame has already been sent to this browser.
                    generation = 0

                    while not store.closed:
                        with store.condition:
                            # Wait up to one second for a new frame or shutdown.
                            store.condition.wait_for(
                                lambda: (
                                    store.closed
                                    or store.generation != generation
                                ),
                                timeout=1.0,
                            )

                        snapshot = store.snapshot()

                        if snapshot is None:
                            # Mark this stale frame seen so the next loop waits.
                            with store.condition:
                                generation = store.generation

                            continue

                        new_generation, jpeg, metadata = snapshot

                        if new_generation == generation:
                            # Skip this picture if the browser already received it.
                            continue

                        generation = new_generation
                        # Send frame/run IDs in the header, followed by JPEG bytes.
                        headers = (
                            f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n"
                            f'X-Frame-Sequence: {metadata["sequence_number"]}\r\n'
                            f'X-Run-ID: {metadata["run_id"]}\r\n\r\n'
                        ).encode()
                        self.wfile.write(headers + jpeg + b"\r\n")
                        self.wfile.flush()

                else:
                    self.respond(
                        404,
                        "text/plain",
                        b"Use /stream.mjpeg, /snapshot.jpg or /frame.json\n",
                    )

            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                # End only this browser request if its connection closes or times out.
                pass

    return Handler


def main(argv=None):
    # Receive NetGear frames, serve them over HTTP and save receiver status at shutdown.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-port", type=int, default=5555)
    parser.add_argument(
        "--idle-timeout-s",
        type=int,
        default=10,
        help="Seconds to wait before the first frame or between frames (default: 10).",
    )
    parser.add_argument(
        "--http-host",
        type=http_bind_host,
        default="127.0.0.1",
        help="Concrete local IPv4 interface for HTTP preview (default: 127.0.0.1).",
    )
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)

    if any(not 1 <= p <= 65535 for p in (args.video_port, args.http_port)):
        parser.error("Ports must be between 1 and 65535")

    if args.idle_timeout_s <= 0:
        parser.error("--idle-timeout-s must be a positive integer")

    from stream_video import NetGearVideoReceiver

    output = args.output_dir or Path("diagnostics/receivers") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    store = LatestFrame()
    receiver = server = thread = None
    status = "failed"
    stop_reason = "exception"
    error_text = None
    exit_code = 0

    try:
        receiver = NetGearVideoReceiver(
            port=args.video_port,
            subscriber_timeout_s=args.idle_timeout_s,
        )
        # Start HTTP serving on a background thread while this thread receives video.
        server = ThreadingHTTPServer(
            (args.http_host, args.http_port),
            handler_for(store),
        )
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        preview_url = f"http://{args.http_host}:{args.http_port}/stream.mjpeg"

        # Advertise readiness only after the video receiver and HTTP server exist.
        if args.ready_file:
            with args.ready_file.open("x", encoding="utf-8") as file:
                json.dump({"url": preview_url}, file)

        print(f"Video receiver ready: {preview_url}", flush=True)

        with (output / "frames.jsonl").open("x", encoding="utf-8") as records:
            while True:
                # Read the next frame; stop on termination or idle timeout.
                item = receiver.receive()

                if item is None:
                    break

                metadata, image = item
                store.publish(metadata, image)
                # Append this received frame's metadata once to the JSONL evidence file.
                records.write(json.dumps(metadata, allow_nan=False) + "\n")
                records.flush()

        # Record the shared outcome NetGear uses for both termination and idle timeout.
        status = "ended"
        stop_reason = "termination_or_idle_timeout"

    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        exit_code = 130

    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        raise

    finally:
        # Ignore further Ctrl+C signals while closing resources and saving final status.
        previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            active_error = sys.exception()
            cleanup_errors = []
            actions = [store.close]

            if server is not None:
                # Stop the HTTP serving loop only if its thread is actually running.
                if thread is not None and thread.is_alive():
                    actions.append(server.shutdown)

                actions.append(server.server_close)

            if receiver is not None:
                actions.append(receiver.close)

            if store.jpeg is not None:
                actions.append(lambda: (output / "latest.jpg").write_bytes(store.jpeg))

            # Run all cleanup actions and collect any errors they raise.
            for action in actions:
                try:
                    action()

                except Exception as cleanup_error:
                    cleanup_errors.append(cleanup_error)

                    if active_error is not None:
                        active_error.add_note(
                            f"Receiver cleanup also failed: {cleanup_error}"
                        )

            if cleanup_errors and active_error is None:
                # Report the first cleanup error when there was no earlier run error.
                status = "failed"
                stop_reason = "cleanup_exception"
                error_text = f"{type(cleanup_errors[0]).__name__}: {cleanup_errors[0]}"

            health = store.health()
            # Keep final run status separate from the frame store's closed/stale status.
            health["stream_status"] = health.pop("status")
            final_status = dict(
                status=status,
                stop_reason=stop_reason,
                error=error_text,
                cleanup_errors=[
                    f"{type(error).__name__}: {error}" for error in cleanup_errors
                ],
                **health,
            )

            try:
                (output / "status.json").write_text(
                    json.dumps(final_status, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )

            except Exception as finalisation_error:
                # Preserve the original error if writing status.json also fails.
                primary_error = active_error or (
                    cleanup_errors[0] if cleanup_errors else None
                )

                if primary_error is None:
                    raise

                primary_error.add_note(
                    f"Receiver status finalisation also failed: {finalisation_error}"
                )

            if cleanup_errors and active_error is None:
                raise cleanup_errors[0]

        finally:
            # Restore the original Ctrl+C handler after finalisation is complete.
            signal.signal(signal.SIGINT, previous_sigint_handler)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
