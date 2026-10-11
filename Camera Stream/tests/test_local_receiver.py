import argparse
from contextlib import contextmanager, redirect_stderr
from http.server import ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import signal
import socket
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import run_video_receiver
from run_video_receiver import LatestFrame, handler_for, main
from run_local_broker import broker_config


def available_port():
    # Ask the operating system for a free local port and return its number.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


@contextmanager
def http_preview(store):
    # Serve the frame store on a temporary local HTTP port and yield its base URL.
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"

    finally:
        # Wake MJPEG requests before waiting for the HTTP server to stop.
        store.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class LocalReceiverTests(unittest.TestCase):
    def test_only_new_fresh_frame_is_available_and_identity_is_preserved(self):
        # Publish and read at one second, then read at four seconds to make the frame stale.
        clock = iter([1.0, 1.0, 4.0]).__next__
        store = LatestFrame(stale_after_s=2.0, clock=clock)
        store.publish(
            {"run_id": "demo", "sequence_number": 12},
            np.zeros((4, 6, 3), np.uint8),
        )
        generation, jpeg, metadata = store.snapshot()
        self.assertEqual(generation, 1)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(metadata["sequence_number"], 12)
        self.assertIsNone(store.snapshot())

    def test_closed_receiver_exposes_no_current_frame(self):
        store = LatestFrame()
        store.publish(
            {"run_id": "demo", "sequence_number": 1},
            np.zeros((4, 6, 3), np.uint8),
        )
        store.close()
        self.assertIsNone(store.snapshot())
        self.assertEqual(store.health()["status"], "closed")

    def test_broker_is_loopback_only_and_has_connection_limit(self):
        config = broker_config(18883)
        self.assertEqual(config["listeners"]["default"]["bind"], "127.0.0.1:18883")
        self.assertEqual(config["listeners"]["default"]["max_connections"], 16)

    def test_http_snapshot_and_health_distinguish_waiting_fresh_stale_and_closed(self):
        now = [1.0]
        store = LatestFrame(clock=lambda: now[0])

        with http_preview(store) as base:
            with urlopen(base + "/health", timeout=2) as response:
                self.assertEqual(json.load(response)["status"], "waiting")

            with self.assertRaises(HTTPError) as unavailable:
                urlopen(base + "/snapshot.jpg", timeout=2)

            self.assertEqual(unavailable.exception.code, 503)
            unavailable.exception.close()

            store.publish(
                {"run_id": "demo", "sequence_number": 12},
                np.full((24, 32, 3), 80, np.uint8),
            )

            with urlopen(base + "/snapshot.jpg", timeout=2) as response:
                self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                decoded = cv2.imdecode(
                    np.frombuffer(response.read(), np.uint8),
                    cv2.IMREAD_COLOR,
                )

            self.assertEqual(decoded.shape, (24, 32, 3))

            with urlopen(base + "/frame.json", timeout=2) as response:
                identity = json.load(response)

            self.assertEqual(identity["status"], "receiving")
            self.assertEqual(identity["latest_identity"]["sequence_number"], 12)

            # Age the stored frame past its limit and check that HTTP stops serving its image.
            now[0] = 4.0

            with urlopen(base + "/health", timeout=2) as response:
                self.assertEqual(json.load(response)["status"], "stale")

            with self.assertRaises(HTTPError) as stale:
                urlopen(base + "/snapshot.jpg", timeout=2)

            self.assertEqual(stale.exception.code, 503)
            stale.exception.close()
            store.close()

            with urlopen(base + "/health", timeout=2) as response:
                self.assertEqual(json.load(response)["status"], "closed")

    def test_http_mjpeg_frame_headers_preserve_run_and_sequence_identity(self):
        # Check that both the advertised address and the old link serve a video frame.
        for path in ("/stream.mjpeg", "/stream.mjpg"):
            with self.subTest(path=path):
                store = LatestFrame()
                store.publish(
                    {"run_id": "demo", "sequence_number": 12},
                    np.zeros((24, 32, 3), np.uint8),
                )

                with http_preview(store) as base, urlopen(
                    base + path, timeout=2
                ) as response:
                    self.assertEqual(
                        response.headers["Content-Type"],
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.assertEqual(response.readline(), b"--frame\r\n")
                    headers = {}

                    # Read the frame's headers, then use its length to read one complete JPEG.
                    while (line := response.readline()) != b"\r\n":
                        key, value = line.decode().strip().split(": ", 1)
                        headers[key] = value

                    self.assertEqual(headers["X-Frame-Sequence"], "12")
                    self.assertEqual(headers["X-Run-ID"], "demo")
                    jpeg = response.read(int(headers["Content-Length"]))
                    self.assertTrue(jpeg.startswith(b"\xff\xd8"))
                    self.assertTrue(jpeg.endswith(b"\xff\xd9"))


class ReceiverCommandTests(unittest.TestCase):
    def test_http_host_accepts_concrete_ipv4_without_a_private_network_restriction(
        self,
    ):
        self.assertTrue(callable(getattr(run_video_receiver, "http_bind_host", None)))

        for value, expected in [
            ("localhost", "127.0.0.1"),
            ("127.0.0.1", "127.0.0.1"),
            ("192.168.68.10", "192.168.68.10"),
            ("8.8.8.8", "8.8.8.8"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(run_video_receiver.http_bind_host(value), expected)

    def test_http_host_rejects_unspecified_multicast_broadcast_and_non_ipv4(self):
        self.assertTrue(callable(getattr(run_video_receiver, "http_bind_host", None)))

        for value in (
            "0.0.0.0",
            "224.0.0.1",
            "239.255.255.250",
            "255.255.255.255",
            "::1",
            "example.com",
            "127.1",
            "192.168.68.10/24",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                run_video_receiver.http_bind_host(value)

    def test_selected_http_host_is_bound_and_advertised_without_changing_netgear(self):
        source = Mock(spec=["receive", "close"])
        port = available_port()
        requested_hosts = []

        def local_server(address, handler):
            requested_hosts.append(address)
            # Record the requested LAN address but bind the test HTTP server to local loopback.
            return ThreadingHTTPServer(("127.0.0.1", address[1]), handler)

        def receive_until_http_is_checked():
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                self.assertEqual(json.load(response)["status"], "waiting")

            return None

        source.receive.side_effect = receive_until_http_is_checked

        with TemporaryDirectory() as directory:
            output, ready = Path(directory) / "run", Path(directory) / "ready.json"

            with patch(
                "stream_video.NetGearVideoReceiver", return_value=source
            ) as video_factory, patch(
                "run_video_receiver.ThreadingHTTPServer", side_effect=local_server
            ), patch(
                "builtins.print"
            ) as printer:
                code = main(
                    [
                        "--http-host",
                        "192.168.68.10",
                        "--http-port",
                        str(port),
                        "--output-dir",
                        str(output),
                        "--ready-file",
                        str(ready),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(requested_hosts, [("192.168.68.10", port)])
            self.assertEqual(
                video_factory.call_args.kwargs.get("host", "127.0.0.1"), "127.0.0.1"
            )
            expected_url = f"http://192.168.68.10:{port}/stream.mjpeg"
            self.assertEqual(json.loads(ready.read_text())["url"], expected_url)
            printer.assert_called_with(
                f"Video receiver ready: {expected_url}", flush=True
            )

        source.close.assert_called_once()

    def test_cli_saves_received_frame_and_separate_process_and_stream_statuses(self):
        source = Mock(spec=["receive", "close"])
        metadata = {
            "run_id": "demo",
            "sequence_number": 12,
            "source_kind": "synthetic",
            "image_size_px": [32, 24],
        }
        image = np.full((24, 32, 3), 80, np.uint8)
        port = available_port()
        observed = {}

        def read_frame_then_check_http():
            # Supply one frame, then inspect its HTTP output before ending the input stream.
            if not observed:
                observed["read"] = True
                return metadata, image

            with urlopen(f"http://127.0.0.1:{port}/frame.json", timeout=2) as response:
                observed["http_health"] = json.load(response)

            with urlopen(
                f"http://127.0.0.1:{port}/snapshot.jpg", timeout=2
            ) as response:
                observed["http_jpeg"] = response.read()

            return None

        source.receive.side_effect = read_frame_then_check_http

        with TemporaryDirectory() as directory:
            output, ready = Path(directory) / "run", Path(directory) / "ready.json"

            with patch(
                "stream_video.NetGearVideoReceiver", return_value=source
            ) as factory:
                code = main(
                    [
                        "--http-port",
                        str(port),
                        "--output-dir",
                        str(output),
                        "--ready-file",
                        str(ready),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIs(type(factory.call_args.kwargs["subscriber_timeout_s"]), int)
            self.assertEqual(factory.call_args.kwargs["subscriber_timeout_s"], 10)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "ended")
            self.assertEqual(status["stop_reason"], "termination_or_idle_timeout")
            self.assertEqual(status["stream_status"], "closed")
            self.assertEqual(status["frames_received"], 1)
            self.assertIsNone(status["error"])
            self.assertEqual(
                json.loads((output / "frames.jsonl").read_text()), metadata
            )
            # Check that the saved JPEG is the exact image previously served over HTTP.
            self.assertEqual(
                (output / "latest.jpg").read_bytes(), observed["http_jpeg"]
            )
            self.assertEqual(observed["http_health"]["latest_identity"], metadata)
            self.assertEqual(
                json.loads(ready.read_text())["url"],
                f"http://127.0.0.1:{port}/stream.mjpeg",
            )

        source.close.assert_called_once()

        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)

    def test_idle_receiver_does_not_claim_stream_completion_or_create_a_fake_image(
        self,
    ):
        source = Mock(spec=["receive", "close"])
        source.receive.return_value = None

        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            with patch("stream_video.NetGearVideoReceiver", return_value=source):
                code = main(
                    ["--http-port", str(available_port()), "--output-dir", str(output)]
                )

            self.assertEqual(code, 0)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "ended")
            self.assertEqual(status["stop_reason"], "termination_or_idle_timeout")
            self.assertEqual(status["frames_received"], 0)
            self.assertFalse((output / "latest.jpg").exists())

        source.close.assert_called_once()

    def test_configured_idle_timeout_reaches_the_video_receiver(self):
        source = Mock(spec=["receive", "close"])
        source.receive.return_value = None

        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            with patch(
                "stream_video.NetGearVideoReceiver", return_value=source
            ) as factory:
                code = main(
                    [
                        "--http-port",
                        str(available_port()),
                        "--output-dir",
                        str(output),
                        "--idle-timeout-s",
                        "60",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(factory.call_args.kwargs["subscriber_timeout_s"], 60)
            self.assertIs(type(factory.call_args.kwargs["subscriber_timeout_s"]), int)

    def test_invalid_idle_timeout_is_rejected_before_receiver_startup(self):
        for value in ("0", "-1", "1.5", "forever"):
            with self.subTest(value=value), TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                errors = StringIO()

                with patch(
                    "stream_video.NetGearVideoReceiver",
                    side_effect=AssertionError("Invalid options started a receiver"),
                ), redirect_stderr(errors), self.assertRaises(SystemExit) as stopped:
                    main(["--output-dir", str(output), "--idle-timeout-s", value])

                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("--idle-timeout-s", errors.getvalue())
                self.assertFalse(output.exists())

    def test_keyboard_interrupt_records_interruption_and_closes_resources(self):
        source = Mock(spec=["receive", "close"])
        source.receive.side_effect = KeyboardInterrupt

        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            with patch("stream_video.NetGearVideoReceiver", return_value=source):
                code = main(
                    ["--http-port", str(available_port()), "--output-dir", str(output)]
                )

            self.assertEqual(code, 130)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(status["stop_reason"], "keyboard_interrupt")

        source.close.assert_called_once()

    def test_sigint_during_http_shutdown_preserves_evidence_and_restores_prior_handler(
        self,
    ):
        prior_handler = signal.getsignal(signal.SIGINT)

        def interrupt_handler(signum, frame):
            signal.default_int_handler(signum, frame)

        class InterruptDuringShutdown(ThreadingHTTPServer):
            def shutdown(self):
                try:
                    # Reproduce Ctrl+C arriving while cleanup is already underway.
                    signal.raise_signal(signal.SIGINT)

                finally:
                    super().shutdown()

        try:
            signal.signal(signal.SIGINT, interrupt_handler)

            for end, expected_status, expected_code in (
                (None, "ended", 0),
                (KeyboardInterrupt(), "interrupted", 130),
            ):
                with self.subTest(
                    status=expected_status
                ), TemporaryDirectory() as directory:
                    source = Mock(spec=["receive", "close"])
                    metadata = {"run_id": "demo", "sequence_number": 12}
                    source.receive.side_effect = [
                        (metadata, np.zeros((24, 32, 3), np.uint8)),
                        end,
                    ]
                    output = Path(directory) / "run"
                    servers = []

                    def create_server(address, handler):
                        server = InterruptDuringShutdown(address, handler)
                        servers.append(server)
                        return server

                    try:
                        with patch(
                            "stream_video.NetGearVideoReceiver", return_value=source
                        ), patch(
                            "run_video_receiver.ThreadingHTTPServer",
                            side_effect=create_server,
                        ):
                            try:
                                code = main(
                                    [
                                        "--http-port",
                                        str(available_port()),
                                        "--output-dir",
                                        str(output),
                                    ]
                                )

                            except KeyboardInterrupt:
                                self.fail(
                                    "SIGINT escaped cleanup before receiver evidence was finalised"
                                )

                        self.assertEqual(code, expected_code)
                        status = json.loads((output / "status.json").read_text())
                        self.assertEqual(status["status"], expected_status)
                        self.assertEqual(status["frames_received"], 1)
                        self.assertTrue(
                            (output / "latest.jpg").read_bytes().startswith(b"\xff\xd8")
                        )
                        source.close.assert_called_once()
                        self.assertIs(
                            signal.getsignal(signal.SIGINT), interrupt_handler
                        )

                    finally:
                        for server in servers:
                            server.server_close()

            # Raise SIGINT again and check that the restored handler produces KeyboardInterrupt.
            with self.assertRaises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGINT)

        finally:
            signal.signal(signal.SIGINT, prior_handler)

    def test_primary_receive_failure_survives_cleanup_failure_and_is_recorded(self):
        prior_handler = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGINT, prior_handler)
        source = Mock(spec=["receive", "close"])
        source.receive.side_effect = RuntimeError("receive failed")
        # Fail cleanup as well so the receive error must remain the primary recorded failure.
        source.close.side_effect = OSError("cleanup failed")

        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            with patch("stream_video.NetGearVideoReceiver", return_value=source):
                with self.assertRaisesRegex(RuntimeError, "receive failed"):
                    main(
                        [
                            "--http-port",
                            str(available_port()),
                            "--output-dir",
                            str(output),
                        ]
                    )

            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error"], "RuntimeError: receive failed")
            self.assertEqual(status["stream_status"], "closed")
            self.assertIn("OSError: cleanup failed", status["cleanup_errors"])

        source.close.assert_called_once()
        self.assertIs(signal.getsignal(signal.SIGINT), prior_handler)
