"""Check receiver launch options without opening network sockets."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import http.client
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep
import unittest
from unittest.mock import patch

import run_receiver


class ReceiverCommandTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.output = StringIO()
        self.errors = StringIO()
        stack.enter_context(redirect_stdout(self.output))
        stack.enter_context(redirect_stderr(self.errors))

        # Replace only socket and thread operations; run the actual command setup.
        self.http_factory = stack.enter_context(
            patch.object(run_receiver, "ThreadingHTTPServer")
        )
        self.server = self.http_factory.return_value
        self.server.__enter__.return_value = self.server
        self.thread_factory = stack.enter_context(
            patch.object(run_receiver, "Thread")
        )
        self.netgear_factory = stack.enter_context(
            patch.object(run_receiver, "create_receiver")
        )
        stack.enter_context(
            patch.object(run_receiver, "receive_frame", return_value = None)
        )
        self.close_netgear = stack.enter_context(
            patch.object(run_receiver, "close_receiver")
        )

    def run_command(self, *arguments):
        with patch("sys.argv", ["run_receiver.py", *arguments]):
            run_receiver.main()

    def test_default_remains_local_and_closes_resources(self):
        self.run_command()
        self.assertEqual(
            self.http_factory.call_args.args[0],
            ("127.0.0.1", 8080),
        )
        self.assertIn("http://127.0.0.1:8080/stream.mjpeg", self.output.getvalue())
        self.netgear_factory.assert_called_once_with(host="127.0.0.1", port=5555)
        self.server.shutdown.assert_called_once()
        self.thread_factory.return_value.join.assert_called_once()
        self.close_netgear.assert_called_once_with(self.netgear_factory.return_value)

    def test_custom_http_address_does_not_change_the_netgear_connection(self):
        self.run_command("--http-host", "192.0.2.10", "--http-port", "8091")
        self.assertEqual(
            self.http_factory.call_args.args[0],
            ("192.0.2.10", 8091),
        )
        self.assertIn("http://192.0.2.10:8091/stream.mjpeg", self.output.getvalue())
        self.netgear_factory.assert_called_once_with(host="127.0.0.1", port=5555)

    def test_wildcard_bind_explains_the_browser_needs_a_lan_address(self):
        self.run_command("--http-host", "0.0.0.0")
        self.assertEqual(
            self.http_factory.call_args.args[0],
            ("0.0.0.0", 8080),
        )
        self.assertNotIn("http://0.0.0.0", self.output.getvalue())
        self.assertIn("LAN IP", self.output.getvalue())

    def test_invalid_ports_are_rejected_before_opening_resources(self):
        for port in ("0", "-1", "65536", "abc"):
            with self.subTest(port = port):
                with self.assertRaises(SystemExit) as stopped:
                    self.run_command("--http-port", port)

                self.assertEqual(stopped.exception.code, 2)
                self.http_factory.assert_not_called()
                self.netgear_factory.assert_not_called()

    def test_bind_failure_reports_the_selected_address(self):
        self.http_factory.side_effect = OSError("Address already in use")

        try:
            with self.assertRaises(SystemExit) as stopped:
                self.run_command("--http-host", "192.0.2.10", "--http-port", "8091")

        except OSError as error:
            self.fail(f"Socket error escaped without a command error message: {error}")

        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("192.0.2.10:8091", self.errors.getvalue())
        self.assertIn("Address already in use", self.errors.getvalue())
        self.netgear_factory.assert_not_called()


    def test_custom_video_listener_uses_selected_port_and_host(self):
        self.run_command("--video-host", "0.0.0.0", "--video-port", "15555")
        self.netgear_factory.assert_called_once_with(host="0.0.0.0", port=15555)

    def test_readiness_is_written_only_after_both_listeners_start(self):
        with tempfile.TemporaryDirectory() as temp:
            ready_file = Path(temp) / "ready.json"
            def bound_receiver(**kwargs):
                self.assertFalse(ready_file.exists())
                self.thread_factory.return_value.start.assert_called_once()
                return object()
            self.netgear_factory.side_effect = bound_receiver
            def receive(receiver):
                state = json.loads(ready_file.read_text())
                self.assertEqual(state["http_port"], 8080)
                self.assertEqual(state["video_port"], 5555)
                self.assertEqual(state["video_host"], "127.0.0.1")
                return None
            with patch.object(run_receiver, "receive_frame", side_effect=receive):
                self.run_command("--ready-file", str(ready_file))
            self.assertFalse(ready_file.exists())

    def test_netgear_bind_failure_never_announces_ready_and_closes_http(self):
        self.netgear_factory.side_effect = RuntimeError("Address already in use")
        with tempfile.TemporaryDirectory() as temp:
            ready_file = Path(temp) / "ready.json"
            with patch("sys.argv", ["run_receiver.py", "--ready-file", str(ready_file)]):
                self.assertNotEqual(run_receiver.main(), 0)
            self.assertFalse(ready_file.exists())
        self.server.shutdown.assert_called_once()
        self.thread_factory.return_value.join.assert_called_once()
        self.close_netgear.assert_not_called()
        self.assertIn("Address already in use", self.errors.getvalue())

    def test_invalid_video_port_is_rejected_before_opening_resources(self):
        with self.assertRaises(SystemExit) as stopped:
            self.run_command("--video-port", "0")
        self.assertEqual(stopped.exception.code, 2)
        self.http_factory.assert_not_called()
        self.netgear_factory.assert_not_called()



class ReceiverSocketTests(unittest.TestCase):
    """Readiness must represent actual listeners, including failure cleanup."""

    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.ready_file = Path(self.temp) / "receiver.ready"

    def free_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def launch(self, http_port, video_port):
        child = subprocess.Popen([
            sys.executable, "-B", "-u", str(Path(run_receiver.__file__)),
            "--http-port", str(http_port), "--video-port", str(video_port),
            "--ready-file", str(self.ready_file),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def cleanup():
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        self.addCleanup(cleanup)
        return child

    @unittest.skipIf(sys.platform == "win32", "Windows uses CTRL_BREAK_EVENT for graceful shutdown")
    def test_real_readiness_http_service_and_sigterm_release_both_ports(self):
        http_port, video_port = self.free_port(), self.free_port()
        child = self.launch(http_port, video_port)
        deadline = monotonic() + 8
        while not self.ready_file.exists():
            if child.poll() is not None or monotonic() >= deadline:
                self.fail(f"Receiver did not become ready; exit={child.poll()}")
            sleep(0.03)
        state = json.loads(self.ready_file.read_text())
        self.assertEqual((state["http_port"], state["video_port"]), (http_port, video_port))
        connection = http.client.HTTPConnection("127.0.0.1", http_port, timeout=2)
        try:
            connection.request("GET", "/missing")
            self.assertEqual(connection.getresponse().status, 404)
        finally:
            connection.close()
        child.send_signal(signal.SIGTERM)
        _, errors = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 0, errors)
        self.assertFalse(self.ready_file.exists())
        for port in (http_port, video_port):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))

    def test_real_busy_ports_never_report_ready(self):
        for busy_listener in ("http", "video"):
            with self.subTest(listener=busy_listener), socket.socket() as occupied:
                occupied.bind(("127.0.0.1", 0))
                occupied.listen()
                busy_port = occupied.getsockname()[1]
                free_port = self.free_port()
                child = self.launch(busy_port if busy_listener == "http" else free_port,
                                    busy_port if busy_listener == "video" else free_port)
                _, errors = child.communicate(timeout=8)
                self.assertNotEqual(child.returncode, 0)
                self.assertFalse(self.ready_file.exists())
                self.assertIn("Receiver error", errors)
                # The original listener remains owned by this test.
                self.assertEqual(occupied.getsockname()[1], busy_port)


if __name__ == "__main__":
    unittest.main()
