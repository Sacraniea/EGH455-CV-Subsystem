"""Contracts at the NetGear boundary, without requiring sockets or camera hardware."""

import importlib.util
import json
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from stream_types import FramePacket


class RecordingNetGear:
    """Model NetGear's public methods and multi-server receive tuple."""

    def __init__(self, **options):
        self.options = options
        self.sent = []
        self.incoming = []
        self.close_calls = []

    def send(self, image, message=None):
        # Save independent copies of the sent pixels and JSON metadata for later assertions.
        self.sent.append((image.copy(), json.loads(json.dumps(message))))

    def recv(self):
        return self.incoming.pop(0)

    def close(self, **options):
        self.close_calls.append(options)


class StreamVideoTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("stream_video"),
            "The local video transport module must exist",
        )
        import stream_video

        self.module = stream_video

    def factory(self, **options):
        self.boundary = RecordingNetGear(**options)
        return self.boundary

    def test_send_keeps_capture_identity_attached_to_output_pixels(self):
        image = np.full((4, 6, 3), 80, dtype=np.uint8)
        packet = FramePacket(image, 27, 1.25, 1.28)

        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            sender = self.module.NetGearVideoSender()

        sender.send(packet, run_id="run-abc")

        pixels, metadata = self.boundary.sent[0]
        np.testing.assert_array_equal(pixels, image)
        self.assertEqual(
            metadata,
            {
                "schema_version": 1,
                "producer": "camera_stream",
                "source": "depthai",
                "source_kind": "camera",
                "run_id": "run-abc",
                "sequence_number": 27,
                "device_timestamp_s": 1.25,
                "host_received_timestamp_s": 1.28,
                "image_size_px": [6, 4],
            },
        )
        self.assertEqual(self.boundary.options["address"], "127.0.0.1")
        self.assertEqual(self.boundary.options["pattern"], 2)
        self.assertTrue(self.boundary.options["multiserver_mode"])
        self.assertEqual(self.boundary.options["jpeg_compression"], "BGR")
        self.assertEqual(self.boundary.options["jpeg_compression_quality"], 90)
        self.assertFalse(self.boundary.options["receive_mode"])

    def test_synthetic_frames_are_identified_as_synthetic_evidence(self):
        packet = FramePacket(np.zeros((4, 6, 3), np.uint8), 1, 1.0, 1.01)

        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            sender = self.module.NetGearVideoSender()

        sender.send(packet, run_id="synthetic-run", source_kind="synthetic")
        metadata = self.boundary.sent[0][1]
        self.assertEqual(metadata["source_kind"], "synthetic")
        self.assertEqual(metadata["source"], "synthetic")

    def test_receive_normalizes_multiserver_tuple_and_preserves_metadata(self):
        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            receiver = self.module.NetGearVideoReceiver()

        metadata = {
            "schema_version": 1,
            "run_id": "run-abc",
            "sequence_number": 27,
            "device_timestamp_s": 1.25,
            "host_received_timestamp_s": 1.28,
            "image_size_px": [6, 4],
        }
        image = np.full((4, 6, 3), 80, dtype=np.uint8)
        self.boundary.incoming = [("5555", metadata, image), None]

        received_metadata, received_image = receiver.receive()

        self.assertEqual(received_metadata, metadata)
        np.testing.assert_array_equal(received_image, image)
        self.assertIsNone(receiver.receive())
        self.assertEqual(self.boundary.options["address"], "127.0.0.1")
        self.assertEqual(self.boundary.options["port"], ["5555"])
        self.assertEqual(self.boundary.options["pattern"], 2)
        self.assertTrue(self.boundary.options["multiserver_mode"])
        self.assertTrue(self.boundary.options["receive_mode"])
        self.assertGreater(self.boundary.options["subscriber_timeout"], 0)

    def test_nonlocal_addresses_and_invalid_ports_are_rejected_before_network_use(self):
        for adapter in (
            self.module.NetGearVideoSender,
            self.module.NetGearVideoReceiver,
        ):
            for host in ("0.0.0.0", "*", "192.168.1.10", "example.com", "::"):
                with self.subTest(adapter=adapter.__name__, host=host):
                    with self.assertRaises(ValueError):
                        adapter(host=host)

            for port in (0, -1, 65536, True, 5555.0, "5555"):
                with self.subTest(adapter=adapter.__name__, port=port):
                    with self.assertRaises(ValueError):
                        adapter(port=port)

    def test_localhost_is_normalized_to_an_explicit_loopback_address(self):
        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            self.module.NetGearVideoSender(host="localhost")

        self.assertEqual(self.boundary.options["address"], "127.0.0.1")

    def test_subscriber_timeout_is_finite_and_forwarded_in_seconds(self):
        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            receiver = self.module.NetGearVideoReceiver(subscriber_timeout_s=2)
            self.boundary.incoming = [None]
            self.assertIsNone(receiver.receive())
            self.assertEqual(self.boundary.options["subscriber_timeout"], 2)

            for timeout in (0, -1, True, 1.5, None):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    self.module.NetGearVideoReceiver(subscriber_timeout_s=timeout)

    def test_close_is_idempotent_and_operations_after_close_do_not_reuse_socket(self):
        packet = FramePacket(np.zeros((4, 6, 3), np.uint8), 1, 1.0, 1.01)

        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            sender = self.module.NetGearVideoSender()

        boundary = self.boundary
        sender.close()
        sender.close()
        self.assertEqual(boundary.close_calls, [{}])

        with self.assertRaises(RuntimeError):
            sender.send(packet, run_id="run-abc")

        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            receiver = self.module.NetGearVideoReceiver()

        receiver.close()
        receiver.close()
        self.assertEqual(self.boundary.close_calls, [{"kill": True}])
        self.assertIsNone(receiver.receive())

    def test_receive_rejects_frames_whose_dimensions_do_not_match_metadata(self):
        with patch.object(self.module, "_load_netgear", return_value=self.factory):
            receiver = self.module.NetGearVideoReceiver()

        self.boundary.incoming = [
            (
                "5555",
                {
                    "schema_version": 1,
                    "run_id": "run-abc",
                    "sequence_number": 27,
                    "device_timestamp_s": 1.25,
                    "host_received_timestamp_s": 1.28,
                    "image_size_px": [640, 480],
                },
                np.zeros((4, 6, 3), np.uint8),
            )
        ]

        with self.assertRaisesRegex(ValueError, "dimensions"):
            receiver.receive()

    def test_import_does_not_load_optional_transport_dependency(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import stream_video; "
                "assert 'vidgear' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class NetGearProcessTests(unittest.TestCase):
    """Characterize real NetGear sockets; receiver must own a separate process."""

    def unused_port(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            return reservation.getsockname()[1]

    def start_receiver(self, port, body):
        # Run the receiver in a child process so its ZeroMQ cleanup cannot affect other tests.
        script = (
            "import json, sys, time\n"
            "from stream_video import NetGearVideoReceiver\n"
            f"receiver = NetGearVideoReceiver(port={port}, subscriber_timeout_s=2)\n"
            "print('ready', flush=True)\n"
            "sys.stdin.readline()\n"
            "try:\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\nfinally:\n    receiver.close()\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self.clean_process, process)

        # Wait for the child's readiness line with a timeout before allowing its test body to run.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(timeout=10), "receiver readiness timed out")
            line = process.stdout.readline().strip()

        if line != "ready":
            _, errors = process.communicate(timeout=5)
            self.fail(f"receiver failed before readiness: {line}\n{errors}")

        # Release the child after reading readiness so communicate can collect all result output.
        process.stdin.write("start\n")
        process.stdin.flush()
        return process

    @staticmethod
    def clean_process(process):
        # Stop a child that is still running, then collect its output and release its pipes.
        if process.poll() is None:
            process.kill()

        process.communicate(timeout=5)

    def test_real_jpeg_roundtrip_preserves_identity_and_dimensions(self):
        from stream_video import NetGearVideoSender

        port = self.unused_port()
        process = self.start_receiver(
            port,
            """metadata, image = receiver.receive()
print(json.dumps({'metadata': metadata, 'shape': list(image.shape),
                  'mean': float(image.mean())}), flush=True)""",
        )
        sender = NetGearVideoSender(port=port)

        try:
            packet = FramePacket(np.full((24, 32, 3), 80, np.uint8), 27, 1.25, 1.28)

            # Send repeated copies while the receiver subscription connects, then collect one frame.
            for _ in range(20):
                sender.send(packet, run_id="roundtrip", source_kind="synthetic")
                time.sleep(0.03)

            output, errors = process.communicate(timeout=5)

        finally:
            sender.close()

        self.assertEqual(process.returncode, 0, errors)
        result = json.loads(output)
        self.assertEqual(result["metadata"]["run_id"], "roundtrip")
        self.assertEqual(result["metadata"]["sequence_number"], 27)
        self.assertEqual(result["metadata"]["device_timestamp_s"], 1.25)
        self.assertEqual(result["metadata"]["host_received_timestamp_s"], 1.28)
        self.assertEqual(result["metadata"]["source_kind"], "synthetic")
        self.assertEqual(result["metadata"]["image_size_px"], [32, 24])
        self.assertEqual(result["shape"], [24, 32, 3])
        self.assertAlmostEqual(result["mean"], 80.0, delta=3.0)

    def test_receiver_without_frames_returns_none_after_bounded_timeout(self):
        process = self.start_receiver(
            self.unused_port(),
            """started = time.monotonic()
received = receiver.receive()
print(json.dumps({'ended': received is None, 'elapsed': time.monotonic()-started}), flush=True)""",
        )
        output, errors = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, errors)
        result = json.loads(output)
        self.assertTrue(result["ended"])
        self.assertGreater(result["elapsed"], 1.0)
        self.assertLess(result["elapsed"], 4.0)

    def test_close_releases_receiver_while_background_read_is_waiting(self):
        process = self.start_receiver(
            self.unused_port(),
            """started = time.monotonic()
receiver.close()
receiver.close()
print(json.dumps({'elapsed': time.monotonic()-started}), flush=True)""",
        )
        output, errors = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, errors)
        self.assertLess(json.loads(output)["elapsed"], 2.0)

    def test_sender_with_no_receiver_does_not_wait_for_acknowledgements(self):
        from stream_video import NetGearVideoSender

        sender = NetGearVideoSender(port=self.unused_port())
        started = time.monotonic()

        try:
            packet = FramePacket(np.zeros((24, 32, 3), np.uint8), 27, 1.25, 1.28)

            for _ in range(30):
                sender.send(packet, run_id="no-receiver", source_kind="synthetic")

        finally:
            sender.close()

        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
