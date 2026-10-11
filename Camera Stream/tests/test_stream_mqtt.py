"""Exercise the MQTT wire boundary, including broker failure and CLI readiness."""

from contextlib import contextmanager
import importlib
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
RECORD = {
    "run_id": "run-test",
    "sequence_number": 17,
    "device_timestamp_s": 1.2,
    "host_received_timestamp_s": 1.3,
    "processors": [
        {"processor": "aruco", "data": {"ids": [7]}, "failure_reason": None}
    ],
}


def read_exact(connection, count):
    # Read exactly the requested byte count; raise EOFError if the peer closes early.
    result = b""

    while len(result) < count:
        part = connection.recv(count - len(result))

        if not part:
            raise EOFError("MQTT peer closed")

        result += part

    return result


def read_packet(connection):
    # Read one complete MQTT packet and return its header byte and body.
    header = read_exact(connection, 1)[0]
    length, multiplier = 0, 1

    # Combine the length bytes until the high bit says no further length byte follows.
    while True:
        value = read_exact(connection, 1)[0]
        length += (value & 127) * multiplier

        if not value & 128:
            return header, read_exact(connection, length)

        multiplier *= 128


def packet(header, body):
    # Prefix the supplied body with its MQTT header and encoded byte length.
    length, encoded = len(body), bytearray()

    # Write seven length bits per byte and set the high bit when another byte follows.
    while True:
        value = length % 128
        length //= 128
        encoded.append(value | (128 if length else 0))

        if not length:
            return bytes([header]) + bytes(encoded) + body


@contextmanager
def broker_peer(handler, host="127.0.0.1"):
    """Run a tiny local broker for a test while keeping the real MQTT client."""
    listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    listener.bind((host, 0))
    listener.listen(1)
    listener.settimeout(3)
    errors, connections = [], []

    def serve():
        try:
            connection, _ = listener.accept()
            connections.append(connection)

            with connection:
                connection.settimeout(3)
                handler(connection)

        # Allow the test client to close the connection without reporting a broker failure.
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass

        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    caller_failed = False

    try:
        yield listener.getsockname()[1]

    except BaseException:
        caller_failed = True
        raise

    finally:
        # Close the listener and client sockets, then wait for the broker thread to finish.
        listener.close()

        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)

            except OSError:
                pass

        thread.join(4)

        # Report a broker error only when it would not replace the test's own failure.
        if errors and not caller_failed:
            raise errors[0]

        if thread.is_alive():
            raise AssertionError("Scripted broker did not stop")


def accept_client(connection):
    header, _ = read_packet(connection)

    if header != 0x10:
        raise AssertionError("Client did not start with MQTT CONNECT")

    # Send a successful CONNACK after checking that the client opened with CONNECT.
    connection.sendall(b"\x20\x02\x00\x00")


class MqttPublisherTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("stream_mqtt"),
            "MQTT publishing implementation is missing",
        )
        return importlib.import_module("stream_mqtt")

    def test_sends_strict_json_with_qos_one_without_retaining(self):
        observed = []

        def handler(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            topic_length = int.from_bytes(body[:2], "big")
            topic = body[2 : 2 + topic_length].decode()
            message_id = body[2 + topic_length : 4 + topic_length]
            # Skip the topic and message ID to decode the JSON, then acknowledge that ID.
            observed.append((header, topic, json.loads(body[4 + topic_length :])))
            connection.sendall(b"\x40\x02" + message_id)
            read_packet(connection)

        with broker_peer(handler) as port:
            publisher = self.module().MqttResultPublisher(port=port)

            try:
                publisher.publish(RECORD)

            finally:
                publisher.close()
                publisher.close()

        self.assertEqual(observed, [(0x32, "taip/local/results", RECORD)])

    def test_missing_puback_is_a_bounded_failure(self):
        def handler(connection):
            accept_client(connection)
            # Read the publish and later disconnect without acknowledging the message.
            read_packet(connection)
            read_packet(connection)

        with broker_peer(handler) as port:
            publisher = self.module().MqttResultPublisher(
                port=port, publish_timeout_s=0.1
            )
            started = time.monotonic()

            try:
                with self.assertRaisesRegex(TimeoutError, "acknowledg"):
                    publisher.publish(RECORD)

                self.assertLess(time.monotonic() - started, 1)

            finally:
                publisher.close()

    def test_disconnect_cannot_claim_delivery(self):
        def handler(connection):
            accept_client(connection)
            # Close the broker connection after reading the publish, before sending PUBACK.
            read_packet(connection)

        with broker_peer(handler) as port:
            publisher = self.module().MqttResultPublisher(
                port=port, publish_timeout_s=0.1
            )

            try:
                with self.assertRaises((ConnectionError, TimeoutError)):
                    publisher.publish(RECORD)

                with self.assertRaises(ConnectionError):
                    publisher.publish(RECORD)

            finally:
                publisher.close()

    def test_failed_handshake_closes_socket_and_background_loop(self):
        closed = threading.Event()

        def handler(connection):
            # Read CONNECT but withhold CONNACK, then detect when the timed-out client closes.
            read_packet(connection)

            try:
                read_packet(connection)
                read_packet(connection)

            except EOFError:
                closed.set()

        before = {thread.ident for thread in threading.enumerate()}

        with broker_peer(handler) as port:
            with self.assertRaisesRegex(TimeoutError, "connect"):
                self.module().MqttResultPublisher(port=port, connect_timeout_s=0.1)

            self.assertTrue(closed.wait(1))
            added = [
                thread
                for thread in threading.enumerate()
                if thread.ident not in before and thread.name.startswith("paho-mqtt")
            ]
            self.assertEqual(added, [])

    def test_rejects_remote_hosts_and_invalid_bounds_before_connecting(self):
        cls = self.module().MqttResultPublisher

        for kwargs in (
            {"host": "example.com"},
            {"host": "127.0.0.2"},
            {"host": "::1"},
            {"port": 0},
            {"port": 65536},
            {"connect_timeout_s": 0},
            {"publish_timeout_s": float("inf")},
            {"topic": "taip/#"},
            {"topic": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cls(**kwargs)

    def test_explicit_remote_mode_uses_configured_host_on_the_wire(self):
        publisher_class = self.module().MqttResultPublisher
        self.assertIn(
            "allow_remote",
            inspect.signature(publisher_class).parameters,
            "Remote MQTT requires an explicit opt-in parameter",
        )
        # Try 127.0.0.2; if it cannot be bound, use the IPv6 loopback address instead.
        host = "127.0.0.2"

        with socket.socket() as probe:
            try:
                probe.bind((host, 0))

            except OSError:
                host = "::1"

        observed = []

        def handler(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            topic_length = int.from_bytes(body[:2], "big")
            topic = body[2 : 2 + topic_length].decode()
            message_id = body[2 + topic_length : 4 + topic_length]
            # Skip the topic and message ID to decode the JSON, then acknowledge that ID.
            observed.append((header, topic, json.loads(body[4 + topic_length :])))
            connection.sendall(b"\x40\x02" + message_id)
            read_packet(connection)

        with broker_peer(handler, host=host) as port:
            publisher = publisher_class(
                host=host,
                port=port,
                topic="gcs/drone_log",
                connect_timeout_s=0.5,
                allow_remote=True,
            )

            try:
                publisher.publish({"action": "test", "content": "local fixture only"})

            finally:
                publisher.close()

        self.assertEqual(
            observed,
            [
                (
                    0x32,
                    "gcs/drone_log",
                    {"action": "test", "content": "local fixture only"},
                )
            ],
        )

    def test_remote_mode_rejects_uris_ports_paths_and_invalid_hosts_before_network(
        self,
    ):
        module = self.module()
        self.assertIn(
            "allow_remote", inspect.signature(module.MqttResultPublisher).parameters
        )
        invalid_hosts = (
            None,
            "",
            " ",
            "0.0.0.0",
            "::",
            "999.0.0.1",
            " broker.example",
            "broker.example ",
            "mqtt://broker.example",
            "broker.example:1883",
            "127.0.0.1:1883",
            "[::1]:1883",
            "[::1]",
            "broker.example/path",
            "broker\\path",
            "broker?x=1",
            "broker#topic",
            "*.example",
            "broker+name",
            "broker\nname",
            "broker\0name",
            "-broker.example",
            "broker-.example",
            "broker..example",
            "a" * 64 + ".example",
        )

        with patch.object(
            module,
            "load_mqtt",
            side_effect=AssertionError("Validation reached network dependency"),
        ):
            for host in invalid_hosts:
                with self.subTest(host=host), self.assertRaises(ValueError):
                    module.MqttResultPublisher(host=host, allow_remote=True)

            for kwargs in (
                {"port": 0},
                {"port": 65536},
                {"port": True},
                {"topic": ""},
                {"topic": "gcs/#"},
                {"topic": "gcs/+"},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    module.MqttResultPublisher(
                        host="broker.example", allow_remote=True, **kwargs
                    )

    def test_remote_mode_accepts_plain_dns_names_and_ip_literals_without_probing_them(
        self,
    ):
        module = self.module()
        self.assertIn(
            "allow_remote", inspect.signature(module.MqttResultPublisher).parameters
        )

        class NetworkSetupReached(Exception):
            pass

        with patch.object(module, "load_mqtt", side_effect=NetworkSetupReached):
            for host in (
                "broker.example",
                "gcs-host",
                "broker.example.",
                "192.0.2.5",
                "2001:db8::1",
            ):
                with self.subTest(host=host), self.assertRaises(NetworkSetupReached):
                    module.MqttResultPublisher(host=host, allow_remote=True)

    def test_localhost_spelling_still_reaches_ipv4_loopback_with_remote_opt_in(self):
        self.assertIn(
            "allow_remote",
            inspect.signature(self.module().MqttResultPublisher).parameters,
        )

        def handler(connection):
            accept_client(connection)
            self.assertEqual(read_packet(connection)[0], 0xE0)

        with broker_peer(handler) as port:
            publisher = self.module().MqttResultPublisher(
                host="localhost", port=port, allow_remote=True
            )
            publisher.close()

    def test_rejects_nonstandard_json_and_binary_before_publishing(self):
        def handler(connection):
            accept_client(connection)
            header, _ = read_packet(connection)
            self.assertEqual(header, 0xE0)

        with broker_peer(handler) as port:
            publisher = self.module().MqttResultPublisher(port=port)

            try:
                for record in ({"measurement": float("nan")}, {"image": b"jpeg"}):
                    with self.subTest(record=record), self.assertRaises(
                        (TypeError, ValueError)
                    ):
                        publisher.publish(record)

            finally:
                publisher.close()


class MqttSubscriberTests(unittest.TestCase):
    def test_cli_readiness_follows_suback_and_writes_received_jsonl(self):
        subscribed, permit_suback = threading.Event(), threading.Event()

        def handler(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            self.assertEqual(header, 0x82)
            self.assertEqual(body[4:-1], b"taip/local/results")
            self.assertEqual(body[-1], 1)
            # Hold back SUBACK so the test can check that readiness has not been reported.
            subscribed.set()
            self.assertTrue(permit_suback.wait(2))
            connection.sendall(b"\x90\x03" + body[:2] + b"\x01")
            topic = b"taip/local/results"
            connection.sendall(
                packet(
                    0x32,
                    len(topic).to_bytes(2, "big")
                    + topic
                    + b"\x00\x07"
                    + json.dumps(RECORD).encode(),
                )
            )

            while True:
                if read_packet(connection)[0] == 0xE0:
                    return

        with TemporaryDirectory() as directory, broker_peer(handler) as port:
            ready, output = (
                Path(directory) / "ready",
                Path(directory) / "received.jsonl",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "run_mqtt_subscriber.py"),
                    "--port",
                    str(port),
                    "--ready-file",
                    str(ready),
                    "--output",
                    str(output),
                    "--max-messages",
                    "1",
                    "--timeout-s",
                    "3",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                self.assertTrue(subscribed.wait(2), "Subscriber did not send SUBSCRIBE")
                self.assertFalse(
                    ready.exists(), "Readiness must wait for broker SUBACK"
                )
                permit_suback.set()
                stdout, stderr = process.communicate(timeout=4)

            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

            self.assertEqual(process.returncode, 0, stderr)
            self.assertTrue(ready.exists())
            self.assertEqual(
                [json.loads(line) for line in output.read_text().splitlines()], [RECORD]
            )
            self.assertIn("run-test", stdout)
            self.assertIn("17", stdout)
            self.assertIn("aruco", stdout)

    def test_cli_does_not_overwrite_existing_evidence(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "received.jsonl"
            output.write_text("existing evidence\n")
            process = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_mqtt_subscriber.py"),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=4,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(output.read_text(), "existing evidence\n")
            self.assertIn("exist", process.stderr.lower())

    def test_cli_rejects_nonfinite_timeout(self):
        process = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_mqtt_subscriber.py"),
                "--timeout-s",
                "nan",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("finite", process.stderr)


if __name__ == "__main__":
    unittest.main()
