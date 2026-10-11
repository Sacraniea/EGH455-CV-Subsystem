"""Recover known-unsent images without replaying uncertain MQTT deliveries."""

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

import run_gcs_image_forwarder as image_forwarder
import stream_mqtt
from test_gcs_image_forwarder import Clock, RecordingPublisher, capture
from test_stream_mqtt import accept_client, broker_peer, read_packet


def disconnected_publisher():
    """Create an idle, disconnected publisher without opening a socket."""
    # Create the real publisher with a disconnected flag and a mock MQTT client.
    publisher = object.__new__(stream_mqtt.MqttResultPublisher)
    publisher._closed = False
    publisher._connected = Event()
    publisher._connection_error = "MQTT broker disconnected while idle"
    publisher._client = Mock()
    return publisher


class ImageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def forwarder(self, **kwargs):
        uploader = image_forwarder.ImageForwarder(
            self.directory, publish=True, broker="127.0.0.1", **kwargs
        )
        self.addCleanup(uploader.close)
        return uploader

    def assert_pending(self, sidecar, entry):
        # Check the returned status and retry file agree that no image publish started.
        self.assertEqual(entry["delivery"], "retry_pending")
        self.assertFalse(entry["publish_attempted"])
        self.assertFalse(sidecar.with_suffix(".failed.json").exists())
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        diagnostic = json.loads(sidecar.with_suffix(".retry.json").read_text())
        self.assertEqual(diagnostic["delivery"], "retry_pending")
        self.assertFalse(diagnostic["publish_attempted"])

    def test_disconnected_precheck_identifies_that_no_publish_started(self):
        error_type = getattr(stream_mqtt, "PublishNotStartedError", None)
        self.assertIsNotNone(
            error_type, "Disconnected precheck needs a distinct safe-to-retry error"
        )
        self.assertTrue(issubclass(error_type, ConnectionError))

        for closed in (False, True):
            with self.subTest(closed=closed):
                publisher = disconnected_publisher()
                publisher._closed = closed

                with self.assertRaises(error_type):
                    publisher.publish({"type": "ArUco"})

                publisher._client.publish.assert_not_called()

    def check_connection_recovery(self, error):
        sidecar, metadata, original = capture(self.directory)
        clock = Clock()
        publisher = RecordingPublisher()
        attempts = []

        def connect(**kwargs):
            attempts.append(kwargs)

            # The first connection fails; the next one uses the same saved photo.
            if len(attempts) == 1:
                raise error

            return publisher

        uploader = self.forwarder(clock=clock, publisher_factory=connect)
        self.assert_pending(sidecar, uploader.step())
        self.assertEqual(
            (self.directory / metadata["image_file"]).read_bytes(), original
        )
        clock.now = 9.999
        # Confirm no reconnect occurs just before the ten-second retry delay expires.
        self.assertIsNone(uploader.step())
        self.assertEqual(len(attempts), 1)
        clock.now = 10.0
        recovered = uploader.step()
        self.assertEqual(recovered["delivery"], "broker_acknowledged")
        self.assertEqual(recovered["event_id"], "run-1:1")
        self.assertEqual(len(publisher.payloads), 1)
        self.assertTrue(sidecar.with_suffix(".sent.json").exists())
        self.assertFalse(sidecar.with_suffix(".retry.json").exists())
        self.assertFalse(sidecar.with_suffix(".failed.json").exists())

    def test_connection_error_retries_the_same_capture_after_backoff(self):
        self.check_connection_recovery(
            ConnectionError("Broker temporarily unavailable")
        )

    def test_connection_timeout_does_not_retire_an_unsent_capture(self):
        self.check_connection_recovery(TimeoutError("No connection acknowledgement"))

    def test_connection_backoff_starts_when_a_slow_connection_attempt_fails(self):
        sidecar, _, _ = capture(self.directory)
        clock = Clock()
        publisher = RecordingPublisher()
        attempts = []

        def connect(**kwargs):
            attempts.append(clock.now)

            if len(attempts) == 1:
                # Make the failed connection consume seven seconds before retry timing begins.
                clock.now += 7.0
                raise TimeoutError("Connection attempt took seven seconds")

            return publisher

        uploader = self.forwarder(clock=clock, publisher_factory=connect)
        self.assert_pending(sidecar, uploader.step())
        self.assertEqual(uploader.queue_status()["next_attempt_in_s"], 10.0)
        clock.now = 16.999
        self.assertIsNone(uploader.step())
        self.assertEqual(attempts, [0.0])
        clock.now = 17.0
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")
        self.assertEqual(attempts, [0.0, 17.0])

    def test_idle_disconnect_recovers_the_same_capture_without_sacrificing_it(self):
        sidecar, _, _ = capture(self.directory)
        clock = Clock()
        healthy = RecordingPublisher()
        uploader = self.forwarder(
            clock=clock, publisher_factory=lambda **kwargs: healthy
        )
        stale = disconnected_publisher()
        # Replace the idle connection with a disconnected client before processing the capture.
        uploader._publisher = stale
        self.assert_pending(sidecar, uploader.step())
        stale._client.publish.assert_not_called()
        self.assertTrue(stale._closed)
        clock.now = 10.0
        recovered = uploader.step()
        self.assertEqual(recovered["event_id"], "run-1:1")
        self.assertEqual(recovered["delivery"], "broker_acknowledged")
        self.assertEqual(len(healthy.payloads), 1)
        self.assertFalse(sidecar.with_suffix(".retry.json").exists())

    def test_missing_publish_acknowledgement_stays_failed_across_restart(self):
        sidecar, _, _ = capture(self.directory)
        clock = Clock()

        class UncertainPublisher(RecordingPublisher):
            def publish(self, payload):
                super().publish(payload)
                # Raise after recording the payload to simulate a send with no broker acknowledgement.
                raise TimeoutError("MQTT publish acknowledgement timed out")

        publisher = UncertainPublisher()
        uploader = self.forwarder(
            clock=clock, publisher_factory=lambda **kwargs: publisher
        )
        entry = uploader.step()
        self.assertEqual(entry["delivery"], "failed")
        self.assertTrue(entry["publish_attempted"])
        self.assertIn("uncertain", entry["acknowledgement_scope"])
        self.assertTrue(sidecar.with_suffix(".failed.json").exists())
        self.assertFalse(sidecar.with_suffix(".retry.json").exists())
        clock.now = 10.0
        self.assertIsNone(uploader.step())
        uploader.close()
        restarted = self.forwarder(
            clock=clock, publisher_factory=lambda **kwargs: publisher
        )
        self.assertIsNone(restarted.step())
        self.assertEqual(len(publisher.payloads), 1)

    def test_publish_timeout_rejects_nonpositive_and_nonfinite_values(self):
        self.assertIn(
            "publish_timeout_s",
            inspect.signature(image_forwarder.ImageForwarder).parameters,
        )

        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.forwarder(publish_timeout_s=value)

    def test_default_image_timeout_accepts_puback_later_than_two_seconds(self):
        sidecar, _, _ = capture(self.directory)

        def handler(connection):
            accept_client(connection)
            _, body = read_packet(connection)
            topic_length = int.from_bytes(body[:2], "big")
            message_id = body[2 + topic_length : 4 + topic_length]
            # Wait 2.25 seconds before acknowledging the received image publish.
            time.sleep(2.25)
            connection.sendall(b"\x40\x02" + message_id)
            read_packet(connection)

        with broker_peer(handler) as port:
            uploader = self.forwarder(port=port)

            try:
                entry = uploader.step()

            finally:
                uploader.close()

        self.assertEqual(entry["delivery"], "broker_acknowledged")
        self.assertTrue(sidecar.with_suffix(".sent.json").exists())
        self.assertFalse(sidecar.with_suffix(".failed.json").exists())

    def test_configured_short_timeout_bounds_a_real_missing_puback(self):
        self.assertIn(
            "publish_timeout_s",
            inspect.signature(image_forwarder.ImageForwarder).parameters,
        )
        sidecar, _, _ = capture(self.directory)

        def handler(connection):
            accept_client(connection)
            # Read the publish and later disconnect without ever returning PUBACK.
            read_packet(connection)
            read_packet(connection)

        with broker_peer(handler) as port:
            uploader = self.forwarder(port=port, publish_timeout_s=0.05)
            started = time.monotonic()

            try:
                entry = uploader.step()

            finally:
                uploader.close()

        self.assertEqual(entry["delivery"], "failed")
        self.assertIn("acknowledgement timed out after 0.05s", entry["error"])
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(sidecar.with_suffix(".failed.json").exists())
        self.assertFalse(sidecar.with_suffix(".retry.json").exists())

    def test_queue_status_distinguishes_completion_retry_and_preview_eligibility(self):
        for sequence, receipt in (
            (1, ".sent.json"),
            (2, ".failed.json"),
            (3, ".retry.json"),
            (4, ".preview.json"),
            (5, None),
        ):
            sidecar, _, _ = capture(self.directory, sequence)

            if receipt is not None:
                sidecar.with_suffix(receipt).write_text("{}")

        def forbidden(**kwargs):
            self.fail("Reading queue status must not open a network connection")

        uploader = self.forwarder(publisher_factory=forbidden)
        self.assertEqual(
            uploader.queue_status(),
            dict(
                pending=3,
                eligible=3,
                sent=1,
                failed=1,
                retrying=1,
                next_attempt_in_s=0.0,
            ),
        )
        # Preview skips the already-previewed capture; publish mode still counts it as eligible.
        preview = image_forwarder.ImageForwarder(
            self.directory, publisher_factory=forbidden
        )
        self.addCleanup(preview.close)
        self.assertEqual(
            preview.queue_status(),
            dict(
                pending=3,
                eligible=2,
                sent=1,
                failed=1,
                retrying=1,
                next_attempt_in_s=0.0,
            ),
        )

    def run_cli(
        self,
        arguments,
        *,
        clock,
        publisher_factory=None,
        sleep_step=None,
        interrupt_at=1000.0
    ):
        """Run the real CLI/uploader with simulated time and no external network."""
        real_forwarder = image_forwarder.ImageForwarder
        uploaders, output = [], []

        def forbidden(**kwargs):
            self.fail("This CLI scenario must not open a network connection")

        def construct(*args, **kwargs):
            # Keep the real uploader logic while injecting the test clock and publisher.
            uploader = real_forwarder(
                *args,
                **kwargs,
                clock=clock,
                publisher_factory=publisher_factory or forbidden
            )
            uploaders.append(uploader)
            return uploader

        def sleep(seconds):
            # Advance simulated time and interrupt the command at the selected test deadline.
            clock.now += seconds if sleep_step is None else sleep_step

            if clock.now >= interrupt_at:
                raise KeyboardInterrupt()

        with patch.object(
            image_forwarder, "ImageForwarder", side_effect=construct
        ), patch.object(image_forwarder, "monotonic", clock), patch.object(
            image_forwarder, "sleep", side_effect=sleep
        ):
            result = image_forwarder.main(
                ["--capture-dir", str(self.directory), *arguments],
                printer=lambda message, **kwargs: output.append((clock.now, message)),
            )

        self.assertTrue(all(uploader._closed for uploader in uploaders))
        return result, output

    def test_cli_zero_runtime_waits_for_ctrl_c_after_ten_minutes(self):
        clock = Clock()
        result, output = self.run_cli(
            ["--timeout-s", "0"], clock=clock, sleep_step=301.0, interrupt_at=650.0
        )
        self.assertEqual(result, 130)
        self.assertGreater(clock.now, 600.0)
        messages = [message for _, message in output]
        self.assertTrue(any("until Ctrl+C" in message for message in messages))
        self.assertTrue(
            any(
                "Stopped by Ctrl+C" in message and "pending=0" in message
                for message in messages
            )
        )
        self.assertFalse(
            any("Runtime limit reached" in message for message in messages)
        )

    def test_cli_expiry_prints_final_queue_summary(self):
        capture(self.directory)
        result, output = self.run_cli(
            ["--timeout-s", ".2"], clock=Clock(), interrupt_at=1.0
        )
        self.assertEqual(result, 0)
        stopped = [
            message for _, message in output if "Runtime limit reached" in message
        ]
        self.assertEqual(len(stopped), 1)
        # The preview has finished, but its capture remains pending for a real publish.
        self.assertIn("pending=1", stopped[0])
        self.assertIn("eligible=0", stopped[0])
        self.assertIn("waiting for new captures", stopped[0])

    def test_cli_reports_selected_destination_and_passes_custom_ack_timeout(self):
        capture(self.directory)
        publisher = RecordingPublisher()
        connections = []

        def connect(**kwargs):
            connections.append(kwargs)
            return publisher

        result, output = self.run_cli(
            [
                "--publish",
                "--broker",
                "127.0.0.1",
                "--port",
                "18884",
                "--publish-timeout-s",
                "17",
                "--timeout-s",
                ".2",
            ],
            clock=Clock(),
            publisher_factory=connect,
            interrupt_at=1.0,
        )
        self.assertEqual(result, 0)
        messages = "\n".join(message for _, message in output)
        self.assertIn(str(self.directory.resolve()), messages)
        self.assertIn("127.0.0.1:18884", messages)
        self.assertIn("gcs/detection", messages)
        self.assertRegex(messages, r"acknowledgement[^\n]*17s")
        self.assertIn("Capture folder is fixed", messages)
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0]["host"], "127.0.0.1")
        self.assertEqual(connections[0]["port"], 18884)
        self.assertEqual(connections[0]["topic"], "gcs/detection")
        self.assertEqual(connections[0]["publish_timeout_s"], 17.0)
        self.assertEqual(len(publisher.payloads), 1)
        self.assertTrue(publisher.closed)

    def test_cli_heartbeat_reports_the_queue_while_waiting_for_bandwidth(self):
        capture(self.directory, 1)
        capture(self.directory, 2)
        publisher = RecordingPublisher()
        result, output = self.run_cli(
            [
                "--publish",
                "--broker",
                "127.0.0.1",
                "--timeout-s",
                "0",
                "--max-mbps",
                ".00001",
            ],
            clock=Clock(),
            publisher_factory=lambda **kwargs: publisher,
            sleep_step=30.0,
            interrupt_at=61.0,
        )
        self.assertEqual(result, 130)
        heartbeat = [
            (when, message) for when, message in output if message.startswith("Queue:")
        ]
        self.assertEqual([when for when, _ in heartbeat], [0.0, 30.0, 60.0])
        self.assertIn("pending=2", heartbeat[0][1])

        for _, message in heartbeat[1:]:
            self.assertIn("pending=1", message)
            self.assertIn("eligible=1", message)
            self.assertIn("broker-acknowledged=1", message)
            self.assertIn("failed=0", message)
            self.assertIn("retrying=0", message)
            self.assertRegex(message, r"next attempt in [1-9][0-9]*\.[0-9]s")

        self.assertEqual(len(publisher.payloads), 1)


if __name__ == "__main__":
    unittest.main()
