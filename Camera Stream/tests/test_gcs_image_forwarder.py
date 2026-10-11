"""Original-image delivery, bounded pacing and non-consuming preview behavior."""

import base64
import hashlib
from contextlib import redirect_stderr
from io import StringIO
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from test_stream_mqtt import accept_client, broker_peer, read_packet


class Clock:
    """Let a test advance time without waiting for the upload interval."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RecordingPublisher:
    """Remember what would be sent, without connecting to the GCS."""

    def __init__(self, clock=None, duration=0.0):
        self.payloads = []
        self.clock, self.duration = clock, duration
        self.closed = False

    def publish(self, payload):
        self.payloads.append(payload)

        # Advance the test clock by the time this publish is meant to take.
        if self.clock:
            self.clock.now += self.duration

    def close(self):
        self.closed = True


def capture(directory, sequence=1):
    # Write a PNG and a matching frame record, then return their capture metadata and bytes.
    image = directory / f"frame-{sequence}.png"
    pixels = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[90, 80, 70], [60, 50, 40], [30, 20, 10]],
        ],
        dtype=np.uint8,
    )
    success, encoded = cv2.imencode(".png", pixels)
    assert success
    original = encoded.tobytes()
    image.write_bytes(original)
    record = dict(
        producer="camera_stream",
        run_id="run-1",
        sequence_number=sequence,
        source_kind="camera",
        device_timestamp_s=1.2,
        host_received_timestamp_s=1.3,
        image_size_px=[3, 2],
        processors=[
            dict(
                processor="aruco",
                failure_reason=None,
                data=dict(
                    results=[
                        dict(
                            marker_id=7,
                            valid=False,
                            pose=None,
                            failure_reason="MISSING_CALIBRATION",
                            corners_px=[[0, 0], [2, 0], [2, 1], [0, 1]],
                        )
                    ]
                ),
            )
        ],
    )
    metadata = dict(
        schema_version=1,
        event_id=f"run-1:{sequence}",
        run_id="run-1",
        sequence_number=sequence,
        observed_at_utc="2026-10-08T01:02:03Z",
        timestamp_basis="host_selection_time_not_camera_exposure",
        marker_ids=[7],
        dictionary_name="DICT_5X5_100",
        source_kind="camera",
        image_file=image.name,
        image_bytes=len(original),
        image_sha256=hashlib.sha256(original).hexdigest(),
        record=record,
    )
    sidecar = directory / f"frame-{sequence}.event.json"
    # Write the companion JSON last so the uploader can find the completed capture.
    sidecar.write_text(json.dumps(metadata))
    return sidecar, metadata, original


class ImageForwarderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("run_gcs_image_forwarder"),
            "Automatic image forwarder has not been implemented",
        )
        self.module = importlib.import_module("run_gcs_image_forwarder")
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def forwarder(self, **kwargs):
        result = self.module.ImageForwarder(self.directory, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_preview_keeps_original_bytes_same_frame_metadata_and_pending_capture(self):
        sidecar, metadata, original = capture(self.directory)

        def forbidden(**kwargs):
            self.fail("Preview opened a publisher")

        uploader = self.forwarder(publisher_factory=forbidden)
        original_sidecar = sidecar.read_bytes()

        entry = uploader.step()
        payload = entry["payload"]

        # Check the outgoing image bytes, capture time and caption against the saved capture.
        self.assertEqual(
            set(payload), {"type", "description", "mime", "image", "timestamp"}
        )
        self.assertEqual(base64.b64decode(payload["image"]), original)
        self.assertEqual(payload["timestamp"], "2026-10-08T01:02:03Z")
        self.assertEqual((payload["type"], payload["mime"]), ("ArUco", "image/png"))
        caption = payload["description"]
        self.assertIn("Marker 7: pose unavailable (calibration missing)", caption)
        self.assertIn("Frame 1 | Device time 1.200 s | Run run-1", caption)
        self.assertNotIn("forward", caption)

        with self.assertRaises(json.JSONDecodeError):
            json.loads(caption)

        self.assertEqual(sidecar.read_bytes(), original_sidecar)
        self.assertEqual(json.loads(sidecar.read_text()), metadata)
        self.assertTrue(sidecar.with_suffix(".preview.json").exists())
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        self.assertFalse(sidecar.with_suffix(".failed.json").exists())
        self.assertEqual(
            (self.directory / metadata["image_file"]).read_bytes(), original
        )
        self.assertIsNone(uploader.step())

    def test_minimum_interval_includes_time_spent_publishing(self):
        capture(self.directory, 1)
        capture(self.directory, 2)
        clock = Clock()
        publisher = RecordingPublisher(clock, duration=3.0)
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            clock=clock,
            publisher_factory=lambda **kwargs: publisher,
        )
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")
        self.assertEqual(clock.now, 3.0)
        # The next publish must wait until ten seconds after the first one started.
        clock.now = 9.999
        self.assertIsNone(uploader.step())
        clock.now = 10.0
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")
        self.assertEqual(len(publisher.payloads), 2)

    def test_slow_initial_connection_cannot_shorten_actual_publish_spacing(self):
        capture(self.directory, 1)
        capture(self.directory, 2)
        clock = Clock()
        publisher = RecordingPublisher(clock)

        def connect(**kwargs):
            # Delay connection by nine seconds before the first publish can start.
            clock.now += 9.0
            return publisher

        uploader = self.forwarder(
            publish=True, broker="127.0.0.1", clock=clock, publisher_factory=connect
        )
        uploader.step()
        self.assertEqual(clock.now, 9.0)
        clock.now = 18.999
        self.assertIsNone(uploader.step())
        clock.now = 19.0
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")

    def test_bandwidth_budget_uses_actual_encoded_json_bytes(self):
        capture(self.directory, 1)
        capture(self.directory, 2)
        clock = Clock()
        publisher = RecordingPublisher(clock, duration=2.0)
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            clock=clock,
            max_mbps=0.0001,
            publisher_factory=lambda **kwargs: publisher,
        )
        first = uploader.step()
        encoded = json.dumps(
            publisher.payloads[0], allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(first["payload_wire_bytes"], len(encoded))
        # Allow 0.08 seconds per encoded JSON byte at the configured 100 bits per second.
        next_start = len(encoded) * 0.08
        self.assertGreater(next_start, 10.0)
        clock.now = next_start - 0.001
        self.assertIsNone(uploader.step())
        clock.now = next_start
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")

    def test_slow_publish_does_not_add_a_second_full_interval(self):
        capture(self.directory, 1)
        capture(self.directory, 2)
        clock = Clock()
        publisher = RecordingPublisher(clock, duration=12.0)
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            clock=clock,
            publisher_factory=lambda **kwargs: publisher,
        )
        uploader.step()
        self.assertEqual(clock.now, 12.0)
        self.assertIsNotNone(uploader.step())

    def test_receipts_prevent_redelivery_and_preview_does_not_block_real_publish(self):
        sent, _, _ = capture(self.directory, 1)
        failed, _, _ = capture(self.directory, 2)
        fresh, _, _ = capture(self.directory, 3)
        # Retire two captures and mark the third as previewed but still eligible to send.
        sent.with_suffix(".sent.json").write_text("{}")
        failed.with_suffix(".failed.json").write_text("{}")
        fresh.with_suffix(".preview.json").write_text("{}")
        publisher = RecordingPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )
        result = uploader.step()
        self.assertEqual(result["event_id"], "run-1:3")
        self.assertEqual(len(publisher.payloads), 1)
        self.assertTrue(fresh.with_suffix(".sent.json").exists())
        self.assertIsNone(uploader.step())

    def test_invalid_capture_is_preserved_failed_once_and_never_reaches_publisher(self):
        cases = (
            "escape",
            "missing",
            "hash",
            "size",
            "identity",
            "symlink",
            "no_observations",
        )

        for case in cases:
            with self.subTest(case=case), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sidecar, metadata, original = capture(directory)

                # Break one capture requirement while keeping the other fixture data intact.
                if case == "escape":
                    metadata["image_file"] = "../outside.png"

                elif case == "missing":
                    (directory / metadata["image_file"]).unlink()

                elif case == "hash":
                    metadata["image_sha256"] = "0" * 64

                elif case == "size":
                    metadata["image_bytes"] += 1

                elif case == "identity":
                    metadata["record"]["sequence_number"] = 999

                elif case == "no_observations":
                    metadata["record"]["processors"][0]["data"]["results"] = []

                elif case == "symlink":
                    (directory / metadata["image_file"]).unlink()
                    (directory / metadata["image_file"]).symlink_to(
                        directory.parent / "outside.png"
                    )

                sidecar.write_text(json.dumps(metadata))
                connections = []

                def connect(**kwargs):
                    connections.append(kwargs)
                    return RecordingPublisher()

                uploader = self.module.ImageForwarder(
                    directory,
                    publish=True,
                    broker="127.0.0.1",
                    publisher_factory=connect,
                )

                try:
                    result = uploader.step()
                    self.assertEqual(result["delivery"], "failed")
                    self.assertEqual(connections, [])
                    self.assertTrue(sidecar.exists())
                    self.assertTrue(sidecar.with_suffix(".failed.json").exists())
                    self.assertFalse(sidecar.with_suffix(".sent.json").exists())
                    self.assertIsNone(uploader.step())

                finally:
                    uploader.close()

    def test_oversized_file_is_rejected_before_builder_reads_it(self):
        sidecar, metadata, _ = capture(self.directory)
        image = self.directory / metadata["image_file"]
        # Make the file exceed the limit while keeping its declared size and hash correct.
        image.write_bytes(b"x" * 10_000)
        metadata.update(
            image_bytes=10_000,
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        )
        sidecar.write_text(json.dumps(metadata))
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            max_payload_mb=0.001,
            publisher_factory=lambda **kwargs: self.fail(
                "Oversized image reached network"
            ),
        )

        with patch.object(
            self.module,
            "gcs_detection_payload",
            side_effect=AssertionError("Oversized file reached image builder"),
        ):
            result = uploader.step()

        self.assertEqual(result["delivery"], "failed")
        self.assertIn("exceed", result["error"].lower())
        self.assertEqual(image.stat().st_size, 10_000)

    def test_preview_failure_does_not_release_or_delete_capture(self):
        sidecar, metadata, _ = capture(self.directory)
        metadata["image_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(metadata))
        result = self.forwarder().step()
        self.assertEqual(result["delivery"], "preview_failed")
        self.assertTrue(sidecar.with_suffix(".preview.json").exists())
        self.assertFalse(sidecar.with_suffix(".failed.json").exists())
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())

    def test_publish_failure_writes_failed_not_sent_and_preserves_bytes(self):
        sidecar, metadata, original = capture(self.directory)

        class BrokenPublisher(RecordingPublisher):
            def publish(self, payload):
                raise TimeoutError("No broker acknowledgement")

        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: BrokenPublisher(),
        )
        result = uploader.step()
        self.assertEqual(result["delivery"], "failed")
        self.assertIn("acknowledgement", result["error"])
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        self.assertEqual(
            (self.directory / metadata["image_file"]).read_bytes(), original
        )

    def test_new_event_can_reconnect_after_failure_without_retrying_old_event(self):
        old, _, _ = capture(self.directory, 1)
        new, _, _ = capture(self.directory, 2)
        clock = Clock()

        class BrokenPublisher(RecordingPublisher):
            def publish(self, payload):
                raise ConnectionError("Broker disconnected")

        broken, healthy = BrokenPublisher(), RecordingPublisher()
        # Fail frame 1 with the first client, then use a new client to send frame 2.
        clients = iter([broken, healthy])
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            clock=clock,
            publisher_factory=lambda **kwargs: next(clients),
        )
        self.assertEqual(uploader.step()["delivery"], "failed")
        self.assertTrue(broken.closed)
        clock.now = 10.0
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")
        self.assertTrue(old.with_suffix(".failed.json").exists())
        self.assertFalse(old.with_suffix(".sent.json").exists())
        self.assertTrue(new.with_suffix(".sent.json").exists())
        self.assertEqual(len(healthy.payloads), 1)
        self.assertIn(
            "Frame 2 | Device time 1.200 s | Run run-1",
            healthy.payloads[0]["description"],
        )

    def test_interrupt_during_preparation_keeps_capture_eligible_after_restart(self):
        sidecar, metadata, original = capture(self.directory)
        publisher = RecordingPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )

        # Interrupt payload preparation before any call to the publisher can begin.
        with patch.object(
            self.module, "gcs_detection_payload", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            uploader.step()

        self.assertFalse(sidecar.with_suffix(".failed.json").exists())
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        self.assertFalse(sidecar.with_suffix(".preview.json").exists())
        self.assertEqual(publisher.payloads, [])
        self.assertEqual(
            (self.directory / metadata["image_file"]).read_bytes(), original
        )
        uploader.close()
        restarted = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )
        self.assertEqual(restarted.step()["delivery"], "broker_acknowledged")
        self.assertEqual(len(publisher.payloads), 1)

    def test_interrupt_during_connection_keeps_capture_eligible_after_restart(self):
        sidecar, _, _ = capture(self.directory)

        def interrupted_connection(**kwargs):
            raise KeyboardInterrupt()

        uploader = self.forwarder(
            publish=True, broker="127.0.0.1", publisher_factory=interrupted_connection
        )

        with self.assertRaises(KeyboardInterrupt):
            uploader.step()

        self.assertFalse(sidecar.with_suffix(".failed.json").exists())
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        uploader.close()
        publisher = RecordingPublisher()
        restarted = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )
        self.assertEqual(restarted.step()["delivery"], "broker_acknowledged")
        self.assertEqual(len(publisher.payloads), 1)

    def test_interrupt_during_publish_records_uncertain_failure_without_auto_retry(
        self,
    ):
        sidecar, metadata, original = capture(self.directory)

        class InterruptedPublisher(RecordingPublisher):
            def publish(self, payload):
                raise KeyboardInterrupt()

        publisher = InterruptedPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            publisher_factory=lambda **kwargs: publisher,
        )

        with self.assertRaises(KeyboardInterrupt):
            uploader.step()

        # An interrupted publish must leave a failure receipt because delivery is uncertain.
        receipt = json.loads(sidecar.with_suffix(".failed.json").read_text())
        self.assertEqual(receipt["delivery"], "failed")
        self.assertIn("uncertain", receipt["acknowledgement_scope"])
        self.assertFalse(sidecar.with_suffix(".sent.json").exists())
        self.assertEqual(
            (self.directory / metadata["image_file"]).read_bytes(), original
        )
        self.assertTrue(publisher.closed)

    def test_null_observation_time_cannot_be_replaced_with_publication_time(self):
        sidecar, metadata, _ = capture(self.directory)
        metadata["observed_at_utc"] = None
        sidecar.write_text(json.dumps(metadata))
        result = self.forwarder().step()
        self.assertEqual(result["delivery"], "preview_failed")
        self.assertIn("timestamp", result["error"].lower())

    def test_encoded_json_overhead_is_included_in_payload_limit(self):
        sidecar, metadata, original = capture(self.directory)
        preview = self.forwarder().step()
        # Permit the image file itself but fall one byte short of the full JSON message.
        limit = preview["payload_wire_bytes"] - 1
        self.assertLess(len(original), limit)
        publisher = RecordingPublisher()
        uploader = self.forwarder(
            publish=True,
            broker="127.0.0.1",
            max_payload_mb=limit / 1_000_000,
            publisher_factory=lambda **kwargs: publisher,
        )
        result = uploader.step()
        self.assertEqual(result["delivery"], "failed")
        self.assertEqual(publisher.payloads, [])
        self.assertIn("payload limit", result["error"])

    def test_actual_mqtt_wire_preserves_exact_image_and_marks_only_broker_acknowledgement(
        self,
    ):
        sidecar, metadata, original = capture(self.directory)
        results = metadata["record"]["processors"][0]["data"]["results"]
        results[0].update(
            valid=True,
            failure_reason=None,
            pose=dict(
                translation_m=[0.005, -0.002, 0.447], coordinate_frame="camera_optical"
            ),
        )
        # Keep a candidate pose on rejected marker 8 to check that its caption withholds it.
        results.append(
            dict(
                results[0],
                marker_id=8,
                valid=False,
                failure_reason="MARKER_ID_NOT_ALLOWED",
            )
        )
        metadata["marker_ids"] = [7, 8]
        metadata["source_kind"] = metadata["record"]["source_kind"] = "synthetic"
        sidecar.write_text(json.dumps(metadata))
        original_sidecar = sidecar.read_bytes()
        observed = []

        def handler(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            # Decode the received topic and JSON, then return PUBACK for the same message ID.
            length = int.from_bytes(body[:2], "big")
            observed.append(
                (header, body[2 : 2 + length].decode(), json.loads(body[4 + length :]))
            )
            connection.sendall(b"\x40\x02" + body[2 + length : 4 + length])
            read_packet(connection)

        with broker_peer(handler) as port:
            uploader = self.module.ImageForwarder(
                self.directory, publish=True, broker="127.0.0.1", port=port
            )

            try:
                result = uploader.step()

            finally:
                uploader.close()

        self.assertEqual(result["delivery"], "broker_acknowledged")
        self.assertEqual(observed[0][:2], (0x32, "gcs/detection"))
        self.assertEqual(base64.b64decode(observed[0][2]["image"]), original)
        caption = observed[0][2]["description"]
        self.assertTrue(caption.startswith("Synthetic demo.\n"))
        self.assertIn("Marker 7: 0.5 cm right, 0.2 cm up, 44.7 cm forward", caption)
        self.assertIn("Marker 8: pose unavailable (marker ID not permitted)", caption)
        self.assertEqual(caption.count("forward"), 1)
        self.assertIn("Frame 1 | Device time 1.200 s | Run run-1", caption)
        self.assertEqual(sidecar.read_bytes(), original_sidecar)
        receipt = json.loads(sidecar.with_suffix(".sent.json").read_text())
        self.assertEqual(receipt["delivery"], "broker_acknowledged")
        self.assertIn("database", receipt["acknowledgement_scope"])

    def test_cli_validates_options_before_any_connection(self):
        common = ["--capture-dir", str(self.directory)]

        for extra in (
            ["--publish"],
            ["--interval-s", "2.9"],
            ["--max-mbps", "0"],
            ["--timeout-s", "nan"],
            ["--max-payload-mb", "0"],
        ):
            with self.subTest(extra=extra), redirect_stderr(
                StringIO()
            ), self.assertRaises(SystemExit) as raised:
                self.module.main(common + extra)

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
