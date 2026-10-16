"""Exercise photo evidence, bounded selection and actual MQTT delivery."""

import base64
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stream_types import FramePacket


def frame(sequence, now, *, markers=(), targets=True, aruco_failed=False):
    image = np.full((36, 48, 3), 25, dtype=np.uint8)
    original = FramePacket(image, sequence, now, now)
    overlay = image.copy()
    overlay[8:20, 10:30] = [20, 210, 30]
    annotated = replace(original, image=overlay)
    processors = [{
        "processor": "aruco",
        "failure_reason": "DETECTOR_ERROR" if aruco_failed else None,
        "data": {
            "config": {"dictionary_name": "DICT_5X5_100"},
            "results": [{
                "marker_id": marker,
                "valid": False,
                "failure_reason": "MISSING_CALIBRATION",
                "pose": None,
                "corners_px": [[1, 1], [5, 1], [5, 5], [1, 5]],
            } for marker in markers],
        },
    }]
    if targets:
        processors.append({
            "processor": "targets", "failure_reason": None,
            "data": {"results": [{
                "class_name": "gauge", "bbox_xyxy_px": [10, 8, 30, 20],
                "reading_status": "pending", "gauge_reading": None,
            }, {
                "class_name": "closedValve", "bbox_xyxy_px": [30, 8, 42, 22],
            }]},
        })
    record = {
        "schema_version": 1, "producer": "camera_stream", "run_id": "test-run",
        "source_kind": "camera", "source": "depthai", "sequence_number": sequence,
        "device_timestamp_s": now, "host_received_timestamp_s": now,
        "image_size_px": [48, 36], "processors": processors,
    }
    return original, annotated, record


def wait_for(predicate, timeout=3):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        value = predicate()
        if value:
            return value
        sleep(0.01)
    raise AssertionError("Photo worker did not finish the expected operation")


def read_packet(connection):
    def read(size):
        result = b""
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                raise EOFError("MQTT peer closed")
            result += chunk
        return result
    header = read(1)[0]
    size, multiplier = 0, 1
    while True:
        digit = read(1)[0]
        size += (digit & 127) * multiplier
        if not digit & 128:
            break
        multiplier *= 128
    return header, read(size)


@contextmanager
def broker(*, acknowledge=True):
    """A local MQTT peer records real wire bytes and optionally withholds PUBACK."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(0.1)
    stopped = Event()
    evidence = {"port": server.getsockname()[1], "received": Event(), "publications": []}

    def serve():
        try:
            while not stopped.is_set():
                try:
                    connection = server.accept()[0]
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(4)
                    read_packet(connection)
                    connection.sendall(b"\x20\x02\x00\x00")
                    header, body = read_packet(connection)
                    size = int.from_bytes(body[:2], "big")
                    evidence.update(
                        header=header,
                        topic=body[2:2 + size].decode(),
                        payload=json.loads(body[4 + size:]),
                    )
                    evidence["publications"].append((monotonic(), evidence["payload"]))
                    evidence["received"].set()
                    if acknowledge:
                        connection.sendall(b"\x40\x02" + body[2 + size:4 + size])
                    try:
                        read_packet(connection)
                    except (EOFError, OSError):
                        pass
        except Exception as error:
            if not stopped.is_set():
                evidence["error"] = error

    worker = Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield evidence
    finally:
        stopped.set()
        server.close()
        worker.join(5)
        if "error" in evidence:
            raise evidence["error"]


class PhotoCaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("detection_photos"),
            "needs the PhotoCapture implementation",
        )
        from detection_photos import PhotoCapture
        self.PhotoCapture = PhotoCapture
        self.temporary = TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "photos"
        self.capture = None

    def tearDown(self):
        if self.capture:
            self.capture.close()
        self.temporary.cleanup()

    def start(self, **options):
        self.capture = self.PhotoCapture(self.directory, **options)
        return self.capture

    def select(self, capture, *, markers=(), targets=True):
        self.assertFalse(capture.submit(*frame(0, 0, markers=markers, targets=targets)))
        args = frame(1, 0.31, markers=markers, targets=targets)
        self.assertTrue(capture.submit(*args))
        return args

    def receipt(self, sequence=1, timeout=3):
        path = self.directory / f"frame-{sequence}.delivery.json"
        wait_for(path.exists, timeout=timeout)
        return json.loads(path.read_text())

    def test_local_preview_saves_original_and_full_resolution_annotation(self):
        capture = self.start()
        original, annotated, record = self.select(capture)
        receipt = self.receipt()
        metadata = json.loads((self.directory / "frame-1.event.json").read_text())
        saved = cv2.imread(str(self.directory / metadata["image_file"]))
        np.testing.assert_array_equal(saved, original.image)
        annotation = self.directory / metadata["annotated_image"]["image_file"]
        self.assertEqual(cv2.imread(str(annotation)).shape, original.image.shape)
        self.assertEqual(metadata["record"], record)
        self.assertEqual(metadata["annotated_image"]["image_sha256"],
                         hashlib.sha256(annotation.read_bytes()).hexdigest())
        self.assertEqual(metadata["timestamp_basis"], "host_selection_time_not_camera_exposure")
        self.assertEqual(receipt["delivery"], "preview")
        self.assertIn("Pressure unavailable", receipt["description"])
        self.assertEqual(receipt["type"], "Gauge, Valve")
        self.assertTrue(np.all(original.image == 25))
        self.assertTrue(np.all(annotated.image[8:20, 10:30] == [20, 210, 30]))

    def test_actual_mqtt_publishes_saved_annotation_and_only_claims_broker_ack(self):
        with broker() as peer:
            capture = self.start(broker="127.0.0.1", port=peer["port"])
            self.select(capture, markers=[7])
            receipt = self.receipt()
            self.assertEqual(receipt["delivery"], "broker_acknowledged")
            self.assertEqual(peer["topic"], "gcs/detection")
            self.assertEqual(peer["header"] & 7, 2)
            payload = peer["payload"]
            self.assertEqual(set(payload), {"type", "description", "mime", "image", "timestamp"})
            self.assertEqual(payload["mime"], "image/jpeg")
            self.assertEqual(payload["type"], "ArUco, Gauge, Valve")
            self.assertEqual(base64.b64decode(payload["image"]),
                             (self.directory / "frame-1.annotated.jpg").read_bytes())
            self.assertIn("database", receipt["acknowledgement_scope"])
            capture.close()

    def test_targets_repeat_at_three_seconds_even_without_pressure(self):
        capture = self.start()
        self.select(capture)
        self.receipt()
        self.assertFalse(capture.submit(*frame(2, 3.30)))
        self.assertTrue(capture.submit(*frame(3, 3.32)))
        self.receipt(3)
        self.assertFalse(capture.submit(*frame(4, 3.5, targets=False)))
        self.assertFalse(capture.submit(*frame(5, 7.0)))
        self.assertTrue(capture.submit(*frame(6, 7.31)))

    def test_markers_need_real_absence_and_cooldown_before_new_sighting(self):
        capture = self.start()
        self.select(capture, markers=[0, 99], targets=False)
        self.receipt()
        self.assertFalse(capture.submit(*frame(2, 20, markers=[0, 99], targets=False)))
        self.assertFalse(capture.submit(*frame(3, 21, targets=False)))
        self.assertFalse(capture.submit(*frame(4, 22, markers=[0, 99], targets=False)))
        self.assertFalse(capture.submit(*frame(5, 23, targets=False)))
        self.assertFalse(capture.submit(*frame(6, 25.1, markers=[0, 99], targets=False)))
        self.assertTrue(capture.submit(*frame(7, 25.41, markers=[0, 99], targets=False)))

    def test_detector_failure_does_not_rearm_handled_markers(self):
        capture = self.start()
        self.select(capture, markers=[7], targets=False)
        self.receipt()
        capture.submit(*frame(2, 11, targets=False, aruco_failed=True))
        capture.submit(*frame(3, 15, markers=[7], targets=False))
        self.assertFalse(capture.submit(*frame(4, 16, markers=[7], targets=False)))

    def test_wrong_frame_annotation_is_rejected_before_consuming_selection(self):
        capture = self.start()
        capture.submit(*frame(0, 0))
        original, annotation, record = frame(1, 0.31)
        self.assertFalse(capture.submit(original, replace(annotation, sequence_number=8), record))
        self.assertTrue(capture.submit(original, annotation, record))
        self.receipt()
        capture.close()
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["stats"]["invalid_frames"], 1)

    def test_annotation_cannot_alias_original_pixel_storage(self):
        capture = self.start()
        original, annotation, record = frame(0, 0)
        self.assertFalse(capture.submit(original, replace(annotation, image=original.image), record))
        capture.close()
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["stats"]["invalid_frames"], 1)

    def test_storage_limit_keeps_camera_running_and_records_skip(self):
        capture = self.start(max_storage_bytes=1)
        self.select(capture)
        receipt = self.receipt()
        self.assertEqual(receipt["delivery"], "skipped_storage")
        self.assertFalse(list(self.directory.glob("*.png")))
        self.assertFalse(list(self.directory.glob("*.jpg")))

    def test_storage_exhaustion_stops_creating_unbounded_skip_receipts(self):
        capture = self.start(max_storage_bytes=1)
        self.select(capture)
        self.receipt()
        for sequence in range(2, 12):
            self.assertFalse(capture.submit(*frame(sequence, sequence * 4)))
        capture.close()
        self.assertEqual(len(list(self.directory.glob("*.delivery.json"))), 1)
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["stats"]["skipped_storage"], 11)

    def test_repeated_encoding_failures_have_bounded_receipt_storage(self):
        capture = self.start(max_storage_bytes=300)
        with patch("detection_photos.cv2.imencode", return_value=(False, None)):
            self.select(capture)
            self.receipt()
            for sequence in range(2, 12):
                if capture.submit(*frame(sequence, sequence * 4)):
                    self.receipt(sequence)
        capture.close()
        self.assertLessEqual(len(list(self.directory.glob("*.delivery.json"))), 3)

    def test_upload_spacing_uses_full_json_bandwidth_budget(self):
        with broker() as peer:
            capture = self.start(broker="127.0.0.1", port=peer["port"], max_mbps=0.001)
            self.select(capture)
            self.receipt()
            self.assertTrue(capture.submit(*frame(2, 3.32)))
            wait_for((self.directory / "frame-2.event.json").exists)
            sleep(3.15)
            self.assertEqual(len(peer["publications"]), 1)
            capture.close()
            self.assertEqual(self.receipt(2)["delivery"], "deferred")

    def test_available_bandwidth_still_preserves_three_second_upload_gap(self):
        with broker() as peer:
            capture = self.start(broker="127.0.0.1", port=peer["port"])
            self.select(capture)
            self.receipt()
            self.assertTrue(capture.submit(*frame(2, 3.32)))
            wait_for(lambda: len(peer["publications"]) == 2, timeout=4)
            first, second = [entry[0] for entry in peer["publications"]]
            self.assertGreaterEqual(second - first, 2.9)
            self.assertEqual(self.receipt(2)["delivery"], "broker_acknowledged")
            capture.close()

    def test_unavailable_broker_preserves_files_with_explicit_deferred_receipt(self):
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            capture = self.start(broker="127.0.0.1", port=unavailable.getsockname()[1])
            self.select(capture)
            self.assertEqual(self.receipt(timeout=5)["delivery"], "deferred")
            self.assertTrue((self.directory / "frame-1.png").exists())
            self.assertTrue((self.directory / "frame-1.annotated.jpg").exists())
            capture.close()

    def test_payload_limit_preserves_saved_pair_and_records_failure(self):
        capture = self.start()
        with patch("detection_photos.MAX_PAYLOAD_BYTES", 100):
            self.select(capture)
            self.assertEqual(self.receipt()["delivery"], "failed")
        self.assertTrue((self.directory / "frame-1.event.json").exists())

    def test_missing_puback_is_unconfirmed_and_is_never_automatically_replayed(self):
        with broker(acknowledge=False) as peer:
            capture = self.start(broker="127.0.0.1", port=peer["port"], publish_timeout_s=0.1)
            self.select(capture)
            receipt = self.receipt()
            self.assertEqual(receipt["delivery"], "unconfirmed")
            self.assertIn("uncertain", receipt["acknowledgement_scope"])
            capture.close()
        with self.assertRaises(FileExistsError):
            self.PhotoCapture(self.directory, broker="127.0.0.1", port=peer["port"])

    def test_close_interrupts_ack_wait_and_retains_accepted_queued_photos(self):
        with broker(acknowledge=False) as peer:
            capture = self.start(broker="127.0.0.1", port=peer["port"], publish_timeout_s=60)
            self.select(capture)
            self.assertTrue(peer["received"].wait(2))
            self.assertTrue(capture.submit(*frame(2, 3.32)))
            started = monotonic()
            capture.close()
            self.assertLess(monotonic() - started, 2)
            self.assertEqual(self.receipt()["delivery"], "unconfirmed")
            self.assertEqual(self.receipt(2)["delivery"], "deferred")
            self.assertTrue((self.directory / "frame-2.event.json").exists())
            self.assertFalse(capture.submit(*frame(3, 7)))

    def test_backpressure_counts_active_job_and_does_not_block_camera(self):
        with broker(acknowledge=False) as peer:
            capture = self.start(broker="127.0.0.1", port=peer["port"], max_pending=1)
            self.select(capture)
            self.assertTrue(peer["received"].wait(2))
            started = monotonic()
            self.assertFalse(capture.submit(*frame(2, 3.32)))
            self.assertLess(monotonic() - started, 0.1)
            capture.close()
            summary = json.loads((self.directory / "summary.json").read_text())
            self.assertEqual(summary["stats"]["skipped_queue"], 1)
            self.assertEqual(summary["last_rejection"]["event_id"], "test-run:2")

    def test_selected_images_are_copied_before_worker_uses_them(self):
        capture = self.start()
        entered, release = Event(), Event()
        real_encode = cv2.imencode
        def blocked_encode(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return real_encode(*args, **kwargs)
        with patch("detection_photos.cv2.imencode", side_effect=blocked_encode):
            original, annotation, record = self.select(capture)
            self.assertTrue(entered.wait(1))
            original.image[:] = 0
            annotation.image[:] = 0
            record["processors"] = []
            release.set()
            self.receipt()
        saved = cv2.imread(str(self.directory / "frame-1.png"))
        self.assertTrue(np.all(saved == 25))
        metadata = json.loads((self.directory / "frame-1.event.json").read_text())
        self.assertTrue(metadata["record"]["processors"])

    def test_encoding_failure_is_recorded_without_killing_worker(self):
        capture = self.start()
        with patch("detection_photos.cv2.imencode", return_value=(False, None)):
            self.select(capture)
            self.assertEqual(self.receipt()["delivery"], "failed")
        self.assertTrue(capture.submit(*frame(2, 3.32)))
        self.assertEqual(self.receipt(2)["delivery"], "preview")

    def test_invalid_limits_fail_before_creating_capture_directory(self):
        for options in ({"interval_s": 0}, {"max_pending": True}, {"max_mbps": float("nan")},
                        {"port": 0}, {"max_storage_bytes": 0}, {"publish_timeout_s": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.PhotoCapture(self.directory, **options)
            self.assertFalse(self.directory.exists())


if __name__ == "__main__":
    unittest.main()
