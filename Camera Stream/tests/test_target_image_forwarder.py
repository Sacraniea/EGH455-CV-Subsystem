"""Keep gauge and valve photos tied to their original frame and accepted readings."""

import base64
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from run_gcs_image_forwarder import ImageForwarder, _prepare
from test_gcs_image_forwarder import capture, Clock, RecordingPublisher
from test_stream_mqtt import accept_client, broker_peer, read_packet


def target_capture(directory, classes=("gauge", "openValve"), *, markers=False):
    sidecar, metadata, original = capture(directory)
    targets = []

    for index, name in enumerate(classes):
        targets.append({
            "track_id": f"target-{index}",
            "class_name": name,
            "bbox_xyxy_px": [0, 0, 3, 2],
            "gauge_reading": (
                {"valid": True, "value_bar": 2.3, "failure_reason": None}
                if name == "gauge" else None
            ),
            "reading_status": "confirmed" if name == "gauge" else None,
        })

    if not markers:
        metadata["marker_ids"] = []
        metadata["record"]["processors"] = []

    metadata["record"]["processors"].append({
        "processor": "targets",
        "failure_reason": None,
        "data": {"results": deepcopy(targets)},
    })
    metadata["target_results"] = targets
    sidecar.write_text(json.dumps(metadata))
    return sidecar, metadata, original


class TargetImageForwarderTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()

    def test_target_only_photo_keeps_both_detections_and_original_file_bytes(self):
        sidecar, metadata, original = target_capture(self.directory)
        before = sidecar.read_bytes()

        _, payload, _ = _prepare(sidecar, self.directory, 12_000_000)

        self.assertEqual(payload["type"], "Gauge, Valve")
        self.assertIn("Pressure: 2.3 bar", payload["description"])
        self.assertIn("Open valve", payload["description"])
        self.assertIn("Frame 1", payload["description"])
        self.assertEqual(base64.b64decode(payload["image"]), original)
        self.assertEqual(payload["timestamp"], metadata["observed_at_utc"])
        self.assertEqual(sidecar.read_bytes(), before)
        self.assertEqual(
            set(payload), {"type", "description", "mime", "image", "timestamp"}
        )

    def test_mixed_marker_gauge_and_valve_share_one_photo_payload(self):
        sidecar, _, original = target_capture(self.directory, markers=True)
        _, payload, _ = _prepare(sidecar, self.directory, 12_000_000)

        self.assertEqual(payload["type"], "ArUco, Gauge, Valve")
        self.assertIn("Marker 7", payload["description"])
        self.assertIn("Open valve", payload["description"])
        self.assertEqual(base64.b64decode(payload["image"]), original)

    def test_unconfirmed_raw_pressure_is_not_used_in_the_photo_caption(self):
        sidecar, metadata, _ = target_capture(self.directory, ("gauge",))
        target = metadata["target_results"][0]
        target["raw_gauge_reading"] = {
            "valid": True, "value_bar": 6.22, "failure_reason": None,
        }
        target["reading_status"] = "pending"
        target["gauge_reading"] = {
            "valid": False, "value_bar": None,
            "failure_reason": "READING_UNCONFIRMED",
        }
        metadata["record"]["processors"][0]["data"]["results"] = [deepcopy(target)]
        sidecar.write_text(json.dumps(metadata))

        _, payload, _ = _prepare(sidecar, self.directory, 12_000_000)

        self.assertIn("awaiting confirmation", payload["description"])
        self.assertNotIn("6.22", payload["description"])
        self.assertNotIn("Pressure:", payload["description"])

    def test_rejects_target_evidence_that_does_not_belong_to_the_saved_frame(self):
        sidecar, metadata, _ = target_capture(self.directory)
        metadata["target_results"][0]["bbox_xyxy_px"] = [100, 100, 200, 200]
        sidecar.write_text(json.dumps(metadata))

        with self.assertRaisesRegex(ValueError, "target.*match"):
            _prepare(sidecar, self.directory, 12_000_000)

    def test_empty_failure_reason_still_rejects_target_photo(self):
        sidecar, metadata, _ = target_capture(self.directory)
        metadata["record"]["processors"][0]["failure_reason"] = ""
        sidecar.write_text(json.dumps(metadata))

        with self.assertRaisesRegex(ValueError, "successful target processor"):
            _prepare(sidecar, self.directory, 12_000_000)

    def test_three_second_upload_interval_still_waits_for_its_budget(self):
        clock = Clock()
        publisher = RecordingPublisher(clock)
        uploader = ImageForwarder(
            self.directory,
            publish=True,
            broker="127.0.0.1",
            interval_s=3.0,
            clock=clock,
            publisher_factory=lambda **kwargs: publisher,
        )
        self.addCleanup(uploader.close)
        capture(self.directory, 1)
        capture(self.directory, 2)

        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")
        clock.now = 2.99
        self.assertIsNone(uploader.step())
        clock.now = 3.0
        self.assertEqual(uploader.step()["delivery"], "broker_acknowledged")

    def test_target_photo_reaches_mqtt_as_one_original_image_and_gets_a_receipt(self):
        sidecar, _, original = target_capture(self.directory)
        received = []

        def broker(connection):
            accept_client(connection)
            header, body = read_packet(connection)
            topic_length = int.from_bytes(body[:2], "big")
            received.append((
                header,
                body[2:2 + topic_length].decode(),
                json.loads(body[4 + topic_length:]),
            ))
            connection.sendall(b"\x40\x02" + body[2 + topic_length:4 + topic_length])
            read_packet(connection)

        with broker_peer(broker) as port:
            uploader = ImageForwarder(
                self.directory,
                publish=True,
                broker="127.0.0.1",
                port=port,
                interval_s=3.0,
            )
            try:
                outcome = uploader.step()
            finally:
                uploader.close()

        self.assertEqual(outcome["delivery"], "broker_acknowledged")
        self.assertEqual(len(received), 1)
        header, topic, payload = received[0]
        self.assertEqual((header, topic), (0x32, "gcs/detection"))
        self.assertEqual(payload["type"], "Gauge, Valve")
        self.assertEqual(base64.b64decode(payload["image"]), original)
        self.assertEqual(
            json.loads(sidecar.with_suffix(".sent.json").read_text())["delivery"],
            "broker_acknowledged",
        )


if __name__ == "__main__":
    unittest.main()
