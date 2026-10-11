"""Original photos for marker sightings and periodic targets, with bounded storage."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stream_types import FramePacket


def frame(sequence, timestamp, marker_ids=(7,)):
    # Create a patterned source frame and a matching record for the requested marker IDs.
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))
    packet = FramePacket(image, sequence, timestamp - 0.01, timestamp)
    identity = dict(
        sequence_number=sequence,
        device_timestamp_s=timestamp - 0.01,
        host_received_timestamp_s=timestamp,
        image_size_px=[32, 24],
    )
    observations = [
        # Give each decoded marker an unavailable pose so capture can run without calibration.
        dict(
            marker_id=marker_id,
            corners_px=[[1.0, 1.0], [8.0, 1.0], [8.0, 8.0], [1.0, 8.0]],
            valid=False,
            pose=None,
            failure_reason="MISSING_CALIBRATION",
            **identity
        )
        for marker_id in marker_ids
    ]
    record = dict(
        schema_version=1,
        producer="camera_stream",
        run_id="run-abc",
        source_kind="camera",
        **identity,
        status="complete",
        processors=[
            dict(
                processor="aruco",
                identity=identity,
                failure_reason=None,
                error=None,
                data=dict(
                    config={"dictionary_name": "DICT_5X5_100"}, results=observations
                ),
            )
        ]
    )
    return packet, record


def target_frame(sequence, timestamp, targets, marker_ids=()):
    # Put the requested targets beside the marker branch in one matching frame.
    packet, record = frame(sequence, timestamp, marker_ids)
    record["processors"].append(
        dict(
            processor="targets",
            failure_reason=None,
            data=dict(results=deepcopy(targets)),
        )
    )
    return packet, record


def target(class_name="gauge", track_id=1):
    return dict(
        track_id=track_id,
        class_name=class_name,
        bbox_xyxy_px=[2, 3, 20, 22],
        gauge_reading=None,
        raw_gauge_reading=dict(valid=False, failure_reason="DIAL_TOO_SMALL"),
    )


class DetectionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("stream_detection_capture"),
            "The one-image-per-sighting capture module must exist",
        )
        import stream_detection_capture

        self.module = stream_detection_capture
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "detections"

    def capture(self, **options):
        capture = self.module.DetectionCapture(self.directory, **options)
        self.addCleanup(capture.close)
        return capture

    def events(self):
        return sorted(self.directory.glob("*.event.json"))

    def audit(self):
        return [
            json.loads(line)
            for line in (self.directory / "capture-audit.jsonl")
            .read_text()
            .splitlines()
        ]

    def test_first_capture_waits_for_continuous_stability(self):
        capture = self.capture()

        # Show marker 7 for only 0.2 seconds and check that no capture is saved yet.
        capture(*frame(1, 0.0))
        capture(*frame(2, 0.2))
        self.assertEqual(self.events(), [])

        capture(*frame(3, 0.31))
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(json.loads(self.events()[0].read_text())["sequence_number"], 3)

    def test_occlusion_before_stability_restarts_the_confirmation_period(self):
        capture = self.capture()

        for sequence, timestamp, ids in [
            (1, 0.0, [7]),
            (2, 0.2, []),
            (3, 0.31, [7]),
            (4, 0.51, [7]),
        ]:
            capture(*frame(sequence, timestamp, ids))

        self.assertEqual(self.events(), [])
        capture(*frame(5, 0.62))
        self.assertEqual(len(self.events()), 1)

    def test_continuously_visible_marker_does_not_repeat_after_cooldown(self):
        capture = self.capture()

        for sequence, timestamp in enumerate((0.0, 0.31, 1.0, 11.0, 25.0), 1):
            capture(*frame(sequence, timestamp))

        self.assertEqual(len(self.events()), 1)

    def test_short_occlusion_does_not_rearm_a_handled_sighting(self):
        capture = self.capture()

        for sequence, timestamp, ids in [
            (1, 0.0, [7]),
            (2, 0.31, [7]),
            (3, 0.5, []),
            (4, 1.0, [7]),
            (5, 1.31, [7]),
            (6, 12.0, [7]),
        ]:
            capture(*frame(sequence, timestamp, ids))

        self.assertEqual(len(self.events()), 1)

    def test_genuine_absence_rearms_but_a_visible_new_sighting_waits_out_cooldown(self):
        capture = self.capture()

        # Remove marker 7 for over two seconds, then show it again before the cooldown ends.
        for sequence, timestamp, ids in [
            (1, 0.0, [7]),
            (2, 0.31, [7]),
            (3, 0.6, []),
            (4, 3.0, []),
            (5, 4.0, [7]),
            (6, 4.31, [7]),
        ]:
            capture(*frame(sequence, timestamp, ids))

        self.assertEqual(len(self.events()), 1)
        capture(*frame(7, 10.4))
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(json.loads(self.events()[1].read_text())["sequence_number"], 7)

    def test_absence_starts_at_first_missing_observation_not_last_seen_frame(self):
        capture = self.capture()

        for sequence, timestamp, ids in [
            (1, 0.0, [7]),
            (2, 0.31, [7]),
            (3, 5.0, []),
            (4, 6.9, [7]),
            (5, 20.0, [7]),
        ]:
            capture(*frame(sequence, timestamp, ids))

        self.assertEqual(len(self.events()), 1)

    def test_withheld_pose_still_captures_lossless_original_pixels_and_same_frame_evidence(
        self,
    ):
        capture = self.capture()
        capture(*frame(1, 0.0))
        packet, record = frame(2, 0.31)
        original_pixels, original_record = packet.image.copy(), deepcopy(record)
        packet.image.flags.writeable = False
        capture(packet, record)
        event = json.loads(self.events()[0].read_text())
        self.assertEqual(event["event_id"], "run-abc:2")
        self.assertEqual(event["marker_ids"], [7])
        self.assertEqual(event["dictionary_name"], "DICT_5X5_100")
        self.assertEqual(event["source_kind"], "camera")
        self.assertEqual(
            event["timestamp_basis"], "host_selection_time_not_camera_exposure"
        )
        self.assertTrue(event["observed_at_utc"].endswith("Z"))
        self.assertEqual(event["record"], original_record)
        self.assertFalse(Path(event["image_file"]).is_absolute())
        self.assertEqual(Path(event["image_file"]).name, event["image_file"])
        original_file = (self.directory / event["image_file"]).read_bytes()
        self.assertEqual(event["image_bytes"], len(original_file))
        self.assertEqual(
            event["image_sha256"], hashlib.sha256(original_file).hexdigest()
        )
        decoded = cv2.imdecode(
            np.frombuffer(original_file, np.uint8), cv2.IMREAD_UNCHANGED
        )
        np.testing.assert_array_equal(decoded, original_pixels)
        np.testing.assert_array_equal(packet.image, original_pixels)
        self.assertEqual(record, original_record)
        self.assertFalse(packet.image.flags.writeable)

    def test_simultaneously_eligible_ids_share_one_png_and_one_event(self):
        capture = self.capture()
        capture(*frame(1, 0.0, [8, 7, 7]))

        # Count PNG encoding calls to check that markers 7 and 8 share one saved image.
        with patch(
            "stream_detection_capture.cv2.imencode", wraps=cv2.imencode
        ) as encode:
            capture(*frame(2, 0.31, [8, 7, 7]))

        self.assertEqual(encode.call_count, 1)
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(len(list(self.directory.glob("*.png"))), 1)
        self.assertEqual(json.loads(self.events()[0].read_text())["marker_ids"], [7, 8])
        self.assertEqual(capture.stats["captured_marker_sightings"], 2)

    def test_full_pending_queue_skips_and_marks_sighting_handled(self):
        capture = self.capture(max_pending=1)

        for sequence, timestamp, ids in [
            (1, 0.0, [7]),
            (2, 0.31, [7]),
            (3, 1.0, [8]),
            (4, 1.31, [8]),
        ]:
            capture(*frame(sequence, timestamp, ids))

        self.assertEqual(len(self.events()), 1)
        self.assertEqual(capture.stats["skipped_pending"], 1)
        # Free the queue slot after marker 8 was skipped; that sighting must stay handled.
        self.events()[0].with_suffix(".sent.json").write_text("{}")

        for sequence, timestamp in [(5, 2.0), (6, 20.0)]:
            capture(*frame(sequence, timestamp, [8]))

        self.assertEqual(len(self.events()), 1)
        skipped = [
            entry for entry in self.audit() if entry.get("reason") == "PENDING_LIMIT"
        ]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["marker_ids"], [8])

    def test_sent_or_failed_receipt_releases_capacity_for_a_new_sighting(self):
        for receipt in (".sent.json", ".failed.json"):
            with self.subTest(receipt=receipt), TemporaryDirectory() as directory:
                root = Path(directory) / "captures"
                capture = self.module.DetectionCapture(root, max_pending=1)

                try:
                    capture(*frame(1, 0.0, [7]))
                    capture(*frame(2, 0.31, [7]))
                    event = next(root.glob("*.event.json"))
                    # Mark marker 7's capture finished before marker 8 becomes eligible.
                    event.with_suffix(receipt).write_text("{}")
                    capture(*frame(3, 1.0, [8]))
                    capture(*frame(4, 1.31, [8]))
                    self.assertEqual(len(list(root.glob("*.event.json"))), 2)

                finally:
                    capture.close()

    def test_storage_limit_counts_prior_images_and_does_not_delete_them(self):
        image_bytes = len(cv2.imencode(".png", frame(1, 0.0)[0].image)[1].tobytes())
        # Leave room for one encoded image but make a second image exceed the storage limit.
        capture = self.capture(max_storage_bytes=2 * image_bytes - 1)
        capture(*frame(1, 0.0, [7]))
        capture(*frame(2, 0.31, [7]))
        first_image = next(self.directory.glob("*.png"))
        preserved = first_image.read_bytes()
        capture(*frame(3, 1.0, [8]))
        capture(*frame(4, 1.31, [8]))
        capture(*frame(5, 20.0, [8]))
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(first_image.read_bytes(), preserved)
        self.assertEqual(capture.stats["image_bytes"], image_bytes)
        self.assertEqual(capture.stats["skipped_storage"], 1)
        self.assertEqual(
            [entry["reason"] for entry in self.audit() if entry["status"] == "skipped"],
            ["STORAGE_LIMIT"],
        )

    def test_existing_event_file_is_never_overwritten_and_failure_does_not_escape(self):
        capture = self.capture()
        capture(*frame(1, 0.0))
        existing = self.directory / "frame-2.png"
        existing.write_bytes(b"existing evidence")
        capture(*frame(2, 0.31))
        self.assertEqual(existing.read_bytes(), b"existing evidence")
        self.assertEqual(self.events(), [])
        self.assertEqual(capture.stats["failed_events"], 1)
        self.assertEqual(self.audit()[0]["reason"], "CAPTURE_FAILED")

    def test_encode_failure_is_audited_without_repeated_attempts_for_visible_marker(
        self,
    ):
        capture = self.capture()
        capture(*frame(1, 0.0))

        with patch(
            "stream_detection_capture.cv2.imencode",
            side_effect=RuntimeError("encoder failed"),
        ):
            capture(*frame(2, 0.31))

        capture(*frame(3, 20.0))
        self.assertEqual(self.events(), [])
        self.assertEqual(capture.stats["failed_events"], 1)
        self.assertIn("encoder failed", self.audit()[0]["error"])

    def test_mismatched_frame_record_cannot_be_attached_to_saved_pixels(self):
        capture = self.capture()
        capture(*frame(1, 0.0))
        packet, record = frame(2, 0.31)
        record["sequence_number"] = 99
        capture(packet, record)
        self.assertEqual(self.events(), [])
        self.assertEqual(capture.stats["failed_events"], 1)
        self.assertIn("identity", self.audit()[0]["error"].lower())

    def test_sidecar_is_published_only_after_complete_png_and_metadata_failure_keeps_storage_accounted(
        self,
    ):
        capture = self.capture()
        capture(*frame(1, 0.0))
        link = os.link
        commits = []

        def fail_sidecar_commit(temporary, destination):
            destination = Path(destination)
            commits.append(destination.name)

            # Check that the complete PNG exists before failing the companion JSON commit.
            if destination.name.endswith(".event.json"):
                metadata = json.loads(Path(temporary).read_text())
                original = (self.directory / metadata["image_file"]).read_bytes()
                self.assertEqual(
                    hashlib.sha256(original).hexdigest(), metadata["image_sha256"]
                )
                self.assertFalse(destination.exists())
                raise OSError("sidecar commit failed")

            return link(temporary, destination)

        with patch("stream_detection_capture.os.link", side_effect=fail_sidecar_commit):
            capture(*frame(2, 0.31))

        self.assertEqual(commits, ["frame-2.png", "frame-2.event.json"])
        self.assertEqual(self.events(), [])
        self.assertEqual(capture.stats["captured_events"], 0)
        self.assertEqual(capture.stats["failed_events"], 1)
        self.assertEqual(
            capture.stats["image_bytes"],
            (self.directory / "frame-2.png").stat().st_size,
        )
        self.assertEqual(list(self.directory.glob("*.tmp")), [])
        self.assertIn("sidecar commit failed", self.audit()[0]["error"])

    def test_processor_failure_breaks_stability_but_cannot_prove_absence(self):
        capture = self.capture()
        capture(*frame(1, 0.0))
        packet, failed = frame(2, 0.25)
        failed["processors"][0]["failure_reason"] = "PROCESSOR_ERROR"
        capture(packet, failed)
        capture(*frame(3, 0.4))
        capture(*frame(4, 0.65))
        self.assertEqual(self.events(), [])
        capture(*frame(5, 0.71))
        self.assertEqual(len(self.events()), 1)

        # Report processing failures for four seconds; they must not count as marker absence.
        for sequence, timestamp in [(6, 1.0), (7, 5.0)]:
            packet, failed = frame(sequence, timestamp)
            failed["processors"][0]["failure_reason"] = "PROCESSOR_ERROR"
            capture(packet, failed)

        capture(*frame(8, 6.0))
        capture(*frame(9, 6.31))
        capture(*frame(10, 15.0))
        self.assertEqual(len(self.events()), 1)

    def test_unsupported_dictionary_or_id_is_audited_and_cannot_expand_marker_state(
        self,
    ):
        capture = self.capture()

        for sequence, marker in enumerate((-1, 100, 1000000), 1):
            capture(*frame(sequence, float(sequence), [marker]))

        packet, record = frame(4, 4.0)
        record["processors"][0]["data"]["config"][
            "dictionary_name"
        ] = "arbitrary_dictionary"
        capture(packet, record)
        self.assertEqual(capture.stats["failed_events"], 4)
        self.assertEqual(self.events(), [])
        # Show every supported ID together and check the rejected IDs added no sightings.
        capture(*frame(5, 5.0, range(100)))
        capture(*frame(6, 5.31, range(100)))
        self.assertEqual(capture.stats["captured_marker_sightings"], 100)
        self.assertEqual(
            json.loads(self.events()[0].read_text())["marker_ids"], list(range(100))
        )

    def test_audit_write_failure_is_counted_without_stopping_capture(self):
        capture = self.capture()
        capture(*frame(1, 0.0))

        with patch.object(
            capture._audit_file, "write", side_effect=OSError("audit disk full")
        ):
            capture(*frame(2, 0.31))

        self.assertEqual(len(self.events()), 1)
        self.assertEqual(capture.stats["audit_failures"], 1)
        capture.close()
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertIn("audit disk full", summary["last_error"])

    def test_close_writes_summary_once_and_rejects_existing_capture_directory(self):
        capture = self.capture()
        capture(*frame(1, 0.0))
        capture(*frame(2, 0.31))
        capture.close()
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["stats"]["captured_events"], 1)
        self.assertEqual(summary["policy"]["stable_s"], 0.3)
        self.assertEqual(summary["policy"]["absence_s"], 2.0)
        self.assertEqual(summary["policy"]["cooldown_s"], 10.0)
        self.assertEqual(summary["policy"]["max_pending"], 3)
        self.assertEqual(summary["policy"]["max_storage_bytes"], 256 * 1024 * 1024)
        capture.close()
        self.assertEqual(
            json.loads((self.directory / "summary.json").read_text()), summary
        )

        with self.assertRaises(FileExistsError):
            self.module.DetectionCapture(self.directory)

    def test_invalid_configuration_is_rejected_before_creating_directory(self):
        for options in (
            {"stable_s": float("nan")},
            {"absence_s": -1},
            {"cooldown_s": -1},
            {"max_pending": 0},
            {"max_pending": True},
            {"max_storage_bytes": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module.DetectionCapture(self.directory, **options)

        self.assertFalse(self.directory.exists())

    def test_targets_share_periodic_original_photos_without_mutating_evidence(self):
        capture = self.capture()
        targets = [target(), target("openValve", 2), target("closedValve", 3)]
        capture(*target_frame(1, 0.0, targets))
        capture(*target_frame(2, 0.2, targets))
        self.assertEqual(self.events(), [])
        packet, record = target_frame(3, 0.31, targets)
        original_record = deepcopy(record)
        original_pixels = packet.image.copy()

        with patch(
            "stream_detection_capture.cv2.imencode", wraps=cv2.imencode
        ) as encode:
            capture(packet, record)

        self.assertEqual(len(self.events()), 1)
        self.assertEqual(encode.call_count, 1)
        event = json.loads(self.events()[0].read_text())
        self.assertEqual(event["schema_version"], 1)
        self.assertEqual(event["marker_ids"], [])
        self.assertEqual(event["target_results"], targets)
        self.assertEqual(event["record"], original_record)
        decoded = cv2.imread(str(self.directory / event["image_file"]))
        np.testing.assert_array_equal(decoded, original_pixels)
        self.assertEqual(record, original_record)

        # Continually visible targets get one shared photo every three seconds.
        capture(*target_frame(4, 3.30, targets))
        self.assertEqual(len(self.events()), 1)
        changed = deepcopy(targets)
        changed[0]["gauge_reading"] = dict(
            valid=True, value_bar=2.9, failure_reason=None
        )
        changed[1]["class_name"] = "closedValve"
        capture(*target_frame(5, 3.32, changed))
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(
            json.loads(self.events()[1].read_text())["target_results"], changed
        )

    def test_targets_capture_without_aruco_branch_or_track_id(self):
        capture = self.capture(stable_s=0)
        observation = target()
        del observation["track_id"]
        packet, record = target_frame(1, 0.0, [observation])
        record["processors"] = record["processors"][1:]
        capture(packet, record)
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(
            json.loads(self.events()[0].read_text())["target_results"], [observation]
        )

    def test_empty_or_failed_target_branch_restarts_target_stability(self):
        capture = self.capture()
        observation = target()
        capture(*target_frame(1, 0.0, [observation]))
        capture(*target_frame(2, 0.2, []))
        capture(*target_frame(3, 0.3, [observation]))
        packet, record = target_frame(4, 0.5, [observation])
        record["processors"][1]["failure_reason"] = "PROCESSOR_ERROR"
        capture(packet, record)
        capture(*target_frame(5, 0.6, [observation]))
        capture(*target_frame(6, 0.8, [observation]))
        self.assertEqual(self.events(), [])
        capture(*target_frame(7, 0.91, [observation]))
        self.assertEqual(len(self.events()), 1)

    def test_failed_aruco_branch_does_not_block_target_photos(self):
        capture = self.capture()

        for sequence, timestamp in [(1, 0.0), (2, 0.31)]:
            packet, record = target_frame(sequence, timestamp, [target()], [7])
            record["processors"][0]["failure_reason"] = "PROCESSOR_ERROR"
            capture(packet, record)

        self.assertEqual(len(self.events()), 1)
        event = json.loads(self.events()[0].read_text())
        self.assertEqual(event["marker_ids"], [])
        self.assertEqual(len(event["target_results"]), 1)

    def test_failed_targets_do_not_block_markers_or_attach_stale_targets(self):
        capture = self.capture()

        for sequence, timestamp in [(1, 0.0), (2, 0.31)]:
            packet, record = target_frame(sequence, timestamp, [target()], [7])
            record["processors"][1]["failure_reason"] = "PROCESSOR_ERROR"
            capture(packet, record)

        self.assertEqual(len(self.events()), 1)
        event = json.loads(self.events()[0].read_text())
        self.assertEqual(event["marker_ids"], [7])
        self.assertEqual(event["target_results"], [])

    def test_marker_photo_includes_visible_targets_and_restarts_target_interval(self):
        capture = self.capture()
        capture(*target_frame(1, 0.0, [target()]))
        capture(*target_frame(2, 0.31, [target()]))
        capture(*target_frame(3, 1.0, [target()], [7]))
        capture(*target_frame(4, 1.31, [target()], [7]))
        self.assertEqual(len(self.events()), 2)
        mixed = json.loads(self.events()[1].read_text())
        self.assertEqual(mixed["marker_ids"], [7])
        self.assertEqual(mixed["target_results"], [target()])
        capture(*target_frame(5, 3.32, [target()], [7]))
        self.assertEqual(len(self.events()), 2)
        capture(*target_frame(6, 4.32, [target()], [7]))
        self.assertEqual(len(self.events()), 3)

    def test_full_target_queue_paces_attempts_and_retries_after_capacity_returns(self):
        capture = self.capture(stable_s=0, max_pending=1)
        capture(*target_frame(1, 0.0, [target()]))
        capture(*target_frame(2, 3.0, [target("closedValve", 2)]))
        self.assertEqual(capture.stats["skipped_pending"], 1)

        for sequence, timestamp in [(3, 3.1), (4, 4.0), (5, 5.9)]:
            capture(*target_frame(sequence, timestamp, [target("closedValve", 2)]))

        self.assertEqual(capture.stats["skipped_pending"], 1)
        self.events()[0].with_suffix(".sent.json").write_text("{}")
        capture(*target_frame(6, 6.0, [target("closedValve", 2)]))
        self.assertEqual(len(self.events()), 2)
        saved_targets = json.loads(self.events()[1].read_text())["target_results"]
        self.assertEqual(saved_targets[0]["track_id"], 2)

    def test_target_storage_failures_are_paced(self):
        capture = self.capture(stable_s=0, max_storage_bytes=1)

        for sequence, timestamp in enumerate([0.0, 0.1, 1.0, 2.99, 3.0], 1):
            capture(*target_frame(sequence, timestamp, [target()]))

        self.assertEqual(self.events(), [])
        self.assertEqual(capture.stats["skipped_storage"], 2)
        self.assertEqual(len(self.audit()), 2)

    def test_unknown_targets_do_not_start_periodic_capture(self):
        capture = self.capture(stable_s=0)
        capture(*target_frame(1, 0.0, [target("other")]))
        capture(*target_frame(2, 10.0, [target("other")]))
        self.assertEqual(self.events(), [])

    def test_target_interval_is_positive_finite_and_recorded_in_policy(self):
        for interval in [0, -1, True, float("nan"), float("inf")]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.module.DetectionCapture(self.directory, target_interval_s=interval)

        capture = self.capture(stable_s=0, target_interval_s=5.0)
        capture(*target_frame(1, 0.0, [target()]))
        capture(*target_frame(2, 3.0, [target()]))
        self.assertEqual(len(self.events()), 1)
        capture(*target_frame(3, 5.0, [target()]))
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(capture.policy["target_interval_s"], 5.0)


if __name__ == "__main__":
    unittest.main()
