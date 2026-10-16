"""Exercise the sender lifecycle with small frames instead of real hardware."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run_stream
from stream_types import FramePacket
from session_records import SessionRecords


class StreamCommandTests(unittest.TestCase):
    def test_finite_oak_run_saves_identity_and_closes_every_resource(self):
        source, sender = Mock(), Mock()
        packets = [FramePacket(np.zeros((80, 100, 3), np.uint8), n, float(n), float(n)) for n in range(3)]
        source.read.side_effect = packets
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            with (
                patch("depthai_source.DepthAIFrameSource", return_value = source),
                patch.object(run_stream, "create_sender", return_value = sender),
            ):
                try:
                    code = run_stream.main([
                        "--source", "oak", "--no-targets", "--no-aruco", "--no-photos",
                        "--max-frames", "3", "--output-dir", folder,
                    ])
                except TypeError as error:
                    self.fail(f"Sender needs configurable finite runs: {error}")

            self.assertEqual(code, 0)
            summary = json.loads((Path(folder) / "run.json").read_text())
            frames = [json.loads(line) for line in (Path(folder) / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["frames_recorded"], 3)
            self.assertEqual(sender.send.call_count, 3)
            for packet, record, sent in zip(packets, frames, sender.send.call_args_list):
                self.assertEqual(record, sent.kwargs["message"])
                self.assertEqual(record["event_id"], f"{summary['run_id']}:{packet.sequence_number}")
                self.assertEqual(record["timestamp_basis"], "depthai_capture_and_host_arrival")
                self.assertEqual(record["source_kind"], "camera")
                self.assertEqual(record["source"], "depthai")
                self.assertEqual(record["image_size_px"], [100, 80])
                self.assertFalse(np.shares_memory(sent.args[0], packet.image))
            source.close.assert_called_once()
            sender.close.assert_called_once()

    def test_camera_failure_keeps_recorded_frames_and_marks_run_failed(self):
        source, sender = Mock(), Mock()
        source.read.side_effect = [
            FramePacket(np.zeros((80, 100, 3), np.uint8), 0, 1.0, 1.0),
            RuntimeError("camera unplugged"),
        ]
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            with (
                patch("depthai_source.DepthAIFrameSource", return_value = source),
                patch.object(run_stream, "create_sender", return_value = sender),
            ):
                try:
                    code = run_stream.main([
                        "--source", "oak", "--no-targets", "--no-aruco", "--no-photos", "--output-dir", folder,
                    ])
                except TypeError as error:
                    self.fail(f"Sender needs recorded failure handling: {error}")
            summary = json.loads((Path(folder) / "run.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["frames_recorded"], 1)
            self.assertIn("camera unplugged", summary["error"])
            source.close.assert_called_once()
            sender.close.assert_called_once()

    def test_final_receipt_failure_produces_a_failed_exit_and_summary(self):
        class FailingReceiptSession(SessionRecords):
            def record(self, record):
                super().record(record)
                # Simulate a background receipt write failing on the final frame.
                self._write_error = OSError("disk full")

        source, sender = Mock(), Mock()
        source.read.return_value = FramePacket(np.zeros((80, 100, 3), np.uint8), 0, 1.0, 1.0)
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()) as output:
            with (
                patch("depthai_source.DepthAIFrameSource", return_value = source),
                patch.object(run_stream, "create_sender", return_value = sender),
                patch.object(run_stream, "SessionRecords", FailingReceiptSession),
            ):
                code = run_stream.main([
                    "--source", "oak", "--no-targets", "--no-aruco", "--no-photos", "--max-frames", "1",
                    "--output-dir", folder,
                ])
            summary = json.loads((Path(folder) / "run.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIn("Status: failed", output.getvalue())


    def test_no_video_keeps_synthetic_processing_and_evidence_without_netgear(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            with patch.object(
                run_stream,
                "create_sender",
                side_effect = AssertionError("Acquisition-only mode opened NetGear"),
            ):
                try:
                    code = run_stream.main([
                        "--source", "synthetic", "--no-targets", "--no-video",
                        "--profile", "480p20", "--max-frames", "10", "--output-dir", folder,
                    ])
                except SystemExit as error:
                    self.fail(f"Sender rejected acquisition-only mode: {error}")
            self.assertEqual(code, 0)
            root = Path(folder)
            summary = json.loads((root / "run.json").read_text())
            records = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["frames_recorded"], 10)
            self.assertTrue(all(record["source_kind"] == "synthetic" for record in records))
            self.assertTrue(all(record["processors"][0]["processor"] == "aruco" for record in records))
            events = list((root / "detections").glob("*.event.json"))
            self.assertEqual(len(events), 1)
            event = json.loads(events[0].read_text())
            self.assertIn(event["record"], records)
            self.assertTrue((root / "detections" / event["image_file"]).exists())
            self.assertTrue((root / "detections" / event["annotated_image"]["image_file"]).exists())

    def test_default_source_opens_oak_and_preserves_camera_identity(self):
        source, sender = Mock(), Mock()
        source.read.return_value = FramePacket(np.zeros((80, 100, 3), np.uint8), 0, 1.0, 1.0)
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            with (
                patch("depthai_source.DepthAIFrameSource", return_value = source) as oak,
                patch.object(run_stream, "create_sender", return_value = sender),
            ):
                code = run_stream.main([
                    "--no-targets", "--no-aruco", "--no-photos", "--max-frames", "1",
                    "--output-dir", folder,
                ])
            self.assertEqual(code, 0)
            oak.assert_called_once()
            record = json.loads((Path(folder) / "frames.jsonl").read_text())
            self.assertEqual(record["source_kind"], "camera")
            self.assertEqual(record["source"], "depthai")
            self.assertEqual(record["timestamp_basis"], "depthai_capture_and_host_arrival")
            source.close.assert_called_once()


    def test_removed_camera_options_are_rejected_before_resources_open(self):
        for removed_option in (("--source", "webcam"), ("--camera-index", "0")):
            with self.subTest(arguments = removed_option):
                with (
                    redirect_stderr(StringIO()),
                    redirect_stdout(StringIO()),
                    patch.object(run_stream, "SessionRecords") as session,
                    patch.object(run_stream, "open_source") as source,
                    patch.object(run_stream, "create_sender") as sender,
                ):
                    session.side_effect = AssertionError("Created evidence before CLI rejection")
                    with self.assertRaises(SystemExit) as stopped:
                        run_stream.main([
                            *removed_option,
                            "--no-targets", "--no-aruco", "--no-photos", "--max-frames", "1",
                        ])
                    self.assertEqual(stopped.exception.code, 2)
                    session.assert_not_called()
                    source.assert_not_called()
                    sender.assert_not_called()


if __name__ == "__main__":
    unittest.main()
