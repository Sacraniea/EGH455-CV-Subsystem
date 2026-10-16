"""Check that a saved run can identify each frame and delivery outcome."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class SessionRecordsTests(unittest.TestCase):
    def session_class(self):
        self.assertIsNotNone(
            importlib.util.find_spec("session_records"),
            "needs saved frame and run evidence",
        )
        from session_records import SessionRecords
        return SessionRecords

    def test_records_frames_and_delivery_separately_under_one_run_id(self):
        with TemporaryDirectory() as folder:
            session = self.session_class()(Path(folder), {"source": "oak"})
            session.record({"run_id": session.run_id, "sequence_number": 4})
            session.record({"run_id": session.run_id, "sequence_number": 7})
            session.mqtt_event({"status": "broker_acknowledged", "event_id": f"{session.run_id}:4"})
            session.close("complete")

            summary = json.loads((Path(folder) / "run.json").read_text())
            frames = [json.loads(line) for line in (Path(folder) / "frames.jsonl").read_text().splitlines()]
            events = [json.loads(line) for line in (Path(folder) / "gcs-logs.jsonl").read_text().splitlines()]
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["frames_recorded"], 2)
            self.assertEqual(summary["sequence_gaps"], 2)
            self.assertEqual([frame["sequence_number"] for frame in frames], [4, 7])
            self.assertEqual(events[0]["event_id"], f"{summary['run_id']}:4")

    def test_rejects_other_runs_and_repeated_frames_without_overwriting_evidence(self):
        with TemporaryDirectory() as folder:
            session_type = self.session_class()
            session = session_type(Path(folder), {})
            try:
                with self.assertRaises(ValueError):
                    session.record({"run_id": "different", "sequence_number": 0})
                frame = {"run_id": session.run_id, "sequence_number": 0}
                session.record(frame)
                with self.assertRaises(ValueError):
                    session.record(frame)
                with self.assertRaises(FileExistsError):
                    session_type(Path(folder), {})
            finally:
                session.close("failed", "source disconnected")
            summary = json.loads((Path(folder) / "run.json").read_text())
            self.assertEqual(summary["frames_recorded"], 1)
            self.assertEqual(summary["error"], "source disconnected")


if __name__ == "__main__":
    unittest.main()
