"""Tests for persistent camera-stream diagnostic records."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stream_diagnostics import StreamSummary
from stream_records import write_complete_run_record
from stream_types import DeviceDetails, StreamProfile


class StreamRecordTests(unittest.TestCase):
    """Protect the integrity of persisted diagnostic evidence."""

    def test_serialization_failure_leaves_no_completed_json(self):
        """Invalid metrics must not leave a partial completed-run file."""

        profile = StreamProfile(
            name="720p30",
            width=1280,
            height=720,
            fps=30.0,
        )
        device = DeviceDetails(
            product_name="OAK-D-LITE",
            mxid="TEST-MXID",
            usb_speed="SUPER",
        )

        # Put an undefined mean delay in the summary to make JSON conversion fail.
        summary = StreamSummary(
            requested_fps=30.0,
            frames_received=100,
            sequence_gaps=0,
            duration_s=3.3,
            received_fps=30.0,
            latency_mean_ms=float("nan"),
            latency_p95_ms=45.0,
            frame_width=1280,
            frame_height=720,
        )

        started_at = datetime(
            2026,
            8,
            24,
            7,
            0,
            tzinfo=timezone.utc,
        )
        finished_at = datetime(
            2026,
            8,
            24,
            7,
            1,
            tzinfo=timezone.utc,
        )

        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "runs"

            # Attempt the save and check that conversion raises before a file is saved.
            with self.assertRaises(ValueError):
                write_complete_run_record(
                    profile=profile,
                    device=device,
                    summary=summary,
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    directory=output_directory,
                )

            completed_records = list(output_directory.glob("*.json"))
            self.assertEqual(completed_records, [])


if __name__ == "__main__":
    unittest.main()
