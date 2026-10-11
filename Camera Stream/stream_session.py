"""Save frame records as they arrive and keep a summary of the run."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class StreamSession:
    def __init__(self, directory, metadata):
        # Create the run folder, frame log and initial run summary.
        self.directory = Path(directory)
        # Reject a folder that already exists instead of replacing its records.
        self.directory.mkdir(parents=True, exist_ok=False)
        # Start the summary with a new run ID and zero recorded frames.
        self.manifest = dict(
            schema_version=1,
            run_id=uuid4().hex,
            status="running",
            started_at_utc=utc_now(),
            metadata=metadata,
            frames_recorded=0,
            partial_frames=0,
        )
        self._timing_sums = {}
        self._frames = (self.directory / "frames.jsonl").open("x", encoding="utf-8")

        try:
            self._save()

        except BaseException:
            # Close the frame log if writing the initial summary fails.
            self._frames.close()
            raise

    def _save(self):
        # Write the summary to a temporary file, then replace run.json with it.
        temporary = self.directory / "run.json.tmp"
        temporary.write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.directory / "run.json")

    def update_metadata(self, **values):
        # Merge the supplied run details into the summary and save it again.
        self.manifest["metadata"].update(values)
        self._save()

    def write(self, record):
        # Add this run's ID and append the frame record as one JSON line.
        record = dict(record, run_id=self.manifest["run_id"])
        self._frames.write(json.dumps(record, allow_nan=False) + "\n")
        # Flush the line so another process can read this frame immediately.
        self._frames.flush()

        # Count this frame and whether any of its processors failed.
        self.manifest["frames_recorded"] += 1
        self.manifest["partial_frames"] += record["status"] == "partial"

        # Add each stage's duration to the total used for its final average.
        for stage, duration in record["timings_ms"].items():
            self._timing_sums[stage] = self._timing_sums.get(stage, 0.0) + duration

    def finish(self, *, status, stop_reason, summary=None, error=None):
        # Close the frame log and save the outcome and average stage durations.
        try:
            self._frames.close()

        finally:
            # Save the final summary even if closing the frame log failed.
            count = self.manifest["frames_recorded"]
            self.manifest.update(
                status=status,
                stop_reason=stop_reason,
                finished_at_utc=utc_now(),
                summary=summary,
                error=error,
                mean_timings_ms=(
                    # Divide each accumulated duration by the recorded frame count.
                    {key: value / count for key, value in self._timing_sums.items()}
                    if count
                    else {}
                ),
            )
            self._save()
