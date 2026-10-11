"""Save each frame's results and the run status without storing video frames."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ArUcoRunRecorder:
    def __init__(self, output_dir: Path, metadata: dict):
        # Open a new run directory and frame log, then initialise the run's counters.
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._file = (self.output_dir / "frames.jsonl").open("x", encoding="utf-8")
        self.counts = dict(
            frames=0,
            empty_frames=0,
            detections=0,
            accepted_poses=0,
            rejected_poses=0,
        )
        self._failures = Counter()
        self._processing_sum = 0.0
        self._processing_max = 0.0
        self._manifest = dict(
            schema_version=1,
            status="running",
            started_at_utc=utc_now(),
            metadata=metadata,
        )

        try:
            self._save_manifest()

        except BaseException:
            # Close the frame log if the initial run status could not be saved.
            self._file.close()
            raise

    def _save_manifest(self):
        # Replace the status file only after the new JSON is fully written.
        temporary = self.output_dir / "run.json.tmp"

        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self._manifest, file, indent=2, allow_nan=False)
            file.write("\n")

        temporary.replace(self.output_dir / "run.json")

    def update_metadata(self, **values):
        # Merge newly available details, such as the camera ID, into the saved status.
        self._manifest["metadata"].update(values)
        self._save_manifest()

    def write(self, record):
        # Append this frame as one JSON line and add its measurements to the run totals.
        self._file.write(json.dumps(record, allow_nan=False) + "\n")
        # Flush this record from the file buffer before processing the next frame.
        self._file.flush()

        # Count frames and accepted/rejected poses, then tally each recorded failure reason.
        results = record["results"]
        self.counts["frames"] += 1
        self.counts["empty_frames"] += int(not results)
        self.counts["detections"] += len(results)
        self.counts["accepted_poses"] += sum(r["valid"] for r in results)
        self.counts["rejected_poses"] += sum(not r["valid"] for r in results)
        self._failures.update(
            r["failure_reason"] for r in results if r["failure_reason"]
        )

        # Accumulate processing time for the final mean and longest-frame measurement.
        elapsed = record["processing_time_ms"]
        self._processing_sum += elapsed
        self._processing_max = max(self._processing_max, elapsed)

    def finish(self, *, status, stop_reason, stream_summary=None, error=None):
        # Close the frame log and save the run's outcome, counts and timing summary.
        try:
            self._file.close()

        finally:
            # Record how the run ended even if closing the frame log failed.
            n = self.counts["frames"]
            self._manifest.update(
                status=status,
                stop_reason=stop_reason,
                finished_at_utc=utc_now(),
                counts=self.counts,
                failure_reasons=dict(self._failures),
                processing_time_ms=dict(
                    mean=self._processing_sum / n if n else None,
                    maximum=self._processing_max if n else None,
                ),
                stream_summary=stream_summary,
                error=error,
            )
            self._save_manifest()
