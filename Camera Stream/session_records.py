"""Keep the settings, frame results and MQTT receipts for one stream run."""

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_directory():
    # Give every launch its own folder, including two launches in the same second.
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parent / "diagnostics" / "runs" / f"{name}-{uuid4().hex[:8]}"


class SessionRecords:
    """Write results as they arrive, then finish the run summary on shutdown."""

    def __init__(self, directory, settings):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents = True, exist_ok = True)
        self.run_id = uuid4().hex
        self._lock = Lock()
        self._closed = False
        self._last_sequence = None
        self._write_error = None
        self.summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_at_utc": utc_now(),
            "status": "running",
            "settings": settings,
            "frames_recorded": 0,
            "partial_frames": 0,
            "sequence_gaps": 0,
        }

        # Refuse to reuse an earlier run instead of overwriting its evidence.
        with (self.directory / "run.json").open("x", encoding = "utf-8") as output:
            json.dump(self.summary, output, indent = 2, allow_nan = False)

        self._frames = (self.directory / "frames.jsonl").open("x", encoding = "utf-8")
        try:
            self._receipts = (self.directory / "gcs-logs.jsonl").open("x", encoding = "utf-8")
        except BaseException:
            self._frames.close()
            raise

    def record(self, record):
        # Match each saved frame to this run and require forward sequence progress.
        sequence = record["sequence_number"]
        if (
            record["run_id"] != self.run_id
            or type(sequence) is not int
            or sequence < 0
            or (self._last_sequence is not None and sequence <= self._last_sequence)
        ):
            raise ValueError("Frame does not belong to this run or repeats an earlier sequence")

        with self._lock:
            # Surface a background receipt-write failure in the main stream too.
            if self._write_error is not None:
                raise OSError(f"Could not save MQTT evidence: {self._write_error}")

            self._frames.write(json.dumps(record, allow_nan = False) + "\n")
            self._frames.flush()
            self.summary["frames_recorded"] += 1
            self.summary["partial_frames"] += any(
                result.get("failure_reason")
                for result in record.get("processors", [])
            )

            if self._last_sequence is not None:
                self.summary["sequence_gaps"] += sequence - self._last_sequence - 1

            self._last_sequence = sequence

    def mqtt_event(self, event):
        # MQTT callbacks run on another thread, so keep receipt writes together.
        with self._lock:
            if self._closed:
                return

            try:
                receipt = {"recorded_at_utc": utc_now(), **event}
                self._receipts.write(json.dumps(receipt, allow_nan = False) + "\n")
                self._receipts.flush()
            except (OSError, ValueError, TypeError) as error:
                self._write_error = error

    def close(self, status = "stopped", error = None):
        with self._lock:
            if self._closed:
                return

            self._closed = True
            self._frames.close()
            self._receipts.close()
            self.summary.update(
                status = "failed" if self._write_error is not None else status,
                error = str(self._write_error) if self._write_error else error,
                finished_at_utc = utc_now(),
            )

            # Replace the summary only after the new contents are fully written.
            temporary = self.directory / "run.json.tmp"
            temporary.write_text(
                json.dumps(self.summary, indent = 2, allow_nan = False),
                encoding = "utf-8",
            )
            temporary.replace(self.directory / "run.json")
