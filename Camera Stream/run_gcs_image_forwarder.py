"""Preview or upload completed detection captures to gcs/detection.

One synchronous upload is in flight at a time, outside the camera process.
The rate budget covers actual UTF-8 JSON bytes, including Base64. It spaces
packet starts; one MQTT packet can burst and MQTT/TCP overhead is additional.
Run one uploader per capture directory. Source images and sidecars are retained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from uuid import uuid4

from gcs_output import gcs_detection_payload, gcs_log_payloads
from stream_mqtt import MqttResultPublisher, PublishNotStartedError, positive_timeout

TOPIC = "gcs/detection"
MAX_SIDECAR_BYTES = 1024 * 1024
TIMESTAMP_BASIS = "host_selection_time_not_camera_exposure"


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _encoded_json(value):
    # Encode the record as the same compact UTF-8 JSON bytes MQTT will send.
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _read_bounded(path, limit):
    # Read at most limit + 1 bytes and reject missing or oversized capture files.
    if not path.is_file():
        raise ValueError(f"Missing regular file: {path.name}")

    if path.stat().st_size > limit:
        raise ValueError(f"{path.name} exceeds the permitted byte limit ({limit})")

    with path.open("rb") as source:
        data = source.read(limit + 1)

    if len(data) > limit:
        raise ValueError(f"{path.name} exceeds the permitted byte limit ({limit})")

    return data


def _reject_constant(value):
    # Reject NaN and infinity tokens while parsing the capture's JSON record.
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _write_receipt(path, entry):
    # Write the delivery outcome to a temporary file, then replace its final receipt.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        with temporary.open("xb") as output:
            output.write(_encoded_json(entry) + b"\n")

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)


def _photo_summary(record, marker_ids, target_results):
    """Describe only the detections recorded alongside this original photo."""
    selected_processors = []
    categories = []

    if marker_ids:
        processor = next(
            (
                p
                for p in record.get("processors", [])
                if p.get("processor") == "aruco"
            ),
            None,
        )
        results = (
            []
            if processor is None
            else processor.get("data", {}).get("results", [])
        )
        selected = [
            result
            for result in results
            if result.get("marker_id") in marker_ids
        ]

        # Check that every photographed marker was actually decoded in this frame.
        if (
            processor is None
            or processor.get("failure_reason") is not None
            or set(marker_ids) != {result.get("marker_id") for result in selected}
        ):
            raise ValueError("Capture marker IDs do not match the frame's observations")

        selected_processors.append(
            {
                **processor,
                "data": {**processor["data"], "results": selected},
            }
        )
        categories.append("ArUco")

    if target_results:
        processor = next(
            (
                p
                for p in record.get("processors", [])
                if p.get("processor") == "targets"
            ),
            None,
        )

        if (
            processor is None
            or processor.get("failure_reason") is not None
        ):
            raise ValueError(
                "Capture targets do not match a successful target processor"
            )

        remaining = list(processor.get("data", {}).get("results", []))
        labels = {"gauge": "Gauge", "openValve": "Valve", "closedValve": "Valve"}

        for target in target_results:
            # Match each selected target once, including its box and pressure status.
            if (
                not isinstance(target, dict)
                or target.get("class_name") not in labels
                or target not in remaining
            ):
                raise ValueError("Capture target results do not match the saved frame")

            remaining.remove(target)

        selected_processors.append(
            {
                **processor,
                "data": {**processor["data"], "results": target_results},
            }
        )
        present = {labels[target["class_name"]] for target in target_results}
        categories.extend(label for label in ("Gauge", "Valve") if label in present)

    # Use the log wording for accepted pressure and unavailable readings in photos too.
    selected_record = {**record, "processors": selected_processors}
    messages = gcs_log_payloads(selected_record)

    if not messages:
        raise ValueError("Capture record has no observations to describe")

    return ", ".join(categories), "\n".join(message["content"] for message in messages)


def _prepare(sidecar, capture_dir, max_payload_bytes):
    # Validate a photo and its companion event record, then build the GCS payload.
    # Reject an event file whose resolved location is outside this capture directory.
    if sidecar.resolve(strict=True).parent != capture_dir:
        raise ValueError("Capture sidecar escapes the capture directory")

    # Read the event JSON within the size limit and reject nonstandard number values.
    metadata = json.loads(
        _read_bounded(sidecar, MAX_SIDECAR_BYTES), parse_constant=_reject_constant
    )

    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Capture sidecar must be a schema_version 1 object")

    run_id, sequence = metadata.get("run_id"), metadata.get("sequence_number")

    # Require an identified run and a nonnegative whole-number frame sequence.
    if (
        not isinstance(run_id, str)
        or not run_id
        or type(sequence) is not int
        or sequence < 0
    ):
        raise ValueError("Capture requires a run_id and nonnegative frame sequence")

    if metadata.get("event_id") != f"{run_id}:{sequence}":
        raise ValueError("Capture event_id does not match its frame identity")

    # Require the timestamp to be labelled as host selection time, not exposure time.
    if metadata.get("timestamp_basis") != TIMESTAMP_BASIS:
        raise ValueError(
            "Capture timestamp must identify host selection time, not camera exposure"
        )

    timestamp = metadata.get("observed_at_utc")

    # Reject a missing capture time or one that cannot be parsed with a timezone.
    if not isinstance(timestamp, str):
        raise ValueError(
            "Capture timestamp must be an explicit timezone-aware ISO 8601 string"
        )

    try:
        timestamp_value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    except ValueError as error:
        raise ValueError(
            "Capture timestamp must be an explicit timezone-aware ISO 8601 string"
        ) from error

    if timestamp_value.utcoffset() is None:
        raise ValueError("Capture timestamp must include a timezone")

    marker_ids = metadata.get("marker_ids")
    target_results = metadata.get("target_results", [])

    # Old ArUco captures have no target_results; new photos can contain either branch.
    if (
        not isinstance(marker_ids, list)
        or any(type(item) is not int or not 0 <= item < 100 for item in marker_ids)
        or not isinstance(target_results, list)
        or not (marker_ids or target_results)
    ):
        raise ValueError(
            "Capture requires marker IDs from 0 to 99 or target results"
        )

    record = metadata.get("record")

    # Reject detection records whose run, frame number or source differs from the photo.
    if (
        not isinstance(record, dict)
        or record.get("producer") != "camera_stream"
        or record.get("run_id") != run_id
        or record.get("sequence_number") != sequence
        or record.get("source_kind") != metadata.get("source_kind")
    ):
        raise ValueError("Capture record does not match its frame identity and source")

    # Prefer the saved annotation when present; invalid annotations must not use the original.
    selected_image = metadata.get("annotated_image", metadata)

    if not isinstance(selected_image, dict):
        raise ValueError("Capture annotated_image must be an object")

    image_name = selected_image.get("image_file")

    # Require an image filename with no directories, slashes or backslashes.
    if (
        not isinstance(image_name, str)
        or not image_name
        or Path(image_name).name != image_name
        or "/" in image_name
        or "\\" in image_name
    ):
        raise ValueError(
            "Capture image_file must be a relative basename inside the capture directory"
        )

    image = (capture_dir / image_name).resolve(strict=True)

    # Resolve file links and reject an image located outside the capture directory.
    if image.parent != capture_dir:
        raise ValueError("Capture image escapes the capture directory")

    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        image.suffix.lower()
    )

    if mime is None:
        raise ValueError("Capture image must be PNG or JPEG")

    byte_count = selected_image.get("image_bytes")
    digest = selected_image.get("image_sha256")

    # Require a positive byte count and a 64-character SHA-256 file fingerprint.
    if (
        type(byte_count) is not int
        or byte_count <= 0
        or not isinstance(digest, str)
        or re.fullmatch("[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("Capture requires an image byte count and SHA-256 digest")

    # Describe all selected detections while keeping one image for this frame.
    target_type, description = _photo_summary(record, marker_ids, target_results)
    empty_payload = dict(
        type=target_type,
        description=description,
        mime=mime,
        image="",
        timestamp=metadata["observed_at_utc"],
    )
    # Estimate total JSON size: fixed fields plus the image's Base64 text.
    expected_size = len(_encoded_json(empty_payload)) + 4 * ((byte_count + 2) // 3)

    if expected_size > max_payload_bytes:
        raise ValueError(
            f"Encoded detection exceeds the payload limit ({expected_size} > {max_payload_bytes} bytes)"
        )

    # Read the image within the allowed size, then compare its bytes and fingerprint.
    selected_bytes = _read_bounded(image, (max_payload_bytes // 4) * 3)

    if len(selected_bytes) != byte_count:
        raise ValueError("Capture image byte count does not match its sidecar")

    if hashlib.sha256(selected_bytes).hexdigest() != digest:
        raise ValueError("Capture image SHA-256 does not match its sidecar")

    with TemporaryDirectory(prefix="gcs-upload-image-") as staging:
        # Write the verified bytes to a private copy and build the payload from it.
        snapshot = Path(staging) / image.name
        snapshot.write_bytes(selected_bytes)
        del selected_bytes
        payload = gcs_detection_payload(
            snapshot, target_type, description, timestamp=metadata["observed_at_utc"]
        )

    wire_bytes = len(_encoded_json(payload))

    # Reject the payload if its actual encoded size exceeds the upload limit.
    if wire_bytes > max_payload_bytes:
        raise ValueError(
            f"Encoded detection exceeds the payload limit ({wire_bytes} > {max_payload_bytes} bytes)"
        )

    return metadata, payload, wire_bytes


class ImageForwarder:
    """Upload one ready photo at a time and save the outcome beside it."""

    def __init__(
        self,
        capture_dir,
        *,
        publish=False,
        broker=None,
        port=1883,
        interval_s=10.0,
        max_mbps=1.0,
        max_payload_mb=12.0,
        publish_timeout_s=60.0,
        clock=monotonic,
        publisher_factory=MqttResultPublisher,
    ):
        self.capture_dir = Path(capture_dir).resolve(strict=True)

        if not self.capture_dir.is_dir():
            raise ValueError("capture-dir must be an existing directory")

        self.interval_s = positive_timeout(interval_s)

        # Allow the three-second photo cadence while retaining a minimum upload gap.
        if self.interval_s < 3.0:
            raise ValueError("interval-s must be at least 3 seconds")

        self.max_mbps = positive_timeout(max_mbps)
        self.publish_timeout_s = positive_timeout(publish_timeout_s)
        self.max_payload_bytes = int(positive_timeout(max_payload_mb) * 1_000_000)

        if self.max_payload_bytes < 1:
            raise ValueError("max-payload-mb must permit at least one byte")

        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be an integer from 1 to 65535")

        if publish and (not isinstance(broker, str) or not broker):
            raise ValueError("publish requires a broker hostname/IP")

        self.publish, self.broker, self.port = publish, broker, port
        self.clock, self.publisher_factory = clock, publisher_factory
        self._publisher = None
        self._next_start = float("-inf")
        self._closed = False
        self.failures = 0

    def _next_sidecar(self):
        # Search event files from oldest to newest for the next unhandled capture.
        for sidecar in sorted(
            self.capture_dir.glob("*.event.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        ):
            if (
                sidecar.with_suffix(".sent.json").exists()
                or sidecar.with_suffix(".failed.json").exists()
            ):
                # Skip photos already acknowledged or recorded as failed uploads.
                continue

            if not self.publish and sidecar.with_suffix(".preview.json").exists():
                # In preview mode, skip photos that already have a saved preview.
                continue

            return sidecar

        return None

    def queue_status(self):
        """Count waiting, completed and failed uploads without sending anything."""
        status = dict(
            pending=0,
            eligible=0,
            sent=0,
            failed=0,
            retrying=0,
            next_attempt_in_s=max(0.0, self._next_start - self.clock()),
        )

        for sidecar in self.capture_dir.glob("*.event.json"):
            # Count each capture as sent, failed or pending from its receipt.
            if sidecar.with_suffix(".sent.json").exists():
                status["sent"] += 1

            elif sidecar.with_suffix(".failed.json").exists():
                status["failed"] += 1

            else:
                status["pending"] += 1

                if self.publish or not sidecar.with_suffix(".preview.json").exists():
                    status["eligible"] += 1

                if sidecar.with_suffix(".retry.json").exists():
                    status["retrying"] += 1

        return status

    def step(self):
        # Handle at most one queued photo, then return its saved delivery outcome.
        if self._closed:
            raise RuntimeError("Image forwarder is closed")

        # Return without selecting a photo until the next permitted send time.
        if self.clock() < self._next_start:
            return None

        sidecar = self._next_sidecar()

        if sidecar is None:
            return None

        started = self.clock()
        entry = dict(
            topic=TOPIC,
            sidecar=sidecar.name,
            prepared_at_utc=_utc_now(),
            delivery="preview",
            publish_attempted=False,
        )
        wire_bytes = 0
        interrupted = None
        connecting = False
        retry_pending = False

        try:
            # Validate the saved evidence and prepare JSON before connecting.
            metadata, payload, wire_bytes = _prepare(
                sidecar, self.capture_dir, self.max_payload_bytes
            )
            # Record the identity of the uploaded file while leaving original evidence intact.
            selected_image = metadata.get("annotated_image", metadata)
            entry.update(
                event_id=metadata["event_id"],
                run_id=metadata["run_id"],
                sequence_number=metadata["sequence_number"],
                image_file=selected_image["image_file"],
                image_bytes=selected_image["image_bytes"],
                image_sha256=selected_image["image_sha256"],
                payload_wire_bytes=wire_bytes,
            )

            if self.publish:
                if self._publisher is None:
                    # Open the broker connection now that a validated image is ready.
                    connecting = True
                    self._publisher = self.publisher_factory(
                        host=self.broker,
                        port=self.port,
                        topic=TOPIC,
                        allow_remote=True,
                        publish_timeout_s=self.publish_timeout_s,
                    )
                    connecting = False

                # Measure upload spacing from when we actually start sending.
                started = self.clock()
                entry["publish_attempted"] = True
                self._publisher.publish(payload)
                # Record the broker ACK and prepare a successful-send receipt.
                entry.update(
                    delivery="broker_acknowledged",
                    acknowledged_at_utc=_utc_now(),
                    acknowledgement_scope="Broker PUBACK only; GCS database ingestion is not confirmed.",
                )
                destination = sidecar.with_suffix(".sent.json")

            else:
                # Save a local preview while leaving the photo available to publish.
                entry["payload"] = payload
                destination = sidecar.with_suffix(".preview.json")

        except (Exception, KeyboardInterrupt) as error:
            if (
                isinstance(error, KeyboardInterrupt)
                and not entry["publish_attempted"]
            ):
                # Stop before sending without marking this photo as handled.
                raise

            interrupted = error if isinstance(error, KeyboardInterrupt) else None
            # Retry only when connection setup failed or publishing never started.
            retry_pending = self.publish and (
                (connecting and isinstance(error, OSError))
                or isinstance(error, PublishNotStartedError)
            )
            entry["error"] = f"{type(error).__name__}: {error}"

            if retry_pending:
                # Save a retry receipt so this photo stays in the pending queue.
                entry.update(
                    delivery="retry_pending",
                    publish_attempted=False,
                    retry_after_s=self.interval_s,
                    acknowledgement_scope="No image handed to MQTT; safe to retry.",
                )
                destination = sidecar.with_suffix(".retry.json")

            else:
                # Record a failure for this mode instead of scheduling a retry.
                self.failures += 1
                entry["delivery"] = "failed" if self.publish else "preview_failed"
                destination = sidecar.with_suffix(
                    ".failed.json" if self.publish else ".preview.json"
                )

            if entry["publish_attempted"]:
                # Mark delivery uncertain if publishing failed without a confirmed ACK.
                entry["acknowledgement_scope"] = (
                    "No PUBACK observed; delivery may be uncertain. No automatic retry."
                )

            if self._publisher is not None and (
                retry_pending or entry["publish_attempted"]
            ):
                # Discard the broken connection before trying the next eligible photo.
                failed_publisher, self._publisher = self._publisher, None

                try:
                    failed_publisher.close()

                except Exception as cleanup_error:
                    entry["cleanup_error"] = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )

        finally:
            if retry_pending:
                # Schedule the retry for one full interval after this failure.
                self._next_start = self.clock() + self.interval_s

            else:
                # Convert message bytes to transmission time at the chosen Mbps budget.
                # Schedule the next send after the larger of that time and the interval.
                self._next_start = started + max(
                    self.interval_s, wire_bytes * 8 / (self.max_mbps * 1_000_000)
                )

        _write_receipt(destination, entry)

        if entry["delivery"] in ("broker_acknowledged", "failed"):
            # Delete the retry receipt after recording a sent or failed outcome.
            sidecar.with_suffix(".retry.json").unlink(missing_ok=True)

        if interrupted is not None:
            raise interrupted

        return entry

    def close(self):
        # Close the broker connection once; ignore later requests to close it again.
        if self._closed:
            return

        self._closed = True

        if self._publisher is not None:
            self._publisher.close()


def build_parser():
    # Define the capture folder, destination, pacing limits and runtime options.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload to GCS; default previews locally only.",
    )
    parser.add_argument(
        "--broker", help="Broker hostname/IP, required only with --publish."
    )
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Runtime in seconds; 0 runs until Ctrl+C (default: 600).",
    )
    parser.add_argument(
        "--publish-timeout-s",
        type=float,
        default=60.0,
        help="Image MQTT acknowledgement wait in seconds (default: 60).",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=10.0,
        help="Minimum seconds between packets; at least 3 (default: 10).",
    )
    parser.add_argument(
        "--max-mbps",
        type=float,
        default=1.0,
        help="Average JSON payload budget in decimal Mbps; packets can burst.",
    )
    parser.add_argument(
        "--max-payload-mb",
        type=float,
        default=12.0,
        help="Maximum encoded JSON payload in decimal MB.",
    )
    return parser


def main(argv=None, *, printer=print):
    # Watch one capture folder and repeatedly process its next eligible photo.
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        # Set no overall deadline for --timeout-s 0; otherwise validate the runtime.
        timeout = None if args.timeout_s == 0 else positive_timeout(args.timeout_s)
        uploader = ImageForwarder(
            args.capture_dir,
            publish=args.publish,
            broker=args.broker,
            port=args.port,
            interval_s=args.interval_s,
            max_mbps=args.max_mbps,
            max_payload_mb=args.max_payload_mb,
            publish_timeout_s=args.publish_timeout_s,
        )

    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    mode = "PUBLISH" if args.publish else "PREVIEW — no GCS connection"
    printer(
        f"{mode}; captures={uploader.capture_dir}; interval={args.interval_s:g}s; "
        f"JSON budget={args.max_mbps:g} Mbps; limit={args.max_payload_mb:g} MB. "
        "A packet can burst. Files are retained.",
        flush=True,
    )

    if args.publish:
        printer(
            f"Destination: {args.broker}:{args.port} topic={TOPIC}; "
            f"image acknowledgement wait={args.publish_timeout_s:g}s.",
            flush=True,
        )

    runtime = "until Ctrl+C" if timeout is None else f"{timeout:g}s"
    printer(
        f"Runtime: {runtime}. Capture folder is fixed; use the new folder for each camera run.",
        flush=True,
    )
    deadline = None if timeout is None else monotonic() + timeout
    next_status = monotonic()

    def report_status(label):
        # Print queue counts and the wait until the next attempt, if one is due.
        status = uploader.queue_status()
        waiting = (
            "; waiting for new captures"
            if not status["eligible"]
            else f'; next attempt in {status["next_attempt_in_s"]:.1f}s'
        )
        printer(
            f'{label}: pending={status["pending"]}, eligible={status["eligible"]}, '
            f'broker-acknowledged={status["sent"]}, failed={status["failed"]}, '
            f'retrying={status["retrying"]}{waiting}.',
            flush=True,
        )
        return status

    try:
        while (
            deadline is None
            or monotonic() < deadline
        ):
            if monotonic() >= next_status:
                # Print queue status every 30 seconds while waiting for photos.
                report_status("Queue")
                next_status = monotonic() + 30.0

            entry = uploader.step()

            if entry is not None:
                # Add the size, retry delay and error only when this outcome has them.
                parts = [
                    f'{entry["delivery"]}: {entry.get("event_id", entry["sidecar"])}'
                ]

                if "payload_wire_bytes" in entry:
                    parts.append(
                        f'{entry["payload_wire_bytes"] / 1_000_000:.2f} MB JSON'
                    )

                if "retry_after_s" in entry:
                    parts.append(f'retry in {entry["retry_after_s"]:g}s')

                if "error" in entry:
                    parts.append(f'{entry["error"]}')

                printer("; ".join(parts), flush=True)

            sleep(
                # Pause briefly, limited by the time remaining before shutdown.
                0.1 if deadline is None else min(0.1, max(0.0, deadline - monotonic()))
            )

        status = report_status("Runtime limit reached; stopped")
        return 1 if uploader.failures or (args.publish and status["retrying"]) else 0

    except KeyboardInterrupt:
        report_status("Stopped by Ctrl+C")
        return 130

    except (OSError, RuntimeError, ValueError, TypeError) as error:
        printer(f"Image forwarder error: {error}", file=sys.stderr, flush=True)
        return 1

    finally:
        uploader.close()


if __name__ == "__main__":
    raise SystemExit(main())
