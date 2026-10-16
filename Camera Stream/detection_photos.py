"""Select same-frame photo pairs and save/send them in one bounded worker.

The camera only selects observations and copies selected frames. The worker saves
original PNGs and full-resolution annotated JPEGs, then optionally publishes the
saved JPEG. A failed connection retains the event as deferred; future captures
get their own connection attempt. Nothing is automatically replayed, including
an uncertain send. Existing folders are refused. PUBACK confirms the broker only.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from gcs_output import gcs_detection_payload, gcs_log_payloads


MAX_PAYLOAD_BYTES = 12_000_000
MAX_RECEIPT_BYTES = 4096


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value):
    return (
        json.dumps(value, allow_nan = False, separators = (",", ":")) + "\n"
    ).encode()


def _write_new(path, data):
    """Expose a completed file without replacing existing evidence."""
    temporary = path.with_name("." + path.name + ".tmp")
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            output.write(data)
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok = True)


@dataclass
class _Sighting:
    visible_since: float | None = None
    absent_since: float | None = None
    handled: bool = False
    last_attempt: float | None = None


class PhotoCapture:
    """Own selection, a bounded queue, and one save/upload worker for one run."""

    def __init__(
        self,
        directory,
        *,
        broker = None,
        port = 1883,
        interval_s = 3,
        max_pending = 3,
        max_storage_bytes = 256 * 1024 * 1024,
        max_mbps = 1,
        publish_timeout_s = 60,
    ):
        for name, value in (
            ("interval_s", interval_s),
            ("max_mbps", max_mbps),
            ("publish_timeout_s", publish_timeout_s),
        ):
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if interval_s < 3:
            raise ValueError("interval_s must be at least three seconds")
        for name, value in (
            ("max_pending", max_pending),
            ("max_storage_bytes", max_storage_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if broker is not None and (
            not isinstance(broker, str)
            or not broker.strip()
        ):
            raise ValueError("broker must be a hostname or IP address")

        self.directory = Path(directory)
        self.directory.mkdir(parents = True, exist_ok = False)
        self.broker, self.port = broker, port
        self.interval_s, self.max_mbps = interval_s, max_mbps
        self.max_storage_bytes = max_storage_bytes
        self.publish_timeout_s = publish_timeout_s
        self.policy = dict(
            stable_s = 0.3,
            absence_s = 2,
            cooldown_s = 10,
            interval_s = interval_s,
            max_pending = max_pending,
            max_storage_bytes = max_storage_bytes,
            max_mbps = max_mbps,
            max_payload_bytes = MAX_PAYLOAD_BYTES,
            publish_timeout_s = publish_timeout_s,
            storage_scope = (
                "Images, metadata and receipts. At most max_pending additional "
                "receipts of 4096 bytes each and one summary may exceed the limit."
            ),
        )
        self.stats = dict.fromkeys((
            "accepted", "captured", "invalid_frames", "skipped_queue", "skipped_storage",
            "failed", "preview", "deferred", "unconfirmed", "broker_acknowledged",
            "storage_bytes", "receipt_failures",
        ), 0)
        self._lock = Lock()
        self._stop = Event()
        self._storage_full = Event()
        self._jobs = Queue(maxsize = max_pending)
        # The active job consumes a slot too, so a slow ACK cannot grow memory usage.
        self._slots = BoundedSemaphore(max_pending)
        self._states = {}
        self._targets_since = self._last_target_attempt = None
        self._run_id = None
        self._last_time = None
        self._last_rejection = None
        self._next_start = 0.0
        self._closed = False
        self._worker = Thread(
            target = self._work,
            name = "detection-photos",
            daemon = True,
        )
        self._worker.start()

    def _reject(self, reason, event_id, error = None):
        with self._lock:
            self.stats[reason] += 1
            self._last_rejection = dict(
                event_id = event_id,
                reason = reason,
                error = error,
            )
        print(f"Photo {event_id}: {reason}" + (f" ({error})" if error else ""))

    def _validate(self, packet, annotated, record):
        if record.get("producer") != "camera_stream":
            raise ValueError("Photos require a native camera_stream record")
        run_id = record.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or self._run_id not in (None, run_id)
        ):
            raise ValueError("Photo record must belong to this run")
        if (
            not isinstance(record.get("source_kind"), str)
            or not record["source_kind"]
        ):
            raise ValueError("Photo record requires a source_kind")
        if np.shares_memory(packet.image, annotated.image):
            raise ValueError("Annotated pixels must be separate from original evidence")
        for candidate in (packet, annotated):
            if (
                type(candidate.sequence_number) is not int
                or candidate.sequence_number < 0
                or candidate.sequence_number != record.get("sequence_number")
                or candidate.device_timestamp_s != record.get("device_timestamp_s")
                or candidate.host_received_timestamp_s
                != record.get("host_received_timestamp_s")
                or list(record.get("image_size_px", []))
                != list(candidate.image.shape[1::-1])
                or candidate.image.shape != packet.image.shape
                or candidate.image.dtype != packet.image.dtype
            ):
                raise ValueError("Photo, annotation and record identities must match")
        now = packet.host_received_timestamp_s
        if (
            not isfinite(now)
            or (self._last_time is not None and now < self._last_time)
        ):
            raise ValueError("Photo selection needs finite, ordered host timestamps")
        self._run_id, self._last_time = run_id, now

    def _select(self, record, now):
        processors = {
            item["processor"]: item
            for item in record.get("processors", [])
        }
        aruco = processors.get("aruco")
        selected = []
        if aruco is None or aruco.get("failure_reason") is not None:
            # Failed processing cannot prove that a previously seen marker disappeared.
            for state in self._states.values():
                state.visible_since = state.absent_since = None
        else:
            data = aruco.get("data", {})
            if data.get("config", {}).get("dictionary_name") != "DICT_5X5_100":
                raise ValueError("Photo policy requires DICT_5X5_100")
            observed = {item["marker_id"] for item in data.get("results", [])}
            if any(type(item) is not int or not 0 <= item < 100 for item in observed):
                raise ValueError("Decoded marker IDs must be from 0 through 99")
            for marker, state in self._states.items():
                if marker not in observed:
                    state.visible_since = None
                    if state.absent_since is None:
                        state.absent_since = now
            for marker in sorted(observed):
                state = self._states.setdefault(marker, _Sighting())
                if state.absent_since is not None:
                    # Only an observed two-second absence starts another sighting.
                    if now - state.absent_since >= 2:
                        state.handled = False
                    state.absent_since = None
                if state.visible_since is None:
                    state.visible_since = now
                if (
                    not state.handled
                    and now - state.visible_since >= 0.3
                    and (state.last_attempt is None or now - state.last_attempt >= 10)
                ):
                    selected.append(marker)
                    # Keep the existing once-per-sighting policy even on a queue skip.
                    state.handled, state.last_attempt = True, now

        targets = processors.get("targets", {})
        results = []
        if targets.get("failure_reason") is None:
            # A gauge with unavailable pressure remains a visible photo target.
            for item in targets.get("data", {}).get("results", []):
                if item.get("class_name") in ("gauge", "openValve", "closedValve"):
                    results.append(item)
        if not results:
            self._targets_since = None
        else:
            if self._targets_since is None:
                self._targets_since = now
            interval_elapsed = (
                self._last_target_attempt is None
                or now - self._last_target_attempt >= self.interval_s
            )
            if selected or (
                now - self._targets_since >= 0.3
                and interval_elapsed
            ):
                # Pace failed attempts too; a marker photo can share these targets.
                self._last_target_attempt = now
                return selected, results
        return selected, []

    def submit(self, packet, annotated_packet, record):
        """Offer this frame without waiting for disk, a queue slot, or the network."""
        if self._closed:
            return False
        event_id = f'{record.get("run_id")}:{packet.sequence_number}'
        try:
            self._validate(packet, annotated_packet, record)
            markers, targets = self._select(record, packet.host_received_timestamp_s)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            self._reject("invalid_frames", event_id, str(error))
            return False
        if not markers and not targets:
            return False
        if self._storage_full.is_set():
            self._reject("skipped_storage", event_id)
            return False
        if not self._slots.acquire(blocking = False):
            self._reject("skipped_queue", event_id)
            return False
        try:
            original, annotated = packet.image.copy(), annotated_packet.image.copy()
            original.flags.writeable = annotated.flags.writeable = False
            job = dict(
                original = original,
                annotated = annotated,
                record = deepcopy(record),
                marker_ids = markers,
                target_results = deepcopy(targets),
                event_id = event_id,
                observed_at_utc = _utc_now(),
            )
            self._jobs.put_nowait(job)
            with self._lock:
                self.stats["accepted"] += 1
            return True
        except Exception:
            self._slots.release()
            raise

    def _save(self, job):
        record = job["record"]
        sequence = record["sequence_number"]
        images = []
        for extension, pixels, options in (
            (".png", job["original"], []),
            (".annotated.jpg", job["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 90]),
        ):
            encoding = ".png" if extension == ".png" else ".jpg"
            success, encoded = cv2.imencode(encoding, pixels, options)
            if not success:
                raise ValueError(f"Photo {extension} encoding failed")
            data = encoded.tobytes()
            identity = dict(
                image_file = f"frame-{sequence}{extension}",
                image_bytes = len(data),
                image_sha256 = hashlib.sha256(data).hexdigest(),
            )
            images.append((identity, data))
        metadata = dict(
            schema_version = 1,
            event_id = job["event_id"],
            run_id = record["run_id"],
            sequence_number = sequence,
            source_kind = record["source_kind"],
            marker_ids = job["marker_ids"],
            target_results = job["target_results"],
            observed_at_utc = job["observed_at_utc"],
            timestamp_basis = "host_selection_time_not_camera_exposure",
            **images[0][0],
            annotated_image = images[1][0],
            record = record,
        )
        sidecar = _json_bytes(metadata)
        size = sum(len(data) for _, data in images) + len(sidecar)
        if self.stats["storage_bytes"] + size > self.max_storage_bytes:
            self._storage_full.set()
            return None
        for identity, data in images:
            _write_new(self.directory / identity["image_file"], data)
            self.stats["storage_bytes"] += len(data)
        # The event becomes visible only when both image files are complete.
        _write_new(self.directory / f"frame-{sequence}.event.json", sidecar)
        self.stats["storage_bytes"] += len(sidecar)
        self.stats["captured"] += 1
        return metadata

    def _payload(self, metadata):
        selected = []
        labels = set()
        for processor in metadata["record"]["processors"]:
            if processor["processor"] == "aruco" and metadata["marker_ids"]:
                results = [
                    item for item in processor["data"]["results"]
                    if item["marker_id"] in metadata["marker_ids"]
                ]
                labels.add("ArUco")
            elif (
                processor["processor"] == "targets"
                and metadata["target_results"]
            ):
                results = metadata["target_results"]
                for item in results:
                    if item["class_name"] == "gauge":
                        labels.add("Gauge")
                    else:
                        labels.add("Valve")
            else:
                continue

            # Reuse the readable log wording for only this photo's observations.
            selected.append({
                **processor,
                "data": {**processor["data"], "results": results},
            })
        messages = gcs_log_payloads({**metadata["record"], "processors": selected})
        target_type = ", ".join(
            label for label in ("ArUco", "Gauge", "Valve")
            if label in labels
        )
        description = "\n".join(message["content"] for message in messages)
        image = metadata["annotated_image"]
        # Bound Base64 growth before asking the shared builder to read the saved JPEG.
        if image["image_bytes"] * 4 / 3 > MAX_PAYLOAD_BYTES:
            raise ValueError("Photo exceeds the 12 MB encoded payload limit")
        payload = gcs_detection_payload(
            self.directory / image["image_file"],
            target_type,
            description,
            timestamp = metadata["observed_at_utc"],
        )
        if len(_json_bytes(payload)) > MAX_PAYLOAD_BYTES:
            raise ValueError("Photo exceeds the 12 MB encoded payload limit")
        return payload

    def _publish(self, payload):
        encoded = _json_bytes(payload)
        if self._stop.wait(max(0, self._next_start - monotonic())):
            return dict(delivery = "deferred", reason = "STOPPED_BEFORE_PUBLISH")
        client = mqtt.Client(
            callback_api_version = mqtt.CallbackAPIVersion.VERSION2,
            protocol = mqtt.MQTTv311,
            reconnect_on_failure = False,
        )
        client.connect_timeout = 3
        client.max_queued_messages_set(1)
        connected = Event()
        outcome = {"accepted": False}

        def on_connect(client, userdata, flags, reason_code, properties):
            outcome["accepted"] = not reason_code.is_failure
            connected.set()

        client.on_connect = on_connect
        client.on_connect_fail = lambda client, userdata: connected.set()
        attempted = False
        try:
            client.connect_async(self.broker, self.port, keepalive = 30)
            if client.loop_start() != mqtt.MQTT_ERR_SUCCESS:
                raise ConnectionError("Photo MQTT loop did not start")
            deadline = monotonic() + 3
            while (
                not connected.is_set()
                and not self._stop.is_set()
                and monotonic() < deadline
            ):
                self._stop.wait(0.05)
            if self._stop.is_set() or not outcome["accepted"]:
                # Retain unsent evidence; a later new capture can try the broker again.
                return dict(
                    delivery = "deferred",
                    reason = "BROKER_UNAVAILABLE_OR_STOPPED",
                )
            attempted = True
            started = monotonic()
            info = client.publish("gcs/detection", encoded, qos = 1, retain = False)
            # Pace packet starts using the actual JSON bytes, including Base64.
            # A packet may burst; MQTT/TCP overhead is additional to this budget.
            self._next_start = started + max(
                self.interval_s,
                len(encoded) * 8 / (self.max_mbps * 1_000_000),
            )
            deadline = started + self.publish_timeout_s
            while info.rc == mqtt.MQTT_ERR_SUCCESS:
                if info.is_published():
                    return dict(
                        delivery = "broker_acknowledged",
                        acknowledged_at_utc = _utc_now(),
                        acknowledgement_scope = (
                            "Broker PUBACK only; GCS database ingestion is not confirmed."
                        ),
                    )
                if self._stop.is_set() or monotonic() >= deadline:
                    break
                self._stop.wait(min(0.05, max(0, deadline - monotonic())))
            return dict(
                delivery = "unconfirmed",
                acknowledgement_scope = (
                    "No PUBACK observed; delivery is uncertain. No automatic replay."
                ),
            )
        except Exception as error:
            scope = "No image handed to MQTT."
            if attempted:
                scope = "Delivery uncertain; no automatic replay."
            return dict(
                delivery = "unconfirmed" if attempted else "deferred",
                error = f"{type(error).__name__}: {error}",
                acknowledgement_scope = scope,
            )
        finally:
            try:
                try:
                    client.disconnect()
                finally:
                    client.loop_stop()
            except Exception as error:
                # Cleanup cannot change a PUBACK already observed into a failed send.
                print(f"Photo MQTT cleanup error: {error}")

    def _work(self):
        while not self._stop.is_set() or not self._jobs.empty():
            try:
                job = self._jobs.get(timeout = 0.1)
            except Empty:
                continue
            sequence = job["record"]["sequence_number"]
            receipt = dict(event_id = job["event_id"], topic = "gcs/detection")
            try:
                metadata = self._save(job)
                if metadata is None:
                    receipt.update(
                        delivery = "skipped_storage",
                        reason = "STORAGE_LIMIT",
                    )
                else:
                    receipt.update(metadata["annotated_image"])
                    payload = self._payload(metadata)
                    receipt.update({
                        key: value for key, value in payload.items()
                        if key != "image"
                    })
                    receipt["payload_wire_bytes"] = len(_json_bytes(payload))
                    if self._stop.is_set():
                        receipt.update(
                            delivery = "deferred",
                            reason = "STOPPED_BEFORE_PUBLISH",
                        )
                    elif self.broker is None:
                        receipt["delivery"] = "preview"
                    else:
                        receipt.update(self._publish(payload))
            except Exception as error:
                receipt.update(
                    delivery = "failed",
                    error = f"{type(error).__name__}: {error}",
                )
            try:
                encoded = _json_bytes(receipt)
                if len(encoded) > MAX_RECEIPT_BYTES:
                    # The complete caption can be regenerated from the saved event.
                    receipt.pop("description", None)
                    receipt["description_omitted"] = (
                        "See the saved event for full observations."
                    )
                    encoded = _json_bytes(receipt)
                if len(encoded) > MAX_RECEIPT_BYTES:
                    raise ValueError("Photo receipt exceeds its bounded storage allowance")
                if (
                    self.stats["storage_bytes"] + len(encoded)
                    >= self.max_storage_bytes
                ):
                    self._storage_full.set()
                _write_new(self.directory / f"frame-{sequence}.delivery.json", encoded)
                self.stats["storage_bytes"] += len(encoded)
            except Exception as error:
                self.stats["receipt_failures"] += 1
                print(f"Photo {job['event_id']}: receipt failed ({error})")
            finally:
                with self._lock:
                    self.stats[receipt["delivery"]] += 1
                print(f"Photo {job['event_id']}: {receipt['delivery']}")
                self._jobs.task_done()
                self._slots.release()

    def close(self):
        """Interrupt network waits and allow queued selections to finish saving."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._worker.join(timeout = 5)
        with self._lock:
            summary = dict(
                run_id = self._run_id,
                policy = self.policy,
                stats = dict(self.stats),
                last_rejection = self._last_rejection,
                worker_stopped = not self._worker.is_alive(),
                accepted_unfinished = self._jobs.unfinished_tasks,
            )
        try:
            _write_new(self.directory / "summary.json", _json_bytes(summary))
        except OSError as error:
            print(f"Photo summary could not be saved: {error}")
        if self._worker.is_alive():
            print(
                "Photo shutdown timed out; accepted unfinished work "
                "is recorded in summary.json."
            )
