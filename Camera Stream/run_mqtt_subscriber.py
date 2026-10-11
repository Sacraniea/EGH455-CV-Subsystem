"""Receive local frame measurements and show their identity separately from video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event
from time import monotonic

from stream_mqtt import load_mqtt, positive_timeout, validate_local_endpoint


def _timeout_argument(value):
    # Convert a valid positive timeout or report it as a command-line argument error.
    try:
        return positive_timeout(value)

    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser():
    # Define where to subscribe, where to save records and when to stop receiving.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18883)
    parser.add_argument("--topic", default="taip/local/results")
    parser.add_argument(
        "--output", type=Path, help="New JSONL evidence file (never overwritten)"
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Stop after this many messages; 0 means run until timeout",
    )
    parser.add_argument(
        "--timeout-s",
        type=_timeout_argument,
        default=30.0,
        help="Total finite runtime in seconds (default: 30)",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Create a new readiness marker only after successful SUBACK",
    )
    return parser


def _reject_constant(value):
    # Reject JSON messages containing NaN or infinite numeric values.
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _summary(record):
    # Build one terminal line containing frame identity and each processor's result.
    branches = []

    for result in record.get("processors", []):
        name = result.get("processor", "?")
        status = result.get("failure_reason") or "ok"
        data = json.dumps(
            result.get("data", {}), allow_nan=False, separators=(",", ":")
        )
        # Limit each processor's displayed data to 160 characters in the terminal.
        branches.append(f"{name}={status} {data[:160]}")

    return (
        f"run={record.get('run_id', '?')} seq={record.get('sequence_number', '?')} "
        f"device_s={record.get('device_timestamp_s', '?')} "
        f"host_s={record.get('host_received_timestamp_s', '?')} " + "; ".join(branches)
    )


def receive(args, *, on_record=None, printer=print):
    # Subscribe locally, process queued JSON records and stop at the requested limit.
    validate_local_endpoint(args.host, args.port, args.topic)

    if args.max_messages < 0:
        raise ValueError("max-messages must be zero or positive")

    mqtt = load_mqtt()
    subscribed, failed = Event(), Event()
    # Hold up to 128 incoming records while the main loop processes earlier ones.
    messages = Queue(maxsize=128)
    errors = []
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv311,
        reconnect_on_failure=False,
    )
    client.connect_timeout = min(3.0, args.timeout_s)
    output = None
    count = 0
    closing = False
    deadline = monotonic() + args.timeout_s

    def fail(message):
        # Save the MQTT error and signal the receive loop to stop.
        errors.append(message)
        failed.set()
        # Release the setup wait so it can inspect the failed flag and report the error.
        subscribed.set()

    def on_connect(client, userdata, flags, reason_code, properties):
        # Subscribe to the chosen topic only after the broker accepts the connection.
        if reason_code.is_failure:
            fail(f"MQTT connection rejected: {reason_code}")
            return

        result, _ = client.subscribe(args.topic, qos=1)

        if result != mqtt.MQTT_ERR_SUCCESS:
            fail(f"MQTT subscription failed: {result}")

    def on_subscribe(client, userdata, mid, reason_codes, properties):
        # Mark setup complete only when the broker acknowledges the subscription.
        if not reason_codes or any(code.is_failure for code in reason_codes):
            fail("MQTT broker rejected the subscription")

        else:
            subscribed.set()

    def on_disconnect(client, userdata, flags, reason_code, properties):
        # Stop the receive loop on unexpected disconnection, but allow normal shutdown.
        if not closing:
            fail(f"MQTT broker disconnected: {reason_code}")

    def on_message(client, userdata, message):
        try:
            # Queue the raw payload without blocking the network callback on processing.
            messages.put_nowait(message.payload)

        except Full:
            # Stop with an explicit error if processing cannot keep up with arrivals.
            fail("MQTT subscriber cannot keep up: receive queue is full")

    client.on_connect = on_connect
    client.on_connect_fail = lambda client, userdata: fail(
        "MQTT broker connection failed"
    )
    client.on_subscribe = on_subscribe
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        if args.output:
            output = args.output.open("x", encoding="utf-8")

        client.connect_async("127.0.0.1", args.port, keepalive=30)
        result = client.loop_start()

        if result != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(f"MQTT network loop could not start: {result}")

        # Wait at most three seconds, or the shorter runtime, for subscription setup.
        if not subscribed.wait(min(3.0, args.timeout_s)):
            raise TimeoutError("MQTT connection/subscription acknowledgement timed out")

        if failed.is_set():
            raise ConnectionError(errors[0])

        if args.ready_file:
            # Only announce readiness after the broker accepts our subscription.
            with args.ready_file.open("x", encoding="utf-8") as ready:
                ready.write(f"subscribed {args.topic}\n")

        printer(
            f"Subscribed to mqtt://127.0.0.1:{args.port}/{args.topic} (QoS 1). "
            "These are frame measurements; video arrives separately. "
            "Match run_id + sequence_number; QoS 1 may repeat a message.",
            flush=True,
        )

        # Receive up to the requested count, or until timeout when the count is zero.
        while (
            not args.max_messages
            or count < args.max_messages
        ):
            if failed.is_set():
                raise ConnectionError(errors[0])

            remaining = deadline - monotonic()

            # At the deadline, fail an unmet message count or finish a timed-only run.
            if remaining <= 0:
                if args.max_messages:
                    raise TimeoutError(
                        f"Received {count}/{args.max_messages} messages before timeout"
                    )

                break

            try:
                payload = messages.get(timeout=min(0.1, remaining))

            except Empty:
                # No message arrived in this wait; recheck the deadline and connection.
                continue

            # Decode the payload as UTF-8 JSON, rejecting NaN and infinity.
            record = json.loads(
                payload.decode("utf-8"), parse_constant=_reject_constant
            )

            if not isinstance(record, dict):
                raise ValueError("MQTT result must be a JSON object")

            serialized = json.dumps(record, allow_nan=False, separators=(",", ":"))

            if output:
                # Append the complete record as one JSON line and flush it to the file.
                output.write(serialized + "\n")
                output.flush()

            if on_record is not None:
                # Pass the decoded record to the caller, such as the GCS log sampler.
                on_record(record)

            printer(_summary(record), flush=True)
            count += 1

        printer(f"Received {count} MQTT message(s).", flush=True)
        return count

    finally:
        # Mark shutdown intentional, then close the connection, thread and file.
        closing = True

        try:
            client.disconnect()

        finally:
            client.loop_stop()

            if output:
                output.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        receive(args)

    except KeyboardInterrupt:
        return 130

    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"MQTT subscriber error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
