"""Preview or publish sampled ArUco, gauge and valve results to the GCS log topic.

Run alongside the existing camera demo. The local MQTT subscription contains
structured frame results, not images. Network delays and failures affect this
separate process; the camera and HTTP video process keep their own lifecycle.
"""

import argparse
from collections import OrderedDict
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

from gcs_output import gcs_log_payloads
from run_mqtt_subscriber import receive, subscription_runtime
from stream_mqtt import MqttResultPublisher, positive_timeout, validate_local_endpoint


class LogSampler:
    """Choose recent detections at a readable rate for the GCS log."""

    def __init__(self, *, interval_s=5.0, clock=monotonic):
        self.interval_s = positive_timeout(interval_s)
        self.clock = clock
        self.last_selected = None
        self.sequences = OrderedDict()

    def select(self, record):
        # Select at most one new frame per interval and keep all its detection logs.
        run_id, sequence = record.get("run_id"), record.get("sequence_number")

        # Reject records without a run ID or a nonnegative integer frame number.
        if (
            not isinstance(run_id, str)
            or not run_id
            or type(sequence) is not int
            or sequence < 0
        ):
            raise ValueError(
                "Frame results require a run_id and nonnegative integer sequence_number"
            )

        previous = self.sequences.get(run_id)

        # Skip duplicate or older frame numbers from the same camera run.
        if previous is not None and sequence <= previous:
            return None

        self.sequences[run_id] = sequence
        self.sequences.move_to_end(run_id)

        # If more than 16 runs are tracked, remove the least recently seen run.
        if len(self.sequences) > 16:
            self.sequences.popitem(last=False)

        payloads = gcs_log_payloads(record)

        if not payloads:
            return None

        now = self.clock()

        # Skip this frame if the interval since the last selected frame has not passed.
        if (
            self.last_selected is not None
            and now - self.last_selected < self.interval_s
        ):
            return None

        self.last_selected = now
        return payloads


def build_parser():
    # Define the source subscription, GCS destination and sampling options.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Send to GCS; without this flag, preview locally only.",
    )
    parser.add_argument(
        "--broker",
        help="GCS broker hostname/IP, required with --publish; no mqtt:// prefix.",
    )
    parser.add_argument(
        "--port", type=int, default=1883, help="GCS broker port (default: 1883)."
    )
    parser.add_argument("--local-port", type=int, default=18883)
    parser.add_argument("--local-topic", default="taip/local/results")
    parser.add_argument(
        "--interval-s",
        type=float,
        default=5.0,
        help=(
            "Minimum seconds between sampled frames; "
            "each can produce several logs (default: 5)."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Runtime in seconds; 0 keeps forwarding until Ctrl+C (default: 600).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many incoming records; 0 uses timeout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSONL file for prepared payloads and delivery outcomes.",
    )
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None, *, printer=print):
    # Validate the command, subscribe to local results and forward selected logs.
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.publish and not args.broker:
        parser.error("--publish requires --broker")

    try:
        sampler = LogSampler(interval_s=args.interval_s)
        args.timeout_s = subscription_runtime(args.timeout_s)
        validate_local_endpoint("127.0.0.1", args.local_port, args.local_topic)

        if (
            not 1 <= args.port <= 65535
            or args.max_frames < 0
        ):
            raise ValueError("Port must be 1..65535 and max-frames must be nonnegative")

        if args.local_topic == "gcs/drone_log":
            # Reject the GCS log topic as input; this reader expects full frame records.
            raise ValueError(
                "The source must be native frame results, not gcs/drone_log"
            )

        output = args.output

        if output is None:
            # Create a unique audit-file name when the caller did not choose one.
            directory = Path("diagnostics/gcs")
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + uuid4().hex[:8]
                + ".jsonl"
            )

        # Close the evidence file and MQTT connection together, including on errors.
        with ExitStack() as resources:
            audit = resources.enter_context(output.open("x", encoding="utf-8"))
            publisher = None

            if args.publish:
                publisher = MqttResultPublisher(
                    host=args.broker,
                    port=args.port,
                    topic="gcs/drone_log",
                    allow_remote=True,
                )
                resources.callback(publisher.close)

            mode = "PUBLISH" if args.publish else "PREVIEW — no GCS connection"
            printer(
                f"{mode}; sample interval={args.interval_s:g}s; evidence={output.resolve()}",
                flush=True,
            )
            printer(
                f"Source: 127.0.0.1:{args.local_port} topic={args.local_topic}; "
                "each sampled frame can produce several detection logs.",
                flush=True,
            )

            if args.publish:
                printer(
                    f"Destination: {args.broker}:{args.port} topic=gcs/drone_log.",
                    flush=True,
                )

            def handle(record):
                # Select this frame's messages together so one target cannot hide another.
                payloads = sampler.select(record)

                if payloads is None:
                    return

                send_error = None

                for payload in payloads:
                    entry = dict(
                        topic="gcs/drone_log",
                        payload=payload,
                        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                        delivery="preview",
                        run_id=record["run_id"],
                        sequence_number=record["sequence_number"],
                    )

                    try:
                        if send_error is not None:
                            # Record the unsent logs after an earlier publication failed.
                            entry.update(
                                delivery="not_attempted",
                                error=(
                                    "Earlier message failed: "
                                    f"{type(send_error).__name__}: {send_error}"
                                ),
                            )

                        elif publisher is not None:
                            publisher.publish(payload)
                            # Record acceptance for this message, not the whole frame.
                            entry["delivery"] = "broker_acknowledged"

                    except (Exception, KeyboardInterrupt) as error:
                        # Keep earlier acknowledgements and record which message failed.
                        entry.update(
                            delivery="failed", error=f"{type(error).__name__}: {error}"
                        )
                        send_error = error

                    finally:
                        # Save each outcome immediately, including a failed send.
                        audit.write(json.dumps(entry, allow_nan=False) + "\n")
                        audit.flush()

                    printer(
                        f'{entry["delivery"]}: {payload["action"]} | {payload["content"]}',
                        flush=True,
                    )

                if send_error is not None:
                    # Stop after saving every selected message's delivery outcome.
                    raise send_error

            subscription = SimpleNamespace(
                # Receive full frame results locally, then forward only selected logs.
                host="127.0.0.1",
                port=args.local_port,
                topic=args.local_topic,
                timeout_s=args.timeout_s,
                max_messages=args.max_frames,
                output=None,
                ready_file=args.ready_file,
            )
            count = receive(
                subscription, on_record=handle, printer=lambda *a, **k: None
            )
            printer(
                f"Processed {count} source frame records. GCS database ingestion requires confirmation on the GCS.",
                flush=True,
            )

        return 0

    except KeyboardInterrupt:
        return 130

    except (OSError, RuntimeError, ValueError, TypeError) as error:
        printer(f"GCS forwarder error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    # Windows sends Ctrl+Break to child groups; run the usual interrupt cleanup.
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    raise SystemExit(main())
