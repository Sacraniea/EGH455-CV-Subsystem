"""Preview or publish one original PNG/JPEG file using the GCS detection schema."""

import argparse
import base64
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

from gcs_output import gcs_detection_payload
from stream_mqtt import MqttResultPublisher


def main(argv=None, *, printer=print):
    # Prepare one original image for the GCS; send it only when --publish is set.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--type", dest="target_type", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument(
        "--timestamp",
        help="Known ISO8601 event time; defaults to current UTC publication time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the full prepared JSON to a new file.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Send once; without this flag, preview locally only.",
    )
    parser.add_argument(
        "--broker", help="GCS hostname/IP, required with --publish; no mqtt:// prefix."
    )
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args(argv)

    if args.publish and not args.broker:
        parser.error("--publish requires --broker")

    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")

        # Validate the image and package its original bytes with the caption and time.
        payload = gcs_detection_payload(
            args.image, args.target_type, args.description, timestamp=args.timestamp
        )
        encoded = json.dumps(payload, allow_nan=False)

        if args.output:
            # Use a new file so an earlier test record isn't overwritten.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded + "\n")

        original = base64.b64decode(payload["image"], validate=True)
        # Show size and a file fingerprint, without filling the terminal with Base64.
        printer(
            json.dumps(
                dict(
                    topic="gcs/detection",
                    type=payload["type"],
                    mime=payload["mime"],
                    timestamp=payload["timestamp"],
                    original_bytes=len(original),
                    sha256=hashlib.sha256(original).hexdigest(),
                    description=payload["description"],
                ),
                indent=2,
            ),
            flush=True,
        )

        if args.publish:
            # Open the chosen broker connection and close it after this one send.
            with ExitStack() as resources:
                publisher = MqttResultPublisher(
                    host=args.broker,
                    port=args.port,
                    topic="gcs/detection",
                    allow_remote=True,
                )
                resources.callback(publisher.close)
                # Publish the image payload and wait for the broker acknowledgement.
                publisher.publish(payload)

            printer(
                "Broker acknowledged gcs/detection; verify the saved image on the GCS.",
                flush=True,
            )

        else:
            printer("PREVIEW ONLY — no GCS connection or publication.", flush=True)

        return 0

    except KeyboardInterrupt:
        return 130

    except (OSError, RuntimeError, ValueError, TypeError) as error:
        printer(f"GCS detection error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
