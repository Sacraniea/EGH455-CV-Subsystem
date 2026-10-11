"""Summarise a frames.jsonl run; optional reference assumes one fixed distance."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

from aruco_cli import positive_finite


def distribution(values):
    # Summarise these measurements, returning None for statistics with no samples.
    if not values:
        return dict(
            count=0,
            mean=None,
            stddev=None,
            minimum=None,
            maximum=None,
            p95=None,
        )

    ordered = sorted(values)
    return dict(
        count=len(values),
        mean=statistics.fmean(values),
        stddev=statistics.pstdev(values),
        minimum=ordered[0],
        maximum=ordered[-1],
        # Round up to the first sorted value covering at least 95% of samples.
        p95=ordered[math.ceil(0.95 * len(values)) - 1],
    )


def evaluate_run(path: Path, *, reference_distance_m=None):
    """Read the standalone ArUco frames.jsonl format."""
    # Count detections and summarise accepted ranges and errors for each marker ID.

    if reference_distance_m is not None:
        positive_finite(str(reference_distance_m))

    frames = 0
    empty = 0
    timings = []
    markers = {}

    with Path(path).open(encoding="utf-8") as file:
        # Read one frame per nonblank line and retain line numbers for error reports.
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                # Read marker results directly from the standalone frame record.
                results = record["results"]
                timing = record["processing_time_ms"]

                # Reject invalid timing values before adding them to the statistics.
                if (
                    not math.isfinite(timing)
                    or timing < 0
                ):
                    raise ValueError("invalid processing time")

                timings.append(timing)
                frames += 1
                empty += int(not results)

                # Collect separate counts and measurements for each observed marker ID.
                for result in results:
                    item = markers.setdefault(
                        str(result["marker_id"]),
                        dict(
                            accepted=0,
                            rejected=0,
                            ranges=[],
                            errors=[],
                            reasons=Counter(),
                        ),
                    )

                    if result["valid"]:
                        # Require three finite coordinates with the marker in front of the camera.
                        translation_m = result["pose"]["translation_m"]

                        if (
                            len(translation_m) != 3
                            or not all(
                                math.isfinite(value) for value in translation_m
                            )
                            or translation_m[2] <= 0
                        ):
                            raise ValueError("invalid accepted translation")

                        # Convert X/Y/Z into camera-to-marker distance for this accepted pose.
                        item["accepted"] += 1
                        item["ranges"].append(math.hypot(*translation_m))
                        error_px = result["reprojection_error_px"]

                        # Require a finite, nonnegative pixel error for every accepted pose.
                        if (
                            error_px is None
                            or not math.isfinite(error_px)
                            or error_px < 0
                        ):
                            raise ValueError("invalid accepted reprojection error")

                        item["errors"].append(error_px)

                    else:
                        # Count rejected poses by reason without adding their ranges.
                        item["rejected"] += 1
                        item["reasons"][
                            result["failure_reason"] or "POSE_UNAVAILABLE"
                        ] += 1

            except (ValueError, KeyError, TypeError, IndexError) as error:
                # Identify the malformed line and stop instead of returning partial statistics.
                raise ValueError(
                    f"Invalid frame record at line {line_number}: {error}"
                ) from error

    report = dict(
        source=str(Path(path).resolve()),
        frames=frames,
        empty_frames=empty,
        reference_distance_m=reference_distance_m,
        reference_assumption="constant camera-to-marker-centre range for the entire run",
        processing_time_ms=distribution(timings),
        markers={},
    )

    # Replace each marker's sample lists with summary statistics for the report.
    for marker_id, item in markers.items():
        ranges = item["ranges"]
        report["markers"][marker_id] = dict(
            accepted=item["accepted"],
            rejected=item["rejected"],
            rejection_reasons=dict(item["reasons"]),
            range_m=distribution(ranges),
            reprojection_error_px=distribution(item["errors"]),
            # Average absolute range errors only when a reference and samples exist.
            mean_absolute_range_error_m=(
                statistics.fmean(
                    abs(distance_m - reference_distance_m)
                    for distance_m in ranges
                )
                if (
                    reference_distance_m is not None
                    and ranges
                )
                else None
            ),
        )

    return report


def main(argv=None):
    # Evaluate the selected log and write the report to a new file or standard output.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument(
        "--reference-distance-m",
        type=positive_finite,
        help="Measured constant range; omit for motion trials or estimated distances.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON file; otherwise print to stdout.",
    )
    args = parser.parse_args(argv)

    try:
        report = evaluate_run(
            args.frames,
            reference_distance_m=args.reference_distance_m,
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"

        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(text)

        else:
            print(text, end="")

    except (OSError, ValueError) as error:
        # Report an unreadable log or invalid record as a failing command.
        print(f"ArUco evaluation error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
