"""Read a BGR image and save strict JSON evidence and an annotated PNG."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2

from aruco_calibration import load_calibration
from aruco_cli import add_processing_arguments, config_from_args
from aruco_overlay import draw_aruco_overlay
from aruco_processing import process_aruco_frame


def main(argv=None) -> int:
    # Read one image using the supplied options, then save its detections and overlay.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing evidence is never overwritten.",
    )
    add_processing_arguments(parser)
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        input_path = args.image.resolve()
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

        # Report unreadable input before attempting any marker processing.
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")

        calibration = load_calibration(args.calibration) if args.calibration else None
        # Wrap the image as a frame packet with no camera sequence or timestamps.
        packet = SimpleNamespace(
            image=image,
            sequence_number=None,
            device_timestamp_s=None,
            host_received_timestamp_s=None,
        )
        # Use the same detection and pose routine as the live camera command.
        results, report = process_aruco_frame(
            packet,
            calibration,
            config,
            max_reprojection_error_px=args.max_reprojection_error_px,
        )
        report["source_image"] = str(input_path)
        overlay = draw_aruco_overlay(image, results)

        # Require a new output directory and save the annotated image inside it.
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        overlay_path = output_dir / "overlay.png"

        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError(f"Could not save overlay: {overlay_path}")

        # Write JSON last so results.json means both output files are complete.
        temporary = output_dir / "results.json.tmp"

        with temporary.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")

        temporary.replace(output_dir / "results.json")

    except (OSError, ValueError, TypeError, KeyError, cv2.error) as error:
        # Print the input, processing or write failure and return a failing exit code.
        print(f"ArUco image error: {error}", file=sys.stderr)
        return 1

    print(f"Saved ArUco results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
