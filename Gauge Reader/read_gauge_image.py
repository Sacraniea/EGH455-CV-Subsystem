"""Run the gauge reader against one candidate image."""

from __future__ import annotations

from gauge_overlay import render_overlay
from gauge_reader import read_gauge
from typing import Callable, Optional, Sequence

import argparse
import cv2
import json
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments accepted by this program."""

    parser = argparse.ArgumentParser(
        description="Read one candidate gauge crop",
    )

    parser.add_argument("input_image")

    # Accept an optional output path for an annotated copy of the input image.
    parser.add_argument(
        "--overlay",
        metavar="OUTPUT_IMAGE",
        help=" save annotated copy of the candidate image",
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    image_loader: Callable[[str], Optional[np.ndarray]] = cv2.imread,
    image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    printer: Callable[[str], None] = print,
) -> int:
    """Read one image, print its JSON result, and return the process exit code."""
    args = build_parser().parse_args(argv)
    image = image_loader(args.input_image)

    # If the input file could not be decoded, report the path and exit with code 1.
    if image is None:
        printer(f"Could not read input image: {args.input_image}")
        return 1

    result = read_gauge(
        image,
        metadata={
            "input_image": args.input_image,
        },
    )

    # Serialize the result as strict JSON before attempting any optional image save.
    document = json.dumps(
        result.to_dict(),
        allow_nan=False,
        sort_keys=True,
    )
    printer(document)

    # If an overlay path was supplied, draw the result on a copy and save that image.
    if args.overlay is not None:
        rendered = render_overlay(
            image,
            result,
        )

        write_succeeded = image_writer(
            args.overlay,
            rendered,
        )

        # If saving fails, print the output error and return exit code 1.
        if not write_succeeded:
            printer(f"Could not write overlay image: {args.overlay}")
            return 1

    # Return 0 for an accepted gauge reading or 2 when the loaded crop was rejected.
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
