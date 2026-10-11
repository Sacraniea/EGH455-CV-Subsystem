"""Common options for image and live commands."""

import argparse
from math import isfinite
from pathlib import Path

from aruco_types import ArUcoConfig


def positive_finite(value: str) -> float:
    # Convert a command-line value to a positive number or report an argument error.
    try:
        number = float(value)

    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error

    # Reject zero, negative values, infinity and NaN before creating the configuration.
    if (
        not isfinite(number)
        or number <= 0
    ):
        raise argparse.ArgumentTypeError("must be finite and greater than zero")

    return number


def add_processing_arguments(parser):
    # Add the marker selection, physical size, calibration and pixel-error options.
    parser.add_argument(
        "--permitted-ids",
        type=int,
        nargs="+",
        help="Restrict accepted marker IDs; omission accepts all DICT_5X5_100 IDs (0-99).",
    )
    parser.add_argument(
        "--marker-side-length-m",
        type=positive_finite,
        default=0.2,
        help="Measured outer black square side in metres (default: 0.2).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="CAM_A perspective calibration JSON; omission withholds poses.",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=positive_finite,
        default=1.0,
        help="Provisional RMS corner-error limit in pixels (default: 1).",
    )


def config_from_args(args):
    # Accept the whole dictionary only when no selection was supplied.
    # An explicit empty collection still means that no marker IDs are permitted.
    permitted_ids = range(100) if args.permitted_ids is None else args.permitted_ids

    # Build the detector configuration and combine repeated permitted IDs into a set.
    return ArUcoConfig(
        args.marker_side_length_m,
        "DICT_5X5_100",
        frozenset(permitted_ids),
    )
