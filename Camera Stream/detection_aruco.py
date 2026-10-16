"""Connect the existing ArUco reader and drawing code to the pipeline."""

from pathlib import Path
import sys


ARUCO_DIRECTORY = Path(__file__).resolve().parent.parent / "ArUco Detection"

# Import the shared reader directly so this adapter does not depend on a stream runner.
if str(ARUCO_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ARUCO_DIRECTORY))

from aruco_calibration import load_calibration, load_stream_calibration
from aruco_cli import add_processing_arguments, config_from_args
from aruco_processing import process_aruco_frame
from aruco_overlay import draw_aruco_overlay
from detection_pipeline import FrameIdentity, ProcessingResult


class ArUcoProcessor:
    """Detect markers once, then draw their saved observations on the overlay."""

    name = "aruco"

    def __init__(self, *, calibration, config, max_reprojection_error_px):
        self.calibration = calibration
        self.config = config
        self.max_reprojection_error_px = max_reprojection_error_px

    def process(self, packet):
        # Keep the established reader's detection, pose checks and record format.
        observations, record = process_aruco_frame(
            packet,
            self.calibration,
            self.config,
            max_reprojection_error_px = self.max_reprojection_error_px,
        )

        return ProcessingResult(
            processor = self.name,
            identity = FrameIdentity.from_packet(packet),
            data = record,
            annotations = observations,
        )

    def draw(self, overlay, result):
        # The existing renderer copies the overlay and keeps unavailable poses labelled.
        return draw_aruco_overlay(overlay, result.annotations)


def prepare_aruco(args, profile):
    """Prepare marker settings and optional OAK calibration before opening a source."""

    # Generated images use only their explicitly selected source's invented geometry.
    if (
        args.source == "synthetic"
        and (args.calibration or args.calibration_mode != "exact")
    ):
        raise ValueError(
            "synthetic frames use generated calibration; omit calibration options"
        )

    # Omitted IDs accept the full dictionary; explicit selections keep their meaning.
    config = config_from_args(args)
    calibration = None
    evidence = None

    if args.calibration:
        calibration, evidence = load_stream_calibration(
            args.calibration,
            image_size_px = (profile.width, profile.height),
            mode = args.calibration_mode,
        )

    # Missing calibration preserves IDs and corners while the reader withholds pose.
    processor = ArUcoProcessor(
        calibration = calibration,
        config = config,
        max_reprojection_error_px = args.max_reprojection_error_px,
    )
    metadata = {
        "dictionary_name": config.dictionary_name,
        "permitted_ids": sorted(config.permitted_ids),
        "marker_side_length_m": config.marker_side_length_m,
        "calibration": evidence,
        "max_reprojection_error_px": args.max_reprojection_error_px,
    }
    return processor, metadata
