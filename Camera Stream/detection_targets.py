"""Connect target detection and gauge reading to the shared camera stream."""

from dataclasses import asdict, replace
from enum import Enum
from hashlib import file_digest, sha256
from math import isfinite
from numbers import Real
from pathlib import Path
import sys

VISION_DIRECTORY = Path(__file__).resolve().parent.parent
TARGET_DIRECTORY = VISION_DIRECTORY / "Target Detection"
GAUGE_DIRECTORY = VISION_DIRECTORY / "Gauge Reader"


for directory in (TARGET_DIRECTORY / "src", GAUGE_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gauge_reader import read_gauge
from gauge_overlay import render_overlay
from gauge_types import GaugeCalibration, ReaderConfig
from detection_pipeline import FrameIdentity, ProcessingResult
from detection_tracking import TargetTracker


class StreamReadingFailure(str, Enum):
    READING_UNCONFIRMED = "READING_UNCONFIRMED"
    INVALID_READING = "INVALID_READING"


def reading_record(reading):
    """Prepare a gauge reading for JSON records and identify replaced numbers."""

    # Store which fields contained numbers that JSON cannot represent.
    nonfinite_fields = []

    def clean(value, field_path):
        # Replace an undefined or infinite measurement with an empty value.
        if (
            isinstance(value, Real)
            and not isfinite(value)
        ):
            nonfinite_fields.append(field_path)
            return None

        # Check every field inside a nested dictionary, such as reader metadata.
        if isinstance(value, dict):
            cleaned_fields = {}

            for key, item in value.items():
                item_path = f"{field_path}.{key}"
                cleaned_fields[key] = clean(item, item_path)

            return cleaned_fields

        # Check each item in a sequence, including pivot and needle coordinates.
        # Return a list so coordinates use the same structure as JSON arrays.
        if isinstance(value, (tuple, list)):
            cleaned_items = []

            for index, item in enumerate(value):
                item_path = f"{field_path}[{index}]"
                cleaned_items.append(
                    clean(item, item_path)
                )

            return cleaned_items

        # Keep the remaining gauge fields unchanged.
        return value

    # Convert the reading to a dictionary and prepare each field for recording.
    record = {}

    for key, value in reading.to_dict().items():
        record[key] = clean(value, key)

    # Record where numbers were replaced so the saved evidence explains the change.
    if nonfinite_fields:
        record["metadata"]["nonfinite_fields"] = nonfinite_fields

    return record

def prepare_targets(model_path = None):
    """Load the target detector and return the settings used for this run."""

    # Use the supplied model file, or the project's default gauge and valve model.
    model_path = Path(
        model_path or TARGET_DIRECTORY / "models" / "50_epoch_best.pt"
    ).expanduser().resolve()

    # Stop before opening the camera if the model file cannot be found.
    if not model_path.is_file():
        raise FileNotFoundError(f"Target model file does not exist: {model_path}")

    # Import the predictor when target detection is being set up.
    try:
        from predict import initialise_predictor

    except ImportError as error:
        raise RuntimeError(
            "Target detection could not be imported. "
            "Run pixi install from the repository root, "
            "then launch the sender using pixi run python. "
            f"Import error: {error}"
        ) from error

    # Load the model once so every frame can use the same predictor.
    predictor = initialise_predictor()

    # Check that the model's numbered classes match the labels our code expects.
    expected_labels = dict(enumerate(predictor.class_names))

    if predictor.model.names != expected_labels:
        raise ValueError(
            "The model's class labels do not match the expected target labels. "
            f"Expected: {expected_labels}; "
            f"model contains: {predictor.model.names}"
        )

    # Connect the predictor to gauge reading, reading confirmation and annotation.
    processor = TargetDetectionProcessor(
        predictor = predictor,
    )

    # Save the model fingerprint so later runs can identify the exact model used.
    # NCNN exports are a folder, so hash every file in it in a fixed order.
    model_files = (
        sorted(path for path in model_path.rglob("*") if path.is_file())
        if model_path.is_dir() else [model_path]
    )
    model_hash = sha256()
    for path in model_files:
        model_hash.update(str(path.relative_to(model_path.parent)).encode())
        with path.open("rb") as model_file:
            model_hash.update(file_digest(model_file, "sha256").digest())
    model_sha256 = model_hash.hexdigest()

    # Return the settings alongside the processor for recording with the run.
    metadata = {
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "class_names": list(predictor.class_names),
        "gauge_calibration": asdict(processor.gauge_calibration),
        "reader_config": asdict(processor.reader_config),
        "target_tracking": asdict(processor.tracker.config),
    }

    return processor, metadata


class TargetDetectionProcessor:
    """Detect gauges and valves, read pressure and prepare their annotations."""

    # Identify these results when combining them with other detectors.
    name = "targets"

    def __init__(
        self,
        *,
        predictor,
        gauge_calibration = GaugeCalibration(),
        reader_config = ReaderConfig(),
    ):
        # Keep the loaded model so every frame can use it.
        self.predictor = predictor

        # Keep the gauge scale used to convert needle angles into pressure.
        self.gauge_calibration = gauge_calibration

        # Keep the image and needle checks used to accept or reject a reading.
        self.reader_config = reader_config

        # Remember targets between frames so each gauge has its own reading history.
        self.tracker = TargetTracker()

    def process(self, packet):
        """Process one frame and clear reading history if processing fails."""

        try:
            # Find this frame's targets and read any detected gauges.
            return self._process_frame(packet)

        except Exception:
            # Discard earlier confirmation readings after a processing error.
            # The next successful frame must begin a fresh confirmation history.
            self.tracker.reset()

            # Pass the error to the shared pipeline to record against this frame.
            raise

    def _process_frame(self, packet):
        # Find the targets in this frame.
        # The predictor returns each target's label, box and any gauge crop.
        predictions = self.predictor.predict_detections(packet.image)

        # Keep the frame number, timestamps and image dimensions with its results.
        identity = FrameIdentity.from_packet(packet)

        # Match these detections to targets seen in earlier frames.
        tracks = self.tracker.associate(predictions, identity)

        # Start fresh lists so this frame cannot reuse an earlier detection.
        detections = []
        annotations = []

        # Process each detected target together with its tracking information.
        for prediction, track in zip(predictions, tracks):
            class_name = prediction["class_name"]
            left, top, right, bottom = prediction["bbox_xyxy_px"]

            # A target has no pressure reading unless it is processed as a gauge.
            reading = None
            reading_status = None

            # Record the target even if its pressure cannot be read.
            detection = {
                "class_name": class_name,
                "bbox_xyxy_px": (left, top, right, bottom),
                "track_id": track.track_id,
                "reading_status": None,
                "raw_gauge_reading": None,
                "gauge_reading": None,
            }

            if class_name == "gauge":
                # Read this gauge's clean crop supplied by the predictor.
                reading = read_gauge(
                    prediction["gauge_crop"],
                    calibration = self.gauge_calibration,
                    config = self.reader_config,
                    metadata = {
                        "coordinate_frame": "crop_pixels",
                        "crop_origin_px": [left, top],
                    },
                )

                # Keep the reader's result before checking agreement between frames.
                detection["raw_gauge_reading"] = reading_record(reading)

                # Check that the pressure is a usable number for confirmation.
                value = reading.value_bar

                finite_pressure = (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and isfinite(value)
                )

                # Give the tracker an accepted pressure to compare with recent readings.
                if (reading.valid and finite_pressure):
                    pressure_to_confirm = value
                else:
                    # An unreadable pressure breaks this gauge's confirmation history.
                    pressure_to_confirm = None

                reading_status = self.tracker.confirm(track,pressure_to_confirm,)

                # Withhold a pressure that the reader accepted but tracking has not confirmed.
                if (reading.valid and reading_status != "confirmed"):

                    if finite_pressure:
                        # The pressure needs more agreeing readings before it can be shown.
                        reason = StreamReadingFailure.READING_UNCONFIRMED

                    else:
                        # The reported pressure cannot be used as a measurement.
                        reason = StreamReadingFailure.INVALID_READING

                    # Keep this frame's pivot, needle and evidence, but remove its pressure.
                    reading = replace(
                        reading,
                        valid = False,
                        value_bar = None,
                        failure_reason = reason,
                    )

                # Store the confirmation status and the reading allowed for display.
                detection["reading_status"] = reading_status
                detection["gauge_reading"] = reading_record(reading)

            # Keep every target, including valves and gauges with unavailable pressure.
            detections.append(detection)

            # Keep the information needed to draw this target later.
            # Drawing will reuse these results without running detection again.
            annotations.append({
                "class_name": class_name,
                "box_xywhn": prediction["box_xywhn"],
                "bbox_xyxy_px": (left, top, right, bottom),
                "gauge_reading": reading,
                "reading_status": reading_status,
            })

        # Return all targets and drawing information with their source frame identity.
        return ProcessingResult(
            self.name,
            identity = identity,
            data = {
                "schema_version": 1,
                "results": detections,
            },
            annotations = tuple(annotations),
        )

    def draw(self, overlay, result):
        """Add target labels and available gauge geometry to the display image."""

        labels = []

        # Prepare the label for every target detected in this frame.
        for annotation in result.annotations:
            label = annotation["class_name"]
            reading = annotation["gauge_reading"]

            # Gauges have a reading object; valves keep their open or closed label.
            if reading is not None:
                if reading.valid:
                    # Show the confirmed pressure to one decimal place.
                    label = f"{label} | {reading.value_bar:.1f} bar"

                elif annotation["reading_status"] == "pending":
                    # Explain that the pressure still needs agreeing readings.
                    label = f"{label} | pressure awaiting confirmation"

                else:
                    # Keep the gauge labelled when its pressure cannot be read.
                    label = f"{label} | pressure unavailable"

            # Give the existing renderer the text and its matching target box.
            labels.append(
                (label, annotation["box_xywhn"])
            )

        # Draw all target boxes and labels, keeping the returned image.
        overlay = self.predictor.annotate_image(
            overlay,
            labels,
        )

        # Add the saved pivot and needle geometry for each gauge.
        for annotation in result.annotations:
            reading = annotation["gauge_reading"]

            if reading is None:
                # Valves already have their labels and need no gauge drawing.
                continue

            # Select this gauge's region from the image we have already labelled.
            left, top, right, bottom = annotation["bbox_xyxy_px"]
            gauge_region = overlay[top:bottom, left:right]

            if gauge_region.size == 0:
                # Keep the target label if there is no image region to draw into.
                continue

            # Draw any available pivot and needle using the saved reading.
            # The full-frame label already shows the pressure or its status.
            annotated_region = render_overlay(
                gauge_region,
                reading,
                show_label = False,
            )

            # Put the annotated region back into the full display image.
            overlay[top:bottom, left:right] = annotated_region

        return overlay