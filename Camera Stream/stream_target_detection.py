"""Connect target detection and gauge reading to the shared camera stream."""

from dataclasses import asdict, replace
from enum import Enum
from hashlib import file_digest
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
from stream_processing import FrameIdentity, ProcessingResult
from stream_target_tracking import TargetTracker


class StreamReadingFailure(str, Enum):
    READING_UNCONFIRMED = "READING_UNCONFIRMED"
    INVALID_READING = "INVALID_READING"


def reading_record(reading):
    """Keep the reader's evidence JSON-safe, recording any nonfinite fields."""

    nonfinite_fields = []

    def clean(value, path):
        if isinstance(value, Real) and not isfinite(value):
            nonfinite_fields.append(path)
            return None

        if isinstance(value, dict):
            return {key: clean(item, f"{path}.{key}") for key, item in value.items()}

        if isinstance(value, (tuple, list)):
            return [clean(item, f"{path}[{index}]") for index, item in enumerate(value)]

        return value

    record = {key: clean(value, key) for key, value in reading.to_dict().items()}
    if nonfinite_fields:
        record["metadata"]["nonfinite_fields"] = nonfinite_fields

    return record


def prepare_targets(model_path = None):
    """Load Aidan's local model once and record which settings the stream uses."""

    # Find the weights beside Aidan's source, regardless of the terminal's folder.
    model_path = Path(
        model_path or TARGET_DIRECTORY / "models" / "50_epoch_best.pt"
    ).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Target model file does not exist: {model_path}")

    # Camera-only and ArUco runs do not need to import the model dependencies.
    try:
        from predict import Predictor

    except ImportError as error:
        raise RuntimeError(
            "Target detection dependencies could not be imported. "
            "Use Camera Stream/.venv-targets/bin/python after installing "
            f"requirements-targets.txt. Import error: {error}"
        ) from error

    # Reuse this predictor for every frame instead of loading weights repeatedly.
    predictor = Predictor(str(model_path))

    # Aidan uses the dataset's label order. Reject weights that give those IDs
    # different meanings, so a valve cannot accidentally be passed to the reader.
    if predictor.model.names != dict(enumerate(predictor.class_names)):
        raise ValueError(
            "Target model class names must match Aidan's dataset label order: "
            f"{predictor.class_names}; weights contain {predictor.model.names}"
        )

    processor = TargetDetectionProcessor(predictor = predictor)

    # Save the file fingerprint so later runs can identify the exact model used.
    with model_path.open("rb") as model_file:
        model_sha256 = file_digest(model_file, "sha256").hexdigest()

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
    """Read this frame's target crops and draw their results on a display copy."""

    name = "targets"

    def __init__(
        self,
        *,
        predictor,
        gauge_calibration = GaugeCalibration(),
        reader_config = ReaderConfig(),
    ):

        self.predictor = predictor
        self.gauge_calibration = gauge_calibration
        self.reader_config = reader_config
        self.tracker = TargetTracker()

    def process(self, packet):
        try:
            return self._process_frame(packet)

        except Exception:
            # A failed inference or reader call cannot bridge two valid histories.
            self.tracker.reset()
            raise

    def _process_frame(self, packet):
        # Get this frame's targets with the boxes and crops Aidan already prepared.
        predictions = self.predictor.predict_detections(packet.image)
        identity = FrameIdentity.from_packet(packet)
        tracks = self.tracker.associate(predictions, identity)
        detections = []
        annotations = []


        for prediction, track in zip(predictions, tracks):
            class_name = prediction["class_name"]
            left, top, right, bottom = prediction["bbox_xyxy_px"]
            reading = None
            reading_status = None

            # Keep the target's position even when its pressure cannot be read.
            detection = {
                "class_name": class_name,
                "bbox_xyxy_px": (left, top, right, bottom),
                "track_id": track.track_id,
                "reading_status": None,
                "raw_gauge_reading": None,
                "gauge_reading": None,
            }

            if class_name == "gauge":

                # Read the supplied clean crop, without cropping the frame again.
                reading = read_gauge(
                    prediction["gauge_crop"],
                    calibration = self.gauge_calibration,
                    config = self.reader_config,
                    metadata = {
                        "coordinate_frame": "crop_pixels",
                        "crop_origin_px": [left, top],
                    },
                )

                # Retain this exact reader result before applying temporal checks.
                detection["raw_gauge_reading"] = reading_record(reading)
                value = reading.value_bar
                finite_pressure = (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and isfinite(value)
                )
                reading_status = self.tracker.confirm(
                    track,
                    value if reading.valid and finite_pressure else None,
                )

                if reading.valid and reading_status != "confirmed":
                    # Preserve this frame's geometry, but never an old pressure.
                    reason = (
                        StreamReadingFailure.READING_UNCONFIRMED
                        if finite_pressure else StreamReadingFailure.INVALID_READING
                    )
                    reading = replace(
                        reading,
                        valid=False,
                        value_bar=None,
                        failure_reason=reason,
                    )

                detection["reading_status"] = reading_status
                detection["gauge_reading"] = reading_record(reading)

            detections.append(detection)

            # Keep the reading object and normalized box for the existing renderers.
            # These drawing instructions are not included in saved JSON records.
            annotations.append({
                "class_name": class_name,
                "box_xywhn": prediction["box_xywhn"],
                "bbox_xyxy_px": (left, top, right, bottom),
                "gauge_reading": reading,
                "reading_status": reading_status,
            })

        # Tie every target to the frame and timestamps that produced it.
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
        """Draw saved target results without running detection or reading again."""

        labels = []

        for annotation in result.annotations:
            label = annotation["class_name"]
            reading = annotation["gauge_reading"]

            if reading is not None:
                # Show the pressure when accepted, or explain that it is unavailable.
                if reading.valid:
                    label = f"{label} | {reading.value_bar:.1f} bar"

                elif annotation["reading_status"] == "pending":
                    label = f"{label} | pressure awaiting confirmation"

                else:
                    label = f"{label} | pressure unavailable"

            labels.append((label, annotation["box_xywhn"]))

        # Aidan's renderer copies the supplied overlay and adds every target box.
        # Put labels on the full frame so a small crop does not cut the text off.
        overlay = self.predictor.annotate_image(overlay, labels)

        for annotation in result.annotations:
            reading = annotation["gauge_reading"]

            if reading is None:
                # Valves have a target box but no gauge geometry to draw.
                continue

            left, top, right, bottom = annotation["bbox_xyxy_px"]
            gauge_region = overlay[top:bottom, left:right]

            if gauge_region.size == 0:
                # Retain the target label when an empty crop has no pixels to draw on.
                continue

            # Draw on the current display region so earlier annotations are retained.
            # The pressure is already in the box label, so only add pivot and needle.
            overlay[top:bottom, left:right] = render_overlay(
                gauge_region,
                reading,
                show_label = False,
            )

        return overlay
