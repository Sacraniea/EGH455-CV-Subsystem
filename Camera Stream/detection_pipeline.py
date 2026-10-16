"""Run the detectors on one frame and draw their results on a separate image."""

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from stream_types import FramePacket


@dataclass(frozen=True)
class FrameIdentity:
    """Keep the frame number, timestamps and dimensions used to match a result."""

    sequence_number: int
    device_timestamp_s: float
    host_received_timestamp_s: float
    image_size_px: tuple[int, int]

    @classmethod
    def from_packet(cls, packet):
        # Copy the source frame's identifying information into its result.
        # Image arrays store height first; our records use width first.
        return cls(
            sequence_number = packet.sequence_number,
            device_timestamp_s = packet.device_timestamp_s,
            host_received_timestamp_s = packet.host_received_timestamp_s,
            image_size_px = (packet.image.shape[1], packet.image.shape[0]),
        )


@dataclass(frozen=True)
class ProcessingResult:
    """Hold one detector's measurements, drawing information and processing status."""

    # Identify the detector and the source frame that produced these results.
    processor: str
    identity: FrameIdentity

    # Keep measurements in dictionaries that can be included in saved records.
    data: dict

    # Keep the objects needed for drawing without putting them in the saved record.
    annotations: tuple[Any, ...] = field(default=(), repr=False)

    # Record a detector-level failure separately from individual unreadable gauges.
    failure_reason: str | None = None
    error: str | None = None
    processing_time_ms: float = 0.0

    def to_record(self):
        # Package measurements, frame identity and any processing error for recording.
        return {
            "processor": self.processor,
            "identity": asdict(self.identity),
            "data": self.data,
            "failure_reason": self.failure_reason,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
        }


class FrameProcessor(Protocol):
    """Describe the name and two methods that each detector adapter supplies."""

    name: str

    # Processing reads the source frame and returns measurements and drawing details.
    def process(self, packet: FramePacket) -> ProcessingResult:
        ...

    # Drawing uses those saved results to add annotations to the display image.
    def draw(self, overlay: np.ndarray, result: ProcessingResult) -> np.ndarray:
        ...


def validate_processors(processors):
    """Check that every detector can be identified by a different, nonempty name."""

    names = [processor.name for processor in processors]

    # A missing name cannot identify a result; repeated names make drawing ambiguous.
    if (
        any(
            not isinstance(name, str) or not name
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("Processors must have unique, nonempty names")


def process_frame(
    packet,
    processors,
    *,
    max_processing_ms = None,
    clock = perf_counter,
):
    """Run each detector and retain its result or failure against this frame."""

    # Keep the same detector list for this frame and check that names are unique.
    processors = tuple(processors)
    validate_processors(processors)

    # If a time limit was supplied, require a finite number greater than zero.
    if (
        max_processing_ms is not None
        and (
            not isfinite(max_processing_ms)
            or max_processing_ms <= 0
        )
    ):
        raise ValueError("max_processing_ms must be finite and positive")

    identity = FrameIdentity.from_packet(packet)
    results = []

    # Prevent detectors from drawing on the original frame while they process it.
    # Remember its previous setting so we can restore it afterwards.
    original_writeable = packet.image.flags.writeable
    packet.image.flags.writeable = False

    try:
        for processor in processors:
            # Measure how long this detector takes to produce its result.
            started = clock()

            try:
                result = processor.process(packet)

                # Require the shared result structure before reading its fields.
                if not isinstance(result, ProcessingResult):
                    raise TypeError("Processor must return ProcessingResult")

                # Reject measurements labelled with another frame or detector.
                if (
                    result.identity != identity
                    or result.processor != processor.name
                ):
                    result = ProcessingResult(
                        processor = processor.name,
                        identity = identity,
                        data = {},
                        failure_reason = "FRAME_IDENTITY_MISMATCH",
                    )

            except Exception as error:
                # Keep the failed detector's error and allow the next detector to run.
                result = ProcessingResult(
                    processor = processor.name,
                    identity = identity,
                    data = {},
                    failure_reason = "PROCESSOR_ERROR",
                    error = f"{type(error).__name__}: {error}",
                )

            elapsed_ms = (clock() - started) * 1000.0

            # Check an optional time limit after processing finishes.
            # This withholds late measurements; it does not interrupt the detector.
            if (
                max_processing_ms is not None
                and elapsed_ms > max_processing_ms
            ):
                result = ProcessingResult(
                    processor = processor.name,
                    identity = identity,
                    data = {},
                    failure_reason = "PROCESSING_DEADLINE_EXCEEDED",
                )

            # Include the elapsed time for both successful and failed results.
            results.append(
                replace(result, processing_time_ms = elapsed_ms)
            )

    finally:
        # Restore the original image setting even if processing is interrupted.
        packet.image.flags.writeable = original_writeable

    return tuple(results)


class ResultCompositor:
    """Combine the detectors' annotations on a copy of the source frame."""

    def __init__(self, processors = ()):
        processors = tuple(processors)
        validate_processors(processors)

        # Use each result's detector name to find the method that draws it.
        self._processors = {}

        for processor in processors:
            self._processors[processor.name] = processor

    def compose(self, packet, results):
        identity = FrameIdentity.from_packet(packet)

        # Draw on a separate image so the original frame remains available as evidence.
        overlay = packet.image.copy()

        for result in results:
            # Stop if these annotations belong to a different source frame.
            if result.identity != identity:
                raise ValueError("Result identity does not match composition frame")

            # Draw completed detector results, including any unreadable-gauge labels.
            # A detector-level failure has no accepted annotations to draw.
            if result.failure_reason is None:
                processor = self._processors[result.processor]
                overlay = processor.draw(overlay, result)

                # Require a separate image with the same dimensions and pixel type.
                if (
                    not isinstance(overlay, np.ndarray)
                    or overlay.shape != packet.image.shape
                    or overlay.dtype != packet.image.dtype
                    or np.shares_memory(overlay, packet.image)
                ):
                    raise ValueError(
                        "Renderer must return an independent, same-size image"
                    )

        # Replace only the pixels; keep the source frame's number and timestamps.
        return replace(packet, image = overlay)
