"""Give each processor the same frame and combine its results for display."""

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from stream_types import FramePacket


@dataclass(frozen=True)
class FrameIdentity:
    sequence_number: int
    device_timestamp_s: float
    host_received_timestamp_s: float
    image_size_px: tuple[int, int]

    @classmethod
    def from_packet(cls, packet):
        # Collect this frame's sequence, timestamps and dimensions as one identity.
        # Reverse the array's height/width order to record width before height.
        return cls(
            packet.sequence_number,
            packet.device_timestamp_s,
            packet.host_received_timestamp_s,
            (packet.image.shape[1], packet.image.shape[0]),
        )


@dataclass(frozen=True)
class ProcessingResult:
    """One processor's measurements and drawing instructions for one frame."""

    processor: str
    identity: FrameIdentity
    # Store the measurements that will be written to JSON and sent over MQTT.
    data: dict
    # Store objects needed for drawing; to_record leaves these out of the JSON.
    annotations: tuple[Any, ...] = field(default=(), repr=False)
    failure_reason: str | None = None
    error: str | None = None
    processing_time_ms: float = 0.0

    def to_record(self):
        # Build the saved result from measurements, frame identity and failure details.
        return {
            "processor": self.processor,
            "identity": asdict(self.identity),
            "data": self.data,
            "failure_reason": self.failure_reason,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
        }


class FrameProcessor(Protocol):
    """Interface for a detector or reader added to the camera stream."""

    # Match results to the processor that produced them using this name.
    name: str

    def process(self, packet: FramePacket) -> ProcessingResult:
        ...

    def draw(self, overlay: np.ndarray, result: ProcessingResult) -> np.ndarray:
        ...


def validate_processors(processors):
    # Collect processor names and reject missing, empty or repeated names.
    names = [p.name for p in processors]

    if (
        any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Processors must have unique, nonempty names")


def process_frame(packet, processors, *, max_processing_ms=None, clock=perf_counter):
    """Run processors one at a time, keeping each failure in the results."""

    # Freeze the processor list for this frame and check each processor's name.
    processors = tuple(processors)
    validate_processors(processors)

    # Reject a processing limit that is zero, negative, infinite or undefined.
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

    # Save the image's current write setting, then prevent edits during processing.
    writable = packet.image.flags.writeable
    packet.image.flags.writeable = False

    try:
        for processor in processors:
            # Time this processor as it reads the image and returns a result.
            started = clock()

            try:
                result = processor.process(packet)

                if not isinstance(result, ProcessingResult):
                    # Reject returns that do not contain the shared result fields.
                    raise TypeError("Processor must return ProcessingResult")

                if (
                    result.identity != identity
                    or result.processor != processor.name
                ):
                    # Reject a result from a different frame or processor.
                    result = ProcessingResult(
                        processor.name,
                        identity,
                        {},
                        failure_reason="FRAME_IDENTITY_MISMATCH",
                    )

            except Exception as error:
                # Record PROCESSOR_ERROR and discard this processor's measurements.
                result = ProcessingResult(
                    processor.name,
                    identity,
                    {},
                    failure_reason="PROCESSOR_ERROR",
                    error=f"{type(error).__name__}: {error}",
                )

            # Convert this processor's elapsed time from seconds to milliseconds.
            elapsed = (clock() - started) * 1000.0

            if (
                max_processing_ms is not None
                and elapsed > max_processing_ms
            ):
                # Discard measurements if this processor exceeded its time limit.
                result = ProcessingResult(
                    processor.name,
                    identity,
                    {},
                    failure_reason="PROCESSING_DEADLINE_EXCEEDED",
                )

            results.append(replace(result, processing_time_ms=elapsed))

    finally:
        # Restore the image's original write setting, including after an exception.
        packet.image.flags.writeable = writable

    return tuple(results)


class ResultCompositor:
    """Draw accepted results on a copy of the original frame."""

    def __init__(self, processors=()):
        # Check the names and build the lookup used to choose each result's renderer.
        processors = tuple(processors)
        validate_processors(processors)
        self._processors = {p.name: p for p in processors}

    def compose(self, packet, results):
        # Draw the supplied results on a copy of this packet's original image.
        identity = FrameIdentity.from_packet(packet)

        # Keep the original pixels unchanged while building the displayed image.
        overlay = packet.image.copy()

        for result in results:
            if result.identity != identity:
                # Stop if these measurements were produced from a different frame.
                raise ValueError("Result identity does not match composition frame")

            if result.failure_reason is None:
                # Ask the matching processor to draw this successful result.
                overlay = self._processors[result.processor].draw(overlay, result)

                if (
                    not isinstance(overlay, np.ndarray)
                    or overlay.shape != packet.image.shape
                    or overlay.dtype != packet.image.dtype
                    or np.shares_memory(overlay, packet.image)
                ):
                    # Reject changed image size, pixel type or reused source memory.
                    raise ValueError(
                        "Renderer must return an independent, same-size image"
                    )

        return replace(packet, image=overlay)
