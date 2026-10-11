"""Read, process and display one frame at a time, then release the source."""

from dataclasses import replace
from time import perf_counter
import sys
from typing import Protocol

from stream_diagnostics import FrameSample, StreamSummary, OnlineDiagnostics
from stream_processing import process_frame, ResultCompositor, validate_processors
from stream_types import FramePacket


class FrameSource(Protocol):
    def read(self) -> FramePacket:
        ...

    def close(self) -> None:
        ...


class FrameDisplay(Protocol):
    def show(self, packet: FramePacket) -> bool:
        ...

    def close(self) -> None:
        ...


def run_stream(
    source: FrameSource,
    display: FrameDisplay,
    *,
    requested_fps: float,
    processors=(),
    compositor=None,
    on_frame=None,
    max_frames=None,
    max_processing_ms=None,
    on_output=None,
    on_source=None,
    on_capture=None,
) -> StreamSummary:
    """Run the frame loop and always close the display and source."""

    try:
        # Reject a frame limit that is not a positive whole number.
        if (
            max_frames is not None
            and (
                type(max_frames) is not int
                or max_frames <= 0
            )
        ):
            raise ValueError("max_frames must be a positive integer")

        # Check the processors and prepare drawing and timing for this run.
        processors = tuple(processors)
        validate_processors(processors)
        compositor = (
            compositor if compositor is not None else ResultCompositor(processors)
        )
        diagnostics = OnlineDiagnostics(requested_fps)
        frame_width = frame_height = None
        stop_reason = "operator_q"

        while True:
            # Read one image with its frame number and capture timestamps.
            started = perf_counter()
            packet = source.read()
            read_finished = perf_counter()

            if frame_width is None:
                # Record the first image's dimensions for the run summary.
                frame_height, frame_width = packet.image.shape[:2]

            # Add this frame's arrival time and sequence number to the totals.
            diagnostics.add(
                FrameSample(
                    packet.sequence_number,
                    packet.device_timestamp_s,
                    packet.host_received_timestamp_s,
                )
            )

            # Run each processor and collect its measurements or failure reason.
            results = process_frame(
                packet,
                processors,
                max_processing_ms=max_processing_ms,
            )
            processing_finished = perf_counter()

            # Draw the results on a copy, then show that image and read the stop flag.
            output = compositor.compose(packet, results)
            composition_finished = perf_counter()

            should_continue = display.show(output)
            display_finished = perf_counter()

            if (
                on_frame is not None
                or on_output is not None
                or on_source is not None
                or on_capture is not None
            ):
                # Build one record with frame identity, results and stage durations.
                # Convert the measured seconds to milliseconds for the saved timings.
                record = {
                    "schema_version": 1,
                    "producer": "camera_stream",
                    "sequence_number": packet.sequence_number,
                    "device_timestamp_s": packet.device_timestamp_s,
                    "host_received_timestamp_s": packet.host_received_timestamp_s,
                    "image_size_px": [packet.image.shape[1], packet.image.shape[0]],
                    "coordinates": "source_px",
                    "status": (
                        "partial"
                        if any(r.failure_reason for r in results)
                        else "complete"
                    ),
                    "processors": [r.to_record() for r in results],
                    "capture_to_dequeue_ms": (
                        packet.host_received_timestamp_s - packet.device_timestamp_s
                    )
                    * 1000.0,
                    "timings_ms": {
                        "source_read": (read_finished - started) * 1000.0,
                        "processing": (processing_finished - read_finished) * 1000.0,
                        "composition": (composition_finished - processing_finished)
                        * 1000.0,
                        "display": (display_finished - composition_finished) * 1000.0,
                        "iteration_before_logging": (display_finished - started)
                        * 1000.0,
                    },
                }

                # Finish these callbacks before reading another frame.
                if on_source is not None:
                    # Pass the original camera pixels to any source-only consumer.
                    on_source(packet, record)

                if on_capture is not None:
                    # Keep the original and its finished overlay together for saved photos.
                    on_capture(packet, output, record)

                if on_output is not None:
                    # Pass the image with drawings to the output callback.
                    on_output(output, record)

                if on_frame is not None:
                    # Save the record, including fields added by earlier callbacks.
                    on_frame(record)

            if not should_continue:
                # Stop when the display reports a close action or quit key.
                break

            if (
                max_frames is not None
                and diagnostics.count >= max_frames
            ):
                # Record that the requested number of frames has been processed.
                stop_reason = "frame_limit"
                break

        # Add the image dimensions and stop reason to the measured run totals.
        return replace(
            diagnostics.summary(),
            frame_width=frame_width,
            frame_height=frame_height,
            stop_reason=stop_reason,
        )

    finally:
        # Close the display and source even when reading or processing failed.
        active_error = sys.exception()
        cleanup_error = None

        for resource in (display, source):
            try:
                resource.close()

            except BaseException as error:
                if active_error is not None:
                    # Add the close failure to the exception already being raised.
                    active_error.add_note(
                        f"Cleanup also failed: {type(error).__name__}: {error}"
                    )

                elif cleanup_error is None:
                    # Keep the first close failure while still closing the source.
                    cleanup_error = error

        if cleanup_error is not None:
            raise cleanup_error
