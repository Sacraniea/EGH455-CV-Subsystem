"""Deterministic calculations for camera-stream diagnostic evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Sequence


@dataclass(frozen=True)
class FrameSample:
    """Record timing and identity evidence for a single camera frame."""

    sequence_number: int
    device_timestamp_s: float
    host_received_timestamp_s: float


@dataclass(frozen=True)
class StreamSummary:
    """Contain the deterministic summary of a collection of frame samples."""

    requested_fps: float
    frames_received: int
    sequence_gaps: int
    duration_s: float | None
    received_fps: float | None
    latency_mean_ms: float | None
    latency_p95_ms: float | None
    frame_width: int | None = None
    frame_height: int | None = None
    latency_p95_sample_count: int | None = None
    latency_p95_window_size: int | None = None
    stop_reason: str = "operator_q"


def summarize_frame_samples(
    samples: Sequence[FrameSample],
    requested_fps: float,
) -> StreamSummary:
    """Calculate the received frame rate from the host-arrival timestamps."""

    # Freeze the samples so counts and timings all describe the same frames.
    samples = tuple(samples)
    count = len(samples)
    latencies_ms = sorted(
        (sample.host_received_timestamp_s - sample.device_timestamp_s) * 1000.0
        for sample in samples
    )
    duration_s = None

    if samples:
        # A single frame has zero duration; multiple frames need a positive span.
        duration_s = (
            samples[-1].host_received_timestamp_s
            - samples[0].host_received_timestamp_s
        )

        if (
            count > 1
            and duration_s <= 0.0
        ):
            raise ValueError("host timestamps must increase.")

    sequence_gaps = 0

    for previous, current in zip(samples, samples[1:]):
        # Count the missing frames, rather than just counting each gap once.
        missing_frames = current.sequence_number - previous.sequence_number - 1
        sequence_gaps += max(0, missing_frames)

    # Keep unavailable timings unset for an empty run or a single-frame FPS.
    return StreamSummary(
        requested_fps=requested_fps,
        frames_received=count,
        sequence_gaps=sequence_gaps,
        duration_s=duration_s,
        received_fps=(count - 1) / duration_s if count > 1 else None,
        latency_mean_ms=sum(latencies_ms) / count if count else None,
        latency_p95_ms=latencies_ms[ceil(0.95 * count) - 1] if count else None,
    )


class OnlineDiagnostics:
    """All-run counts/mean with a bounded, explicitly recent latency percentile."""

    def __init__(self, requested_fps, *, window_size=2048):
        # Prepare frame totals and a limited list of recent delays for this run.
        # Require a positive frame rate and a positive whole-number window size.
        if (
            not isfinite(requested_fps)
            or requested_fps <= 0
        ):
            raise ValueError("requested_fps must be finite and positive")

        if (
            type(window_size) is not int
            or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")

        self.requested_fps = requested_fps
        # Drop the oldest stored delay whenever this list reaches its size limit.
        self.latencies_ms = deque(maxlen=window_size)
        self.count = self.gaps = 0
        self.latency_sum = 0.0
        self.first = self.last = None

    def add(self, sample):
        # Check one frame's timing, then add its delay and sequence gaps to the run.
        # Reject timestamps that contain infinity or an undefined numeric value.
        if not all(
            isfinite(v)
            for v in (sample.device_timestamp_s, sample.host_received_timestamp_s)
        ):
            raise ValueError("Frame timestamps must be finite")

        if self.last is not None:
            # Reject a frame that arrives out of timestamp or sequence order.
            if (
                sample.host_received_timestamp_s
                <= self.last.host_received_timestamp_s
            ):
                raise ValueError("host timestamps must increase")

            if sample.sequence_number <= self.last.sequence_number:
                raise ValueError("Frame sequence numbers must increase within a run")

            # Count any frame numbers skipped since the previous sample.
            self.gaps += sample.sequence_number - self.last.sequence_number - 1

        # Store this delay in milliseconds and add it to the all-frame total.
        latency = (
            sample.host_received_timestamp_s - sample.device_timestamp_s
        ) * 1000.0
        self.latencies_ms.append(latency)
        self.latency_sum += latency
        self.count += 1

        if self.first is None:
            # Keep the first arrival so summary can calculate elapsed time later.
            self.first = sample

        self.last = sample

    def summary(self):
        # Return the run totals, all-frame mean and recent-frame 95th percentile.
        duration = (
            None
            if self.first is None
            else self.last.host_received_timestamp_s
            - self.first.host_received_timestamp_s
        )
        # Sort recent delays and select the value at the rounded-up 95-percent rank.
        ordered = sorted(self.latencies_ms)
        return StreamSummary(
            requested_fps=self.requested_fps,
            frames_received=self.count,
            sequence_gaps=self.gaps,
            duration_s=duration,
            received_fps=(self.count - 1) / duration if self.count > 1 else None,
            latency_mean_ms=self.latency_sum / self.count if self.count else None,
            latency_p95_ms=ordered[ceil(0.95 * len(ordered)) - 1] if ordered else None,
            latency_p95_sample_count=len(ordered),
            latency_p95_window_size=self.latencies_ms.maxlen,
        )
