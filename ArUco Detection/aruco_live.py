"""Connect ArUco processing to the standalone live command's display."""

from collections import deque
from dataclasses import replace
from math import ceil, isfinite

import cv2

from aruco_overlay import draw_aruco_overlay
from aruco_pose import validate_reprojection_limit
from aruco_processing import process_aruco_frame
from aruco_types import ArUcoConfig, CameraCalibration


class ArUcoDisplay:
    """Save results from the original frame and display an annotated copy."""

    def __init__(
        self,
        display,
        *,
        calibration: CameraCalibration | None,
        config: ArUcoConfig,
        max_reprojection_error_px: float,
        on_result=None,
        max_frames: int | None = None,
    ):
        # Validate the limits and retain the display, calibration and result callback.
        validate_reprojection_limit(max_reprojection_error_px)

        if (
            max_frames is not None
            and max_frames <= 0
        ):
            raise ValueError("Maximum frames must be positive")

        self._display = display
        self._calibration = calibration
        self._config = config
        self._max_error = max_reprojection_error_px
        self._on_result = on_result
        self._max_frames = max_frames
        self.frames_processed = 0
        self.stop_reason = "operator_q"

    def show(self, packet) -> bool:
        # Process one camera packet, display its overlay and return whether to continue.
        results, record = process_aruco_frame(
            packet,
            self._calibration,
            self._config,
            max_reprojection_error_px=self._max_error,
        )

        # Pass this frame's record to the callback before displaying the image.
        if self._on_result is not None:
            self._on_result(record)

        # Replace only the displayed image; keep the packet's capture metadata.
        overlay = draw_aruco_overlay(packet.image, results)
        should_continue = self._display.show(replace(packet, image=overlay))
        self.frames_processed += 1

        # Stop when the wrapped display requests it, recording the operator stop reason.
        if not should_continue:
            self.stop_reason = "operator_q"
            return False

        # Otherwise stop after the configured number of processed frames is reached.
        if (
            self._max_frames is not None
            and self.frames_processed >= self._max_frames
        ):
            self.stop_reason = "frame_limit"
            return False

        return True

    def close(self) -> None:
        self._display.close()


class OpenCVPreview:
    """Own only the standalone ArUco preview window, created on its first frame."""

    def __init__(self, *, window_name):
        self.window_name = window_name
        self._opened = False

    def show(self, packet):
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._opened = True
        cv2.imshow(self.window_name, packet.image)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q") and cv2.getWindowProperty(
            self.window_name, cv2.WND_PROP_VISIBLE
        ) >= 1

    def close(self):
        if self._opened:
            self._opened = False
            cv2.destroyWindow(self.window_name)


def run_live(source, display, *, requested_fps):
    """Read until the display stops; the caller owns source/display cleanup."""

    if not isfinite(requested_fps) or requested_fps <= 0:
        raise ValueError("requested_fps must be finite and positive")
    latencies = deque(maxlen=2048)
    count = gaps = 0
    latency_sum = 0.0
    first_received = last_received = last_sequence = None
    width = height = None

    while True:
        packet = source.read()
        received, captured = packet.host_received_timestamp_s, packet.device_timestamp_s
        if not all(isfinite(value) for value in (received, captured)):
            raise ValueError("Frame timestamps must be finite")
        if last_received is not None and received <= last_received:
            raise ValueError("Host timestamps must increase")
        if last_sequence is not None and packet.sequence_number <= last_sequence:
            raise ValueError("Frame sequence numbers must increase within a run")

        if first_received is None:
            first_received = received
            height, width = packet.image.shape[:2]
        if last_sequence is not None:
            gaps += packet.sequence_number - last_sequence - 1
        last_sequence, last_received = packet.sequence_number, received
        latency = (received - captured) * 1000.0
        latencies.append(latency)
        latency_sum += latency
        count += 1

        if not display.show(packet):
            break

    duration = last_received - first_received
    ordered = sorted(latencies)
    return {
        "requested_fps": requested_fps,
        "frames_received": count,
        "sequence_gaps": gaps,
        "duration_s": duration,
        "received_fps": (count - 1) / duration if count > 1 else None,
        "latency_mean_ms": latency_sum / count,
        "latency_p95_ms": ordered[ceil(.95 * len(ordered)) - 1],
        "frame_width": width,
        "frame_height": height,
        "latency_p95_sample_count": len(ordered),
        "latency_p95_window_size": latencies.maxlen,
        "stop_reason": getattr(display, "stop_reason", "operator_q"),
    }
