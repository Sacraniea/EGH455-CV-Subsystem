"""Connect ArUco processing to the standalone live command's display."""

from dataclasses import replace

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
