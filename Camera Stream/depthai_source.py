"""DepthAI v2 adapter for acquiring colour frames from the OAK-D Lite."""

from __future__ import annotations

from math import isfinite
from time import monotonic, sleep

import depthai as dai

from stream_types import (
    DeviceDetails,
    FramePacket,
    StreamProfile,
)

from typing import Any, Callable


def build_color_pipeline(
    profile: StreamProfile,
) -> dai.Pipeline:
    """Build the OAK colour-camera pipeline for one approved profile."""

    pipeline = dai.Pipeline()

    # Configure the colour camera with this profile's frame rate and output size.
    color_camera = pipeline.createColorCamera()
    color_camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    color_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color_camera.setFps(profile.fps)
    color_camera.setVideoSize(
        profile.width,
        profile.height,
    )

    # Connect the camera video to the rgb output that sends frames to the host.
    xlink_output = pipeline.createXLinkOut()
    xlink_output.setStreamName("rgb")
    color_camera.video.link(xlink_output.input)

    return pipeline


def frame_packet_from_message(
    message: dai.ImgFrame,
    *,
    host_received_timestamp_s: float,
) -> FramePacket:
    """Convert a DepthAI message to a shared FramePacket."""

    # Read the message's frame number and convert its capture time to seconds.
    sequence_number = message.getSequenceNum()
    device_timestamp_s = message.getTimestamp().total_seconds()

    # Convert the message pixels to the BGR array used by OpenCV processors.
    image = message.getCvFrame()

    return FramePacket(
        image=image,
        sequence_number=sequence_number,
        device_timestamp_s=device_timestamp_s,
        host_received_timestamp_s=host_received_timestamp_s,
    )


class DepthAIFrameSource:
    """Own one OAK-D Lite device and its colour-frame output queue."""

    def __init__(
        self,
        profile: StreamProfile,
        *,
        device_factory: Callable[[dai.Pipeline], Any] | None = None,
        clock_now: Callable[[], float] | None = None,
        read_timeout_s: float | None = None,
    ) -> None:
        """Open the configured device and prepare its host output queue."""

        # Reject a supplied read timeout unless it is finite and greater than zero.
        if (
            read_timeout_s is not None
            and (
                not isfinite(read_timeout_s)
                or read_timeout_s <= 0
            )
        ):
            raise ValueError("read_timeout_s must be finite and positive")

        self._read_timeout_s = read_timeout_s
        pipeline = build_color_pipeline(profile)

        # Open the pipeline on a device, using the supplied factory when present.
        factory = dai.Device if device_factory is None else device_factory
        self._device = factory(pipeline)

        # Prepare the output queue and read the connected device's details.
        try:
            # Keep only the newest waiting frame while detection processes the last one.
            self._queue = self._device.getOutputQueue(
                name="rgb",
                maxSize=1,
                blocking=False,
            )

            # Use the DepthAI clock for host arrival times unless one was supplied.
            self._clock_now = (
                (lambda: dai.Clock.now().total_seconds())
                if clock_now is None
                else clock_now
            )

            # Read the USB speed name, accepting either an enum name or its text.
            usb_speed_value = self._device.getUsbSpeed()
            usb_speed_name = getattr(
                usb_speed_value,
                "name",
                str(usb_speed_value).rsplit(".", 1)[-1],
            )

            self.device_details = DeviceDetails(
                product_name=self._device.getProductName(),
                mxid=self._device.getMxId(),
                usb_speed=usb_speed_name,
            )

        except BaseException:
            # Close the device if queue or device-detail setup fails, then re-raise.
            self._device.close()
            raise

    def read(self) -> FramePacket:
        """Wait for the next message and convert it into the shared contract."""

        if self._read_timeout_s is None:
            # Wait until the output queue supplies a frame.
            message = self._queue.get()

        else:
            # Check the queue until a frame arrives or the read deadline expires.
            deadline = monotonic() + self._read_timeout_s

            while True:
                message = self._queue.tryGet()

                if message is not None:
                    break

                if monotonic() >= deadline:
                    raise TimeoutError(
                        f"No camera frame received within {self._read_timeout_s:g} s"
                    )

                # Wait five milliseconds before checking an empty queue again.
                sleep(0.005)

        # Record the host arrival time before converting the message's pixels.
        host_received_timestamp_s = self._clock_now()

        return frame_packet_from_message(
            message,
            host_received_timestamp_s=host_received_timestamp_s,
        )

    def close(self) -> None:
        """Release the DepthAI device and its USB connection."""

        self._device.close()
