"""Tests for the DepthAI-specific frame-source boundary."""

import unittest

from datetime import timedelta
import depthai as dai

from depthai_source import (
    DepthAIFrameSource,
    build_color_pipeline,
    frame_packet_from_message,
)
import numpy as np
from stream_types import get_stream_profile


class PreparedOutputQueue:
    """Return one prepared real DepthAI message."""

    def __init__(self, message):
        self._message = message

    def get(self):
        """Behave like the blocking DepthAI queue boundary."""

        return self._message


class RecordingDevice:
    """Record how the source configures and releases a DepthAI device."""

    def __init__(self, pipeline, message):
        self.pipeline = pipeline
        self.output_queue = PreparedOutputQueue(message)
        self.queue_request = None
        self.closed = False

    def getOutputQueue(self, *, name, maxSize, blocking):
        """Record the requested host-queue behaviour."""

        self.queue_request = {
            "name": name,
            "max_size": maxSize,
            "blocking": blocking,
        }
        return self.output_queue

    def getProductName(self):
        return "OAK-D-LITE"

    def getMxId(self):
        return "TEST-MXID"

    def getUsbSpeed(self):
        return dai.UsbSpeed.SUPER

    def close(self):
        """Record that the source released the device."""

        self.closed = True


class DepthAIPipelineTests(unittest.TestCase):
    """Verify the actual pipeline sent to the OAK-D Lite."""

    def test_pipeline_uses_profile_settings_and_rgb_video_output(self):
        """Both profiles must configure CAM_A and expose its video stream."""

        for profile_name in ("720p30", "1080p30"):
            with self.subTest(profile=profile_name):
                profile = get_stream_profile(profile_name)
                pipeline = build_color_pipeline(profile)

                # Read the configured pipeline and look up its camera and output nodes.
                pipeline_data = pipeline.serializeToJson()["pipeline"]
                nodes = {
                    node_data["name"]: node_data
                    for _, node_data in pipeline_data["nodes"]
                }

                color_camera = nodes["ColorCamera"]
                camera_properties = color_camera["properties"]

                # Check the camera socket, sensor size and selected output profile.
                self.assertEqual(
                    camera_properties["boardSocket"],
                    dai.CameraBoardSocket.CAM_A.value,
                )
                self.assertEqual(
                    camera_properties["resolution"],
                    dai.ColorCameraProperties.SensorResolution.THE_1080_P.value,
                )
                self.assertEqual(
                    camera_properties["videoWidth"],
                    profile.width,
                )
                self.assertEqual(
                    camera_properties["videoHeight"],
                    profile.height,
                )
                self.assertEqual(
                    camera_properties["fps"],
                    profile.fps,
                )

                xlink = nodes["XLinkOut"]
                self.assertEqual(
                    xlink["properties"]["streamName"],
                    "rgb",
                )

                # Check that the sole connection links camera video to the host output.
                self.assertEqual(
                    len(pipeline_data["connections"]),
                    1,
                )
                connection = pipeline_data["connections"][0]
                self.assertEqual(
                    connection["node1Output"],
                    "video",
                )
                self.assertEqual(
                    connection["node2Input"],
                    "in",
                )

    def test_depthai_message_becomes_bgr_frame_packet(self):
        """Pixels and timing evidence must remain associated during conversion."""

        image = np.arange(
            18,
            dtype=np.uint8,
        ).reshape((2, 3, 3))

        # Put these pixels, frame number and capture time into a real DepthAI message.
        message = dai.ImgFrame()
        message.setWidth(3)
        message.setHeight(2)
        message.setType(dai.ImgFrame.Type.BGR888i)
        message.setFrame(image)
        message.setSequenceNum(73)
        message.setTimestamp(timedelta(seconds=1.250))

        packet = frame_packet_from_message(
            message,
            host_received_timestamp_s=1.275,
        )

        # Check conversion preserved the pixels, capture time and host arrival time.
        np.testing.assert_array_equal(
            packet.image,
            image,
        )
        self.assertEqual(
            packet.sequence_number,
            73,
        )
        self.assertEqual(
            packet.device_timestamp_s,
            1.250,
        )
        self.assertEqual(
            packet.host_received_timestamp_s,
            1.275,
        )

    def test_source_reads_packet_and_exposes_device_diagnostics(self):
        """The source must configure, identify, read and release its device."""

        image = np.full(
            (2, 3, 3),
            fill_value=120,
            dtype=np.uint8,
        )
        message = dai.ImgFrame()
        message.setWidth(3)
        message.setHeight(2)
        message.setType(dai.ImgFrame.Type.BGR888i)
        message.setFrame(image)
        message.setSequenceNum(88)
        message.setTimestamp(timedelta(seconds=5.000))

        created_devices = []

        def device_factory(pipeline):
            # Keep the constructed device so the test can inspect setup and cleanup.
            device = RecordingDevice(
                pipeline,
                message,
            )
            created_devices.append(device)
            return device

        source = DepthAIFrameSource(
            get_stream_profile("720p30"),
            device_factory=device_factory,
            clock_now=lambda: 5.025,
        )

        device = created_devices[0]

        # Check the queue limit and device details returned by the source.
        self.assertEqual(
            device.queue_request,
            {
                "name": "rgb",
                "max_size": 4,
                "blocking": False,
            },
        )
        self.assertEqual(
            source.device_details.product_name,
            "OAK-D-LITE",
        )
        self.assertEqual(
            source.device_details.mxid,
            "TEST-MXID",
        )
        self.assertEqual(
            source.device_details.usb_speed,
            "SUPER",
        )

        # Read the prepared message and compare its pixels and identity with the input.
        packet = source.read()

        np.testing.assert_array_equal(
            packet.image,
            image,
        )
        self.assertEqual(packet.sequence_number, 88)
        self.assertEqual(
            packet.host_received_timestamp_s,
            5.025,
        )

        source.close()

        self.assertTrue(device.closed)

    def test_device_closes_when_output_queue_setup_raises(self):
        """A partially initialized source must release its opened device."""

        class QueueFailureDevice:
            """Represent hardware that opens but cannot create its output queue."""

            def __init__(self):
                self.closed = False

            def getOutputQueue(self, *, name, maxSize, blocking):
                # Simulate failure while creating a queue on an already opened device.
                raise RuntimeError("queue setup failed")

            def close(self):
                """Record release of the simulated physical device."""

                self.closed = True

        created_devices = []

        def device_factory(pipeline):
            device = QueueFailureDevice()
            created_devices.append(device)
            return device

        with self.assertRaisesRegex(
            RuntimeError,
            "queue setup failed",
        ):
            DepthAIFrameSource(
                get_stream_profile("720p30"),
                device_factory=device_factory,
            )

        # Check that constructor failure closed the one device it had opened.
        self.assertEqual(len(created_devices), 1)
        self.assertTrue(created_devices[0].closed)


if __name__ == "__main__":
    unittest.main()
