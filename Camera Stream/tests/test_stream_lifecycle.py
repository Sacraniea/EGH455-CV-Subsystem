import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from dataclasses import replace
import numpy as np
import depthai as dai
from depthai_source import DepthAIFrameSource
from stream_types import get_stream_profile, FramePacket, DeviceDetails
from stream_controller import run_stream
from stream_records import build_complete_run_record
from stream_diagnostics import summarize_frame_samples, FrameSample
from test_depthai_source import RecordingDevice
from test_stream_controller import SequenceFrameSource, FailingCloseDisplay
from local_display import OpenCVDisplay
from test_local_display import RecordingCv2


class LifecycleTests(unittest.TestCase):
    def test_source_timeout_does_not_wait_forever_or_fabricate_frame(self):
        # Make the camera queue stay empty and set a 0.1-second read limit.
        device = RecordingDevice(None, None)
        device.output_queue.tryGet = lambda: None
        source = DepthAIFrameSource(
            get_stream_profile("720p30"),
            device_factory=lambda p: device,
            read_timeout_s=0.1,
        )

        try:
            # Advance the clock past the deadline and check that reading times out.
            with patch("depthai_source.monotonic", side_effect=[0.0, 0.2]), patch(
                "depthai_source.sleep"
            ):
                with self.assertRaises(TimeoutError):
                    source.read()

        finally:
            source.close()

        self.assertTrue(device.closed)

    def test_timeout_source_returns_real_message_when_available(self):
        # Put a real DepthAI image message in the queue and check its converted size.
        message = dai.ImgFrame()
        message.setWidth(2)
        message.setHeight(2)
        message.setType(dai.ImgFrame.Type.BGR888i)
        message.setFrame(np.zeros((2, 2, 3), np.uint8))
        device = RecordingDevice(None, message)
        device.output_queue.tryGet = lambda: message
        source = DepthAIFrameSource(
            get_stream_profile("720p30"),
            device_factory=lambda p: device,
            read_timeout_s=0.1,
            clock_now=lambda: 1.0,
        )

        try:
            self.assertEqual(source.read().image.shape, (2, 2, 3))

        finally:
            source.close()

    def test_primary_failure_survives_cleanup_failure(self):
        # Fail the camera read and display close, then check the camera error survives.
        source = SequenceFrameSource([])

        def fail():
            raise RuntimeError("camera failed first")

        source.read = fail

        with self.assertRaisesRegex(RuntimeError, "camera failed first"):
            run_stream(source, FailingCloseDisplay(), requested_fps=30.0)

        self.assertTrue(source.closed)

    def test_close_before_first_frame_does_not_destroy_nonexistent_window(self):
        # Close before the first frame and check that no window is destroyed.
        boundary = RecordingCv2([])
        display = OpenCVDisplay(window_name="test", cv2_module=boundary)
        display.close()
        self.assertEqual(boundary.destroyed_windows, [])

    def test_window_close_button_stops_stream(self):
        # Report a closed preview window and check that streaming stops and closes once.
        boundary = RecordingCv2([-1])
        boundary.visible = 0
        display = OpenCVDisplay(window_name="test", cv2_module=boundary)
        packet = FramePacket(np.zeros((4, 6, 3), np.uint8), 1, 1.0, 1.02)
        self.assertFalse(display.show(packet))
        display.close()
        display.close()
        self.assertEqual(boundary.destroyed_windows, ["test"])

    def test_acquisition_record_names_stop_reason_and_percentile_window(self):
        # Save one frame's summary and check its stop reason and recent-sample count.
        summary = replace(
            summarize_frame_samples([FrameSample(1, 1.0, 1.02)], 30),
            stop_reason="frame_limit",
            latency_p95_sample_count=1,
            latency_p95_window_size=2048,
        )
        now = datetime.now(timezone.utc)
        record = build_complete_run_record(
            profile=get_stream_profile("720p30"),
            device=DeviceDetails("oak", "id", "SUPER"),
            summary=summary,
            started_at_utc=now,
            finished_at_utc=now,
        )
        self.assertEqual(record["stop_reason"], "frame_limit")
        self.assertEqual(record["measurement"]["latency_p95_window_size"], 2048)
        self.assertEqual(record["measurement"]["latency_p95_sample_count"], 1)
