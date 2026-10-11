"""Tests for the dependency-independent local stream controller."""

import unittest

import numpy as np

from stream_controller import run_stream
from stream_types import FramePacket


class SequenceFrameSource:
    """Small deterministic source used to exercise the real controller."""

    def __init__(self, packets):
        self._packets = iter(packets)
        self.closed = False

    def read(self):
        """Return the next prepared packet."""

        return next(self._packets)

    def close(self):
        """Record that the controller has released the source."""

        self.closed = True


class StopAfterTwoFramesDisplay:
    """Record packets and ask the controller to stop after two frames."""

    def __init__(self):
        self.sequence_numbers = []
        self.closed = False

    def show(self, packet):
        """Return False after the second displayed frame."""

        self.sequence_numbers.append(packet.sequence_number)
        return len(self.sequence_numbers) < 2

    def close(self):
        """Record that the controller has released the display."""

        self.closed = True


class FailingDisplay:
    """Simulate a graphical display error during a stream."""

    def __init__(self):
        self.closed = False

    def show(self, packet):
        """Fail as an OpenCV operation could fail at runtime."""

        raise RuntimeError("display failed")

    def close(self):
        """Record whether cleanup occurred after the failure."""

        self.closed = True


class FailingCloseDisplay:
    """Simulate failure while releasing the graphical window."""

    def __init__(self):
        self.close_attempted = False

    def show(self, packet):
        """Stop normally after displaying one frame."""

        return False

    def close(self):
        """Record the attempt, then simulate an OpenCV cleanup failure."""

        self.close_attempted = True
        raise RuntimeError("display cleanup failed")


class StreamControllerTests(unittest.TestCase):
    """Define how frame sources, displays and diagnostics are composed."""

    def test_frame_packets_reach_display_and_diagnostic_summary(self):
        """The controller must display received packets and retain evidence."""

        # Prepare two small BGR frames with known sequence numbers and arrival times.
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        source = SequenceFrameSource(
            [
                FramePacket(
                    image=image.copy(),
                    sequence_number=100,
                    device_timestamp_s=0.980,
                    host_received_timestamp_s=1.000,
                ),
                FramePacket(
                    image=image.copy(),
                    sequence_number=101,
                    device_timestamp_s=1.080,
                    host_received_timestamp_s=1.100,
                ),
            ]
        )
        display = StopAfterTwoFramesDisplay()

        summary = run_stream(
            source,
            display,
            requested_fps=30.0,
        )

        self.assertEqual(display.sequence_numbers, [100, 101])
        self.assertEqual(summary.frames_received, 2)

        # Check that the summary records the supplied images' three-by-two size.
        self.assertEqual(summary.frame_width, 3)
        self.assertEqual(summary.frame_height, 2)

        self.assertEqual(summary.sequence_gaps, 0)
        self.assertAlmostEqual(summary.received_fps, 10.0)

    def test_resources_close_when_display_raises(self):
        """Camera and display resources must close on an exceptional exit."""

        image = np.zeros((2, 3, 3), dtype=np.uint8)

        source = SequenceFrameSource(
            [
                FramePacket(
                    image=image,
                    sequence_number=200,
                    device_timestamp_s=2.980,
                    host_received_timestamp_s=3.000,
                ),
            ]
        )
        display = FailingDisplay()

        # Make display fail, then check that its exception is still reported.
        with self.assertRaisesRegex(
            RuntimeError,
            "display failed",
        ):
            run_stream(
                source,
                display,
                requested_fps=30.0,
            )

        self.assertTrue(source.closed)
        self.assertTrue(display.closed)

    def test_source_closes_when_display_cleanup_raises(self):
        """Camera cleanup must run even when display cleanup fails."""

        image = np.zeros(
            (2, 3, 3),
            dtype=np.uint8,
        )

        source = SequenceFrameSource(
            [
                FramePacket(
                    image=image,
                    sequence_number=300,
                    device_timestamp_s=3.980,
                    host_received_timestamp_s=4.000,
                ),
            ]
        )
        display = FailingCloseDisplay()

        # Make display cleanup fail, then check that source cleanup still occurs.
        with self.assertRaisesRegex(
            RuntimeError,
            "display cleanup failed",
        ):
            run_stream(
                source,
                display,
                requested_fps=30.0,
            )

        self.assertTrue(display.close_attempted)
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
