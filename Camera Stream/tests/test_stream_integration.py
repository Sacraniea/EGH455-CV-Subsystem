import unittest
import numpy as np
from stream_controller import run_stream
from stream_diagnostics import FrameSample, OnlineDiagnostics
from stream_types import FramePacket
from test_stream_processing import Processor
from test_stream_controller import SequenceFrameSource


class Display:
    def __init__(self):
        # Retain the displayed packets and whether the controller closed the display.
        self.packets = []
        self.closed = False

    def show(self, packet):
        # Record this output frame and let the controller's frame limit stop the run.
        self.packets.append(packet)
        return True

    def close(self):
        self.closed = True


class StreamIntegrationTests(unittest.TestCase):
    def test_controller_composes_logs_and_stops_at_frame_limit(self):
        # Supply frames 1 and 3 so the summary must count missing frame 2.
        packets = [
            FramePacket(np.zeros((4, 6, 3), np.uint8), i, float(i), i + 0.02)
            for i in (1, 3)
        ]
        source, display = SequenceFrameSource(packets), Display()
        records = []
        # Process and draw both frames, saving each resulting record into this list.
        summary = run_stream(
            source,
            display,
            requested_fps=30.0,
            processors=[Processor("aruco")],
            max_frames=2,
            on_frame=records.append,
        )
        # Check the stop reason, missing frames, records and separate overlay.
        self.assertEqual(summary.stop_reason, "frame_limit")
        self.assertEqual(summary.sequence_gaps, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["sequence_number"], 3)
        self.assertEqual(records[1]["status"], "complete")
        self.assertEqual(records[1]["processors"][0]["processor"], "aruco")
        self.assertTrue(all(value >= 0 for value in records[1]["timings_ms"].values()))
        self.assertTrue(np.all(packets[0].image == 0))
        self.assertTrue(np.any(display.packets[0].image != 0))
        self.assertTrue(source.closed and display.closed)

    def test_branch_failure_is_logged_as_partial_and_stream_continues(self):
        # Fail one processor and check that the other still produces a logged result.
        def fail(packet):
            raise RuntimeError("inference failure")

        source = SequenceFrameSource(
            [FramePacket(np.zeros((4, 6, 3), np.uint8), 1, 1.0, 1.02)]
        )
        records = []
        run_stream(
            source,
            Display(),
            requested_fps=30,
            processors=[Processor("bad", fail), Processor("good")],
            max_frames=1,
            on_frame=records.append,
        )
        self.assertEqual(records[0]["status"], "partial")
        self.assertIsNone(records[0]["processors"][1]["failure_reason"])

    def test_log_failure_releases_source_and_display(self):
        # Fail while saving a record and check that the source and display still close.
        def fail(record):
            raise OSError("disk full")

        source = SequenceFrameSource(
            [FramePacket(np.zeros((4, 6, 3), np.uint8), 1, 1.0, 1.02)]
        )
        display = Display()

        with self.assertRaisesRegex(OSError, "disk full"):
            run_stream(source, display, requested_fps=30, max_frames=1, on_frame=fail)

        self.assertTrue(source.closed and display.closed)

    def test_online_diagnostics_bounds_samples_but_keeps_all_run_counts(self):
        # Feed 100 increasing delays while retaining only three for the percentile.
        stats = OnlineDiagnostics(30.0, window_size=3)

        for i in range(100):
            stats.add(FrameSample(i * 2, float(i), i + 0.01 * (i + 1)))

        # Check all-frame counts and mean alongside the percentile of the recent three.
        result = stats.summary()
        self.assertEqual(result.frames_received, 100)
        self.assertEqual(result.sequence_gaps, 99)
        self.assertEqual(result.latency_p95_sample_count, 3)
        self.assertAlmostEqual(result.latency_mean_ms, 505.0)
        self.assertAlmostEqual(result.latency_p95_ms, 1000.0)
        self.assertEqual(len(stats.latencies_ms), 3)

    def test_invalid_frame_limit_closes_resources_without_reading(self):
        # Reject invalid frame counts and check that even unused resources are closed.
        for limit in (0, -1, 1.5, True):
            source, display = SequenceFrameSource([]), Display()

            with self.subTest(limit=limit), self.assertRaises(ValueError):
                run_stream(source, display, requested_fps=30, max_frames=limit)

            self.assertTrue(source.closed and display.closed)
