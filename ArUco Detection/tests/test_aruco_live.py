import unittest

import cv2
import numpy as np

from stream_types import FramePacket
from aruco_live import ArUcoDisplay, run_live
from aruco_types import ArUcoConfig


class RecordingDisplay:
    """Capture displayed packets without opening a window."""

    def __init__(self):
        self.packets = []
        self.closed = False

    def show(self, packet):
        # Save the displayed packet for inspection and request an immediate stream stop.
        self.packets.append(packet)
        return False

    def close(self):
        self.closed = True


class OneFrameSource:
    """Supply one frame without opening a physical camera."""

    def __init__(self, packet):
        self._packets = iter((packet,))
        self.closed = False

    def read(self):
        return next(self._packets)

    def close(self):
        self.closed = True


class ArUcoDisplayTests(unittest.TestCase):

    def test_displays_overlay_preserving_source_and_frame_metadata(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)

        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        image[50:250, 50:250] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        original = image.copy()

        packet = FramePacket(
            image=image,
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
        )
        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )
        display = RecordingDisplay()
        adapter = ArUcoDisplay(
            display,
            calibration=None,
            config=config,
            max_reprojection_error_px=1.0,
        )

        should_continue = adapter.show(packet)

        # Check that the adapter returned the fake display's stop request.
        self.assertFalse(should_continue)
        self.assertEqual(len(display.packets), 1)
        shown = display.packets[0]

        self.assertIsNot(shown, packet)
        self.assertEqual(shown.sequence_number, 42)
        self.assertEqual(shown.device_timestamp_s, 12.5)
        self.assertEqual(shown.host_received_timestamp_s, 12.6)

        np.testing.assert_array_equal(packet.image, original)
        self.assertFalse(np.shares_memory(shown.image, packet.image))
        self.assertEqual(shown.image.shape, original.shape)

        # Compare pixels on the top marker edge to confirm that the outline was drawn.
        self.assertTrue(np.any(shown.image[50, 80:220] != original[50, 80:220]))

    def test_live_loop_stops_without_taking_resource_ownership(self):
        packet = FramePacket(
            image=np.full((300, 400, 3), 255, dtype=np.uint8),
            sequence_number=42,
            device_timestamp_s=12.5,
            host_received_timestamp_s=12.6,
        )
        source = OneFrameSource(packet)
        display = RecordingDisplay()

        adapter = ArUcoDisplay(
            display,
            calibration=None,
            config=ArUcoConfig(
                marker_side_length_m=0.2,
                dictionary_name="DICT_5X5_100",
                permitted_ids=frozenset({7}),
            ),
            max_reprojection_error_px=1.0,
        )

        summary = run_live(
            source,
            adapter,
            requested_fps=30.0,
        )

        self.assertEqual(summary["frames_received"], 1)
        self.assertEqual(len(display.packets), 1)
        self.assertFalse(display.closed)
        self.assertFalse(source.closed)


    def test_summary_keeps_all_run_counts_and_recent_latency_percentile(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        packets = iter(
            FramePacket(image, i * 2, i + 1.0 - (1.0 if i == 0 else .01), i + 1.0)
            for i in range(2050)
        )

        class Source:
            def read(self):
                return next(packets)

        class Display:
            stop_reason = "frame_limit"

            def show(self, packet):
                return packet.sequence_number < 4098

        summary = run_live(Source(), Display(), requested_fps=30)
        self.assertEqual(summary["frames_received"], 2050)
        self.assertEqual(summary["sequence_gaps"], 2049)
        self.assertEqual(summary["duration_s"], 2049)
        self.assertEqual(summary["received_fps"], 1)
        self.assertAlmostEqual(summary["latency_mean_ms"], 10 + 990 / 2050)
        self.assertAlmostEqual(summary["latency_p95_ms"], 10)
        self.assertEqual(summary["latency_p95_sample_count"], 2048)
        self.assertEqual(summary["latency_p95_window_size"], 2048)
        self.assertEqual((summary["frame_width"], summary["frame_height"]), (6, 4))
        self.assertEqual(summary["stop_reason"], "frame_limit")

    def test_repeated_sequence_and_invalid_timestamps_stop_the_loop(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        for second in (
            FramePacket(image, 1, 2.0, 2.1),
            FramePacket(image, 2, 1.0, 1.1),
            FramePacket(image, 2, float("nan"), 2.1),
        ):
            packets = iter((FramePacket(image, 1, 1.0, 1.1), second))

            class Source:
                def read(self):
                    return next(packets)

            class Display:
                def show(self, packet):
                    return True

            with self.subTest(second=second), self.assertRaises(ValueError):
                run_live(Source(), Display(), requested_fps=30)


if __name__ == "__main__":
    unittest.main()
