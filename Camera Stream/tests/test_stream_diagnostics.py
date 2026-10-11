"""Tests for deterministic camera stream diagnostic calculations."""

import unittest

from stream_diagnostics import FrameSample, summarize_frame_samples


class StreamDiagnosticsTests(unittest.TestCase):
    """Describe the evidence expected from a complete camera run."""

    def test_received_fps_uses_first_and_last_host_arrivals(self):
        """Received FPS must measure completed intervals between host arrivals."""

        # Supply four arrivals spanning 0.3 seconds, giving three intervals at 10 FPS.
        samples = [
            FrameSample(
                sequence_number=10,
                device_timestamp_s=99.980,
                host_received_timestamp_s=100.000,
            ),
            FrameSample(
                sequence_number=11,
                device_timestamp_s=100.000,
                host_received_timestamp_s=100.100,
            ),
            FrameSample(
                sequence_number=12,
                device_timestamp_s=100.180,
                host_received_timestamp_s=100.200,
            ),
            FrameSample(
                sequence_number=13,
                device_timestamp_s=100.280,
                host_received_timestamp_s=100.300,
            ),
        ]

        summary = summarize_frame_samples(
            samples,
            requested_fps=30.0,
        )

        self.assertEqual(summary.requested_fps, 30.0)
        self.assertEqual(summary.frames_received, 4)
        self.assertAlmostEqual(summary.duration_s, 0.3)
        self.assertAlmostEqual(summary.received_fps, 10.0)

    def test_fewer_than_two_frames_have_no_received_fps(self):
        """Zero or one frame cannot establish a frame-arrival rate."""

        # Summarize no frames and check that duration and frame rate stay unset.
        no_samples = summarize_frame_samples(
            [],
            requested_fps=30.0,
        )

        self.assertEqual(no_samples.frames_received, 0)
        self.assertIsNone(no_samples.duration_s)
        self.assertIsNone(no_samples.received_fps)

        # Summarize one frame and check for zero duration with no frame rate.
        one_sample = summarize_frame_samples(
            [
                FrameSample(
                    sequence_number=40,
                    device_timestamp_s=200.000,
                    host_received_timestamp_s=200.020,
                ),
            ],
            requested_fps=30.0,
        )

        self.assertEqual(one_sample.frames_received, 1)
        self.assertEqual(one_sample.duration_s, 0.0)
        self.assertIsNone(one_sample.received_fps)

    def test_non_increasing_host_timestamps_are_rejected(self):
        """Multiple frames require a positive host-observation duration."""

        invalid_cases = {
            "repeated timestamp": [
                FrameSample(
                    sequence_number=1,
                    device_timestamp_s=9.980,
                    host_received_timestamp_s=10.000,
                ),
                FrameSample(
                    sequence_number=2,
                    device_timestamp_s=9.990,
                    host_received_timestamp_s=10.000,
                ),
            ],
            "backwards timestamp": [
                FrameSample(
                    sequence_number=1,
                    device_timestamp_s=9.980,
                    host_received_timestamp_s=10.000,
                ),
                FrameSample(
                    sequence_number=2,
                    device_timestamp_s=9.970,
                    host_received_timestamp_s=9.990,
                ),
            ],
        }

        # Check that equal or decreasing first-to-last arrival times are rejected.
        for case_name, samples in invalid_cases.items():
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(
                    ValueError,
                    "host timestamps must increase",
                ):
                    summarize_frame_samples(
                        samples,
                        requested_fps=30.0,
                    )

    def test_sequence_gaps_count_each_missing_frame(self):
        """Sequence discontinuities must count lost frames, not gap events."""

        samples = [
            FrameSample(
                sequence_number=100,
                device_timestamp_s=49.980,
                host_received_timestamp_s=50.000,
            ),
            FrameSample(
                sequence_number=101,
                device_timestamp_s=50.080,
                host_received_timestamp_s=50.100,
            ),
            FrameSample(
                sequence_number=104,
                device_timestamp_s=50.180,
                host_received_timestamp_s=50.200,
            ),
            FrameSample(
                sequence_number=105,
                device_timestamp_s=50.280,
                host_received_timestamp_s=50.300,
            ),
        ]

        summary = summarize_frame_samples(
            samples,
            requested_fps=30.0,
        )

        # Check that the jump from frame 101 to 104 counts missing frames 102 and 103.
        self.assertEqual(summary.sequence_gaps, 2)

    def test_latency_summary_uses_every_available_sample(self):
        """Latency must describe both typical behaviour and slower outliers."""

        samples = [
            FrameSample(
                sequence_number=20,
                device_timestamp_s=10.000,
                host_received_timestamp_s=10.010,
            ),
            FrameSample(
                sequence_number=21,
                device_timestamp_s=10.100,
                host_received_timestamp_s=10.120,
            ),
            FrameSample(
                sequence_number=22,
                device_timestamp_s=10.200,
                host_received_timestamp_s=10.230,
            ),
            FrameSample(
                sequence_number=23,
                device_timestamp_s=10.300,
                host_received_timestamp_s=10.400,
            ),
        ]

        summary = summarize_frame_samples(
            samples,
            requested_fps=30.0,
        )

        # Check the mean of the four delays: 10, 20, 30 and 100 milliseconds.
        self.assertAlmostEqual(summary.latency_mean_ms, 40.0)

        # Check that the rounded-up 95-percent rank selects the fourth delay, 100 ms.
        self.assertAlmostEqual(summary.latency_p95_ms, 100.0)

        # Check that one frame still reports its 25 ms delay without a frame rate.
        one_sample = summarize_frame_samples(
            [
                FrameSample(
                    sequence_number=30,
                    device_timestamp_s=20.000,
                    host_received_timestamp_s=20.025,
                ),
            ],
            requested_fps=30.0,
        )

        self.assertIsNone(one_sample.received_fps)
        self.assertAlmostEqual(one_sample.latency_mean_ms, 25.0)
        self.assertAlmostEqual(one_sample.latency_p95_ms, 25.0)


if __name__ == "__main__":
    unittest.main()
