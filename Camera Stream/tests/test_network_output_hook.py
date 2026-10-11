import unittest
import numpy as np
from stream_controller import run_stream
from stream_types import FramePacket
from test_stream_controller import SequenceFrameSource
from test_stream_integration import Display
from test_stream_processing import Processor


class NetworkOutputHookTests(unittest.TestCase):
    def test_source_sink_receives_unannotated_pixels_and_matching_evidence(self):
        image = np.zeros((4, 6, 3), np.uint8)
        source = SequenceFrameSource([FramePacket(image, 8, 1.0, 1.02)])
        originals = []

        def retain_original(packet, record):
            originals.append((packet.image.copy(), record.copy()))

        run_stream(
            source,
            Display(),
            requested_fps=30.0,
            max_frames=1,
            processors=[Processor("aruco")],
            on_source=retain_original,
        )
        captured, record = originals[0]
        # Check that the source hook captured zero-valued pixels before the processor drew on them.
        np.testing.assert_array_equal(captured, image)
        self.assertFalse(np.any(captured))
        self.assertEqual(record["sequence_number"], 8)
        self.assertEqual(record["device_timestamp_s"], 1.0)
        self.assertEqual(record["processors"][0]["identity"]["sequence_number"], 8)

    def test_sink_receives_composed_frame_and_its_result_before_local_recording(self):
        image = np.zeros((4, 6, 3), np.uint8)
        source = SequenceFrameSource([FramePacket(image, 8, 1.0, 1.02)])
        received = []
        records = []

        def output(packet, record):
            # Add run_id in the output hook and check that the later record sink receives it.
            record["run_id"] = "trial"
            received.append((packet, record.copy()))

        run_stream(
            source,
            Display(),
            requested_fps=30.0,
            max_frames=1,
            processors=[Processor("aruco")],
            on_output=output,
            on_frame=records.append,
        )
        packet, record = received[0]
        self.assertEqual(packet.sequence_number, 8)
        self.assertEqual(record["sequence_number"], 8)
        self.assertEqual(record["device_timestamp_s"], 1.0)
        self.assertEqual(records[0]["run_id"], "trial")
        self.assertTrue(np.any(packet.image))
        self.assertFalse(np.any(image))

    def test_transport_failure_closes_source_and_preview(self):
        source = SequenceFrameSource(
            [FramePacket(np.zeros((4, 6, 3), np.uint8), 8, 1.0, 1.02)]
        )
        display = Display()

        def fail(packet, record):
            raise ConnectionError("receiver lost")

        with self.assertRaisesRegex(ConnectionError, "receiver lost"):
            run_stream(
                source,
                display,
                requested_fps=30.0,
                max_frames=1,
                on_output=fail,
            )

        self.assertTrue(source.closed and display.closed)
