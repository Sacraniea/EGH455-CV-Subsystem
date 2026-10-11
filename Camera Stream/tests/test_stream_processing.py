import unittest
from dataclasses import replace
import numpy as np

from stream_types import FramePacket
from stream_processing import (
    FrameIdentity,
    ProcessingResult,
    process_frame,
    ResultCompositor,
)


class Processor:
    def __init__(self, name, action=None):
        # Store an optional processing action and the packets this test adapter sees.
        self.name, self.action, self.seen = name, action, []

    def process(self, packet):
        # Record the input and run the supplied action, or return an empty result.
        self.seen.append(packet)

        if self.action:
            return self.action(packet)

        return ProcessingResult(
            self.name, FrameIdentity.from_packet(packet), {"targets": []}
        )

    def draw(self, image, result):
        # Change one overlay pixel so the test can detect whether drawing occurred.
        image[0, 0] = 255
        return image


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.packet = FramePacket(np.zeros((4, 6, 3), np.uint8), 10, 1.0, 1.02)

    def test_same_packet_to_each_consumer_and_copy_only_composition(self):
        # Process one packet twice and draw the results onto a separate output image.
        processors = [Processor("detector"), Processor("aruco")]
        results = process_frame(self.packet, processors)
        output = ResultCompositor(processors).compose(self.packet, results)
        # Check that both adapters saw the source packet and only its copy was drawn on.
        self.assertTrue(all(p.seen[0] is self.packet for p in processors))
        self.assertEqual(
            FrameIdentity.from_packet(output), FrameIdentity.from_packet(self.packet)
        )
        self.assertFalse(np.shares_memory(output.image, self.packet.image))
        self.assertTrue(np.all(self.packet.image == 0))
        self.assertTrue(np.all(output.image[0, 0] == 255))

    def test_stale_result_with_matching_sequence_but_wrong_timestamp_is_rejected(self):
        # Return the current frame number with a different capture timestamp.
        p = Processor(
            "old",
            lambda packet: ProcessingResult(
                "old",
                replace(FrameIdentity.from_packet(packet), device_timestamp_s=0.0),
                {"targets": [1]},
            ),
        )
        (result,) = process_frame(self.packet, [p])
        # Check that the stale result loses its measurements and produces no drawing.
        self.assertEqual(result.failure_reason, "FRAME_IDENTITY_MISMATCH")
        self.assertEqual(result.data, {})
        output = ResultCompositor([p]).compose(self.packet, [result])
        np.testing.assert_array_equal(output.image, self.packet.image)

    def test_branch_error_preserves_other_branch_and_source_writeability(self):
        # Attempt to edit the source pixels in one processor while another succeeds.
        def bad(packet):
            packet.image[0, 0] = 99

        results = process_frame(self.packet, [Processor("bad", bad), Processor("good")])
        # Check that the failed edit is reported and the original pixels stay unchanged.
        self.assertEqual(results[0].failure_reason, "PROCESSOR_ERROR")
        self.assertIsNone(results[1].failure_reason)
        self.assertTrue(self.packet.image.flags.writeable)
        self.assertTrue(np.all(self.packet.image == 0))

    def test_compositor_independently_refuses_wrong_frame(self):
        # Give the drawing stage a result created for a different frame number.
        p = Processor("a")
        result = p.process(replace(self.packet, sequence_number=9))

        with self.assertRaisesRegex(ValueError, "identity"):
            ResultCompositor([p]).compose(self.packet, [result])

    def test_elapsed_budget_discards_returned_result(self):
        # Return clock readings 100 ms apart while allowing only 50 ms of processing.
        clock = iter([1.0, 1.1]).__next__
        (result,) = process_frame(
            self.packet,
            [Processor("slow")],
            max_processing_ms=50.0,
            clock=clock,
        )
        self.assertEqual(result.failure_reason, "PROCESSING_DEADLINE_EXCEEDED")
        self.assertEqual(result.data, {})
        self.assertAlmostEqual(result.processing_time_ms, 100.0)

    def test_duplicate_names_rejected_before_any_processor_runs(self):
        # Register two equal names and check that neither processor receives a frame.
        a, b = Processor("same"), Processor("same")

        with self.assertRaises(ValueError):
            process_frame(self.packet, [a, b])

        self.assertEqual(a.seen, [])

    def test_keyboard_interrupt_is_not_downgraded_to_branch_failure(self):
        # Raise Ctrl+C inside a processor and check that it still interrupts the run.
        def stop(packet):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            process_frame(self.packet, [Processor("stop", stop)])

        # Check that interruption restores permission to write to the source image.
        self.assertTrue(self.packet.image.flags.writeable)
