import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate_aruco_run import evaluate_run


class EvaluationTests(unittest.TestCase):
    def test_counts_all_frames_and_computes_range_error_only_with_reference(self):
        frames = [dict(sequence_number=1, processing_time_ms=2.0, results=[])]

        # Add two accepted ranges, each 1 cm away from the supplied 30 cm reference.
        for seq, z in enumerate((0.29, 0.31), start=2):
            frames.append(
                dict(
                    sequence_number=seq,
                    processing_time_ms=4.0,
                    results=[
                        dict(
                            marker_id=7,
                            valid=True,
                            failure_reason=None,
                            pose=dict(translation_m=[0.0, 0.0, z]),
                            reprojection_error_px=0.5,
                        )
                    ],
                )
            )

        frames.append(
            dict(
                sequence_number=4,
                processing_time_ms=6.0,
                results=[
                    dict(
                        marker_id=7,
                        valid=False,
                        pose=None,
                        failure_reason="REPROJECTION_ERROR_TOO_HIGH",
                        reprojection_error_px=2.0,
                    )
                ],
            )
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))

            # Evaluate the same log with and without a measured reference distance.
            report = evaluate_run(path, reference_distance_m=0.30)
            unknown = evaluate_run(path)

        self.assertEqual(report["frames"], 4)
        self.assertEqual(report["empty_frames"], 1)
        self.assertEqual(report["markers"]["7"]["accepted"], 2)
        self.assertEqual(report["markers"]["7"]["rejected"], 1)
        self.assertAlmostEqual(report["markers"]["7"]["range_m"]["mean"], 0.30)
        self.assertAlmostEqual(report["markers"]["7"]["range_m"]["stddev"], 0.01)
        self.assertAlmostEqual(
            report["markers"]["7"]["mean_absolute_range_error_m"], 0.01
        )
        self.assertIsNone(unknown["markers"]["7"]["mean_absolute_range_error_m"])
        self.assertEqual(report["processing_time_ms"]["p95"], 6.0)

    def test_malformed_log_is_reported_not_silently_discarded(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text('{"broken":')

            with self.assertRaisesRegex(ValueError, "line 1"):
                evaluate_run(path)

    def test_empty_run_has_no_invented_measurements(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text("")

            report = evaluate_run(path)

        self.assertEqual(report["markers"], {})
        self.assertIsNone(report["processing_time_ms"]["mean"])
