"""Check that the stream loads one model and records its current-frame results."""

from contextlib import chdir, redirect_stderr
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import stream_target_detection as targets
from run_camera_stream import main
from test_run_camera_stream import PreparedSource


class PreparedPredictor:
    """Replace only model inference; use the real target and gauge adapter."""

    class_names = ["closedValve", "gauge", "openValve"]

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = SimpleNamespace(names = dict(enumerate(self.class_names)))
        self.calls = 0

    def predict_detections(self, image):
        self.calls += 1

        if self.calls > 1:
            return []

        # Supply one unreadable gauge and a valve, followed by an empty frame.
        return [
            {
                "class_name": "gauge",
                "box_xywhn": (0.25, 0.5, 0.5, 1.0),
                "bbox_xyxy_px": (0, 0, 3, 4),
                "gauge_crop": image[:4, :3].copy(),
            },
            {
                "class_name": "openValve",
                "box_xywhn": (0.75, 0.5, 0.5, 1.0),
                "bbox_xyxy_px": (3, 0, 6, 4),
                "gauge_crop": None,
            },
        ]

    def annotate_image(self, image, labels):
        return image.copy()


class TargetCommandTests(unittest.TestCase):
    def test_target_only_capture_saves_periodic_photos_with_unavailable_pressure(self):
        from stream_types import FramePacket

        image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        source = PreparedSource()
        source._packets = iter(
            FramePacket(image.copy(), number, timestamp, timestamp + 0.02)
            for number, timestamp in enumerate((1.0, 1.4, 2.0, 4.5), start=1)
        )
        original_predict = PreparedPredictor.predict_detections

        def keep_targets_visible(predictor, pixels):
            # Repeat the detector fixture so capture can observe a continuous sighting.
            predictor.calls = 0
            return original_predict(predictor, pixels)

        with patch.object(PreparedPredictor, "predict_detections", keep_targets_visible):
            code = main(
                [
                    "--targets",
                    "--target-model", str(self.model),
                    "--headless",
                    "--max-frames", "4",
                    "--capture-detections",
                    "--capture-interval-s", "3",
                    "--output-dir", str(self.directory / "session"),
                ],
                source_factory=lambda profile: source,
                printer=lambda message: None,
            )

        manifest, frames = self.read_run()
        events = sorted((self.directory / "session/detections").glob("*.event.json"))
        self.assertEqual(code, 0)
        self.assertEqual(len(events), 2)
        self.assertEqual(
            manifest["metadata"]["detection_capture"]["target_interval_s"], 3.0
        )

        for event_path in events:
            event = json.loads(event_path.read_text())
            self.assertEqual(event["marker_ids"], [])
            self.assertEqual(len(event["target_results"]), 2)
            self.assertIsNone(event["target_results"][0]["gauge_reading"]["value_bar"])
            self.assertEqual(event["record"], frames[event["sequence_number"] - 1])

        self.assertTrue(source.closed)

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model.pt"
        self.model.write_bytes(b"local test weights")
        self.loaded = []

        def construct(model_path):
            predictor = PreparedPredictor(model_path)
            self.loaded.append(predictor)
            return predictor

        # Keep optional Torch imports out of command tests in the camera environment.
        self.module_patch = patch.dict(
            sys.modules,
            {"predict": SimpleNamespace(Predictor = construct)},
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_default_model_path_is_independent_of_working_directory(self):
        model = self.directory / "models" / "50_epoch_best.pt"
        model.parent.mkdir()
        model.write_bytes(b"default test weights")

        with patch.object(targets, "TARGET_DIRECTORY", self.directory):
            with chdir(self.directory.parent):
                processor, metadata = targets.prepare_targets()

        self.assertEqual(len(self.loaded), 1)
        self.assertIs(processor.predictor, self.loaded[0])
        self.assertEqual(Path(self.loaded[0].model_path), model.resolve())
        self.assertEqual(metadata["model_sha256"], sha256(model.read_bytes()).hexdigest())
        self.assertEqual(metadata["class_names"], PreparedPredictor.class_names)

    def test_missing_model_is_rejected_before_loading_predictor(self):
        with self.assertRaisesRegex(FileNotFoundError, "model"):
            targets.prepare_targets(self.directory / "missing.pt")

        self.assertEqual(self.loaded, [])

    def test_rejects_weights_with_a_different_class_order(self):
        predictor = PreparedPredictor(str(self.model))
        predictor.model.names = {0: "gauge", 1: "closedValve", 2: "openValve"}

        with patch("predict.Predictor", return_value = predictor):
            with self.assertRaisesRegex(ValueError, "class names"):
                targets.prepare_targets(self.model)

    def test_custom_model_requires_targets_before_opening_source(self):
        with redirect_stderr(StringIO()) as error:
            with self.assertRaises(SystemExit):
                main(["--target-model", str(self.model)])

        self.assertIn("--target-model requires --targets", error.getvalue())
        self.assertEqual(self.loaded, [])

    def test_loads_once_and_saves_gauge_failure_valve_and_empty_next_frame(self):
        source = PreparedSource()

        with chdir(self.directory):
            exit_code = main(
                [
                    "--targets", "--target-model", "model.pt",
                    "--headless", "--max-frames", "2",
                    "--output-dir", "session",
                ],
                source_factory = lambda profile: source,
                printer = lambda message: None,
            )

        manifest, frames = self.read_run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self.loaded), 1)
        self.assertEqual(self.loaded[0].calls, 2)
        self.assertEqual(Path(self.loaded[0].model_path), self.model.resolve())
        self.assertTrue(source.closed)
        self.assertEqual(manifest["metadata"]["enabled_processors"], ["targets"])
        self.assertEqual(manifest["metadata"]["targets"]["model_path"], str(self.model.resolve()))

        first = frames[0]["processors"][0]
        self.assertIsNone(first["failure_reason"])
        self.assertEqual(frames[0]["sequence_number"], 1)
        self.assertEqual(frames[0]["device_timestamp_s"], 0.980)
        gauge, valve = first["data"]["results"]
        self.assertFalse(gauge["gauge_reading"]["valid"])
        self.assertIsNone(gauge["gauge_reading"]["value_bar"])
        self.assertEqual(gauge["gauge_reading"]["failure_reason"], "DIAL_TOO_SMALL")
        self.assertEqual(valve["class_name"], "openValve")
        self.assertEqual(frames[1]["processors"][0]["data"]["results"], [])

    def test_synthetic_source_supports_targets_alone_and_with_aruco(self):
        for aruco in (False, True):
            with self.subTest(aruco = aruco), chdir(self.directory):
                output = "with-aruco" if aruco else "targets-only"
                argv = [
                    "--synthetic", "--targets", "--target-model", str(self.model),
                    "--headless", "--max-frames", "1", "--output-dir", output,
                ]

                if aruco:
                    argv.extend(["--aruco", "--permitted-ids", "7"])

                exit_code = main(argv, printer = lambda message: None)
                manifest, frames = self.read_run(output)
                self.assertEqual(exit_code, 0)
                self.assertEqual(frames[0]["status"], "complete")
                processors = frames[0]["processors"]
                self.assertEqual(processors[-1]["processor"], "targets")

                if aruco:
                    self.assertEqual(processors[0]["processor"], "aruco")
                    calibration = manifest["metadata"]["aruco"]["calibration"]
                    self.assertEqual(calibration["transform"], "synthetic")

                else:
                    self.assertIsNone(manifest["metadata"]["aruco"])

    def test_camera_only_does_not_load_a_target_model(self):
        with chdir(self.directory):
            main(
                ["--headless", "--max-frames", "2", "--output-dir", "session"],
                source_factory = lambda profile: PreparedSource(),
                printer = lambda message: None,
            )

        self.assertEqual(self.loaded, [])

    def read_run(self, name = "session"):
        directory = self.directory / name
        manifest = json.loads((directory / "run.json").read_text())
        frames = [
            json.loads(line)
            for line in (directory / "frames.jsonl").read_text().splitlines()
        ]
        return manifest, frames


if __name__ == "__main__":
    unittest.main()
