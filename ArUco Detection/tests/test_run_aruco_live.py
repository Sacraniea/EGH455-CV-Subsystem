"""Run the live command with a fake camera and display."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from stream_types import FramePacket, DeviceDetails
from run_aruco_live import main


class Source:
    def __init__(self, fail=False):
        # Prepare a marker frame and a blank frame for the live runner's two-frame test.
        image = np.full((300, 400, 3), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
            7,
            200,
        )
        image[50:250, 50:250] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        self.original = image.copy()
        # Skip sequence 42 to check that the stream reports a dropped frame.
        self.packets = [
            FramePacket(image, 41, 12.0, 12.05),
            FramePacket(np.full_like(image, 255), 43, 12.1, 12.15),
        ]
        self.iterator = iter(self.packets)
        self.device_details = DeviceDetails("OAK-D-LITE", "test-device", "SUPER")
        self.closed = False
        self.close_count = 0
        self.fail = fail

    def read(self):
        # Simulate a disconnected camera when requested; otherwise return the next frame.
        if self.fail:
            raise RuntimeError("camera disconnected")

        return next(self.iterator)

    def close(self):
        self.closed = True
        self.close_count += 1


class Display:
    def __init__(self):
        self.packets = []
        self.closed = False
        self.close_count = 0

    def show(self, packet):
        # Retain each displayed packet and let the configured frame limit stop the run.
        self.packets.append(packet)
        return True

    def close(self):
        self.closed = True
        self.close_count += 1


class LiveRunnerTests(unittest.TestCase):
    def run_command(self, root, source, display, extra=()):
        # Run two frames through the real command using the supplied fake camera and display.
        return main(
            [
                "--permitted-ids",
                "7",
                "--marker-side-length-m",
                ".15",
                "--output-dir",
                str(root / "run"),
                "--max-frames",
                "2",
                *extra,
            ],
            source_factory=lambda profile: source,
            display_factory=lambda **kwargs: display,
            printer=lambda message: None,
        )

    def test_live_run_records_empty_frames_identity_settings_and_cleanup(self):
        source, display = Source(), Display()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_command(root, source, display), 0)
            frames = [
                json.loads(line)
                for line in (root / "run" / "frames.jsonl").read_text().splitlines()
            ]
            manifest = json.loads((root / "run" / "run.json").read_text())

        self.assertEqual(source.close_count, 1)
        self.assertEqual(display.close_count, 1)
        self.assertEqual([f["sequence_number"] for f in frames], [41, 43])
        self.assertEqual(
            frames[0]["results"][0]["failure_reason"], "MISSING_CALIBRATION"
        )
        self.assertEqual(frames[1]["results"], [])
        self.assertEqual(frames[1]["device_timestamp_s"], 12.1)
        self.assertEqual(frames[0]["config"]["marker_side_length_m"], 0.15)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["stop_reason"], "frame_limit")
        self.assertEqual(manifest["counts"]["frames"], 2)
        self.assertEqual(manifest["counts"]["empty_frames"], 1)
        self.assertEqual(manifest["stream_summary"]["sequence_gaps"], 1)
        np.testing.assert_array_equal(source.packets[0].image, source.original)
        self.assertFalse(
            np.shares_memory(display.packets[0].image, source.packets[0].image)
        )

    def test_matching_calibration_is_used_and_copied_into_run_evidence(self):
        source, display = Source(), Display()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "cal.json"
            calibration.write_text(
                json.dumps(
                    dict(
                        camera_socket="CAM_A",
                        device_mxid="test-device",
                        distortion_model="CameraModel.Perspective",
                        image_size_px=[400, 300],
                        camera_matrix=[[600, 0, 200], [0, 600, 150], [0, 0, 1]],
                        distortion_coefficients=[0] * 5,
                    )
                )
            )

            self.assertEqual(
                self.run_command(
                    root,
                    source,
                    display,
                    ["--calibration", str(calibration)],
                ),
                0,
            )
            manifest = json.loads((root / "run" / "run.json").read_text())
            frame = json.loads(
                (root / "run" / "frames.jsonl").read_text().splitlines()[0]
            )

        self.assertTrue(frame["results"][0]["valid"])
        self.assertAlmostEqual(
            frame["results"][0]["pose"]["translation_m"][2],
            600 * 0.15 / 199,
            places=5,
        )
        self.assertEqual(
            manifest["metadata"]["calibration"]["original_record"]["device_mxid"],
            "test-device",
        )
        self.assertEqual(len(manifest["metadata"]["calibration"]["sha256"]), 64)

    def test_window_startup_failure_releases_camera(self):
        source = Source()

        with TemporaryDirectory() as directory:

            def failing_display(**kwargs):
                raise RuntimeError("window creation failed")

            self.assertEqual(
                main(
                    [
                        "--permitted-ids",
                        "7",
                        "--output-dir",
                        str(Path(directory) / "run"),
                    ],
                    source_factory=lambda profile: source,
                    display_factory=failing_display,
                    printer=lambda message: None,
                ),
                1,
            )

        self.assertTrue(source.closed)

    def test_camera_failure_closes_resources_and_records_failure(self):
        source, display = Source(fail=True), Display()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_command(root, source, display), 1)
            manifest = json.loads((root / "run" / "run.json").read_text())

        self.assertEqual(source.close_count, 1)
        self.assertEqual(display.close_count, 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("camera disconnected", manifest["error"])

    def test_bad_calibration_closes_camera_before_opening_window(self):
        source = Source()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "cal.json"
            calibration.write_text(
                json.dumps(
                    dict(
                        camera_socket="CAM_A",
                        device_mxid="wrong",
                        distortion_model="CameraModel.Perspective",
                        image_size_px=[400, 300],
                        camera_matrix=[[600, 0, 200], [0, 600, 150], [0, 0, 1]],
                        distortion_coefficients=[0] * 5,
                    )
                )
            )

            def forbidden_display(**kwargs):
                raise AssertionError(
                    "window must not open for wrong device calibration"
                )

            code = main(
                [
                    "--permitted-ids",
                    "7",
                    "--calibration",
                    str(calibration),
                    "--output-dir",
                    str(root / "run"),
                ],
                source_factory=lambda profile: source,
                display_factory=forbidden_display,
                printer=lambda message: None,
            )

        self.assertEqual(code, 1)
        self.assertTrue(source.closed)

    def test_existing_output_and_invalid_threshold_do_not_open_camera(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def forbidden_source(profile):
                raise AssertionError("camera must not open")

            code = main(
                ["--permitted-ids", "7", "--output-dir", str(root)],
                source_factory=forbidden_source,
                printer=lambda message: None,
            )
            self.assertEqual(code, 1)

            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                main(
                    ["--permitted-ids", "7", "--max-reprojection-error-px", "nan"],
                    source_factory=forbidden_source,
                )

            self.assertEqual(caught.exception.code, 2)

    def test_keyboard_interrupt_is_recorded_and_closes_resources(self):
        source, display = Source(), Display()
        # Raise on the next camera read, as if Ctrl+C interrupted the stream.
        def interrupted_read():
            raise KeyboardInterrupt

        source.read = interrupted_read

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_command(root, source, display), 130)
            manifest = json.loads((root / "run" / "run.json").read_text())

        self.assertEqual(source.close_count, 1)
        self.assertEqual(display.close_count, 1)
        self.assertEqual(manifest["status"], "interrupted")

    def test_help_from_any_directory(self):
        script = Path(__file__).resolve().parents[1] / "run_aruco_live.py"

        with TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, "-B", str(script), "--help"],
                cwd=directory,
                text=True,
                capture_output=True,
            )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--calibration-mode", process.stdout)

    def test_logging_failure_still_closes_camera_and_display(self):
        source, display = Source(), Display()

        with (
            TemporaryDirectory() as directory,
            patch(
                "aruco_records.ArUcoRunRecorder.write",
                side_effect=OSError("disk full"),
            ),
        ):
            root = Path(directory)
            self.assertEqual(self.run_command(root, source, display), 1)
            manifest = json.loads((root / "run" / "run.json").read_text())

        self.assertEqual(source.close_count, 1)
        self.assertEqual(display.close_count, 1)
        self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
