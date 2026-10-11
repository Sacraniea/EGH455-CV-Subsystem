import json
import io
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np

from run_camera_stream import main
from stream_types import FramePacket
from test_run_camera_stream import PreparedSource, StopAfterTwoDisplay


class StreamCommandTests(unittest.TestCase):
    def test_capture_flag_saves_original_and_overlay_for_one_marker_sighting(self):
        # Draw marker 7 in a white image and supply five frames of the same sighting.
        image = np.full((600, 800, 3), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100), 7, 200
        )
        image[200:400, 300:500] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        source = PreparedSource()
        # Advance frame identity and time while keeping the marker continuously visible.
        source._packets = iter(
            FramePacket(image.copy(), sequence, timestamp, timestamp + 0.02)
            for sequence, timestamp in enumerate((1.0, 1.1, 1.2, 1.4, 1.5), start=42)
        )

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            code = main(
                [
                    "--aruco",
                    "--permitted-ids",
                    "7",
                    "--headless",
                    "--max-frames",
                    "5",
                    "--capture-detections",
                    "--output-dir",
                    str(output),
                ],
                source_factory=lambda profile: source,
                printer=lambda _: None,
            )
            self.assertEqual(code, 0)
            # Keep one event with the original and the drawn version of the same frame.
            images = list((output / "detections").glob("*.png"))
            self.assertEqual(len(images), 2)
            events = list((output / "detections").glob("*.event.json"))
            self.assertEqual(len(events), 1)
            event = json.loads(events[0].read_text())
            original = cv2.imread(str(events[0].parent / event["image_file"]))
            overlay = cv2.imread(
                str(events[0].parent / event["annotated_image"]["image_file"])
            )
            np.testing.assert_array_equal(original, image)
            self.assertEqual(overlay.shape, image.shape)
            self.assertTrue(np.any(overlay != original))
            self.assertEqual(event["sequence_number"], 45)
            frame_records = [
                json.loads(line)
                for line in (output / "frames.jsonl").read_text().splitlines()
            ]
            # Compare the capture's record with the log entry for its selected frame.
            expected = next(
                record for record in frame_records if record["sequence_number"] == 45
            )
            self.assertEqual(event["record"], expected)
            self.assertIsNone(
                event["record"]["processors"][0]["data"]["results"][0]["pose"]
            )

        self.assertTrue(source.closed)

    def test_headless_aruco_records_marker_and_stops_without_a_window(self):
        # Run one generated marker frame through the command with preview disabled.
        image = np.full((600, 800, 3), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100), 7, 200
        )
        image[200:400, 300:500] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        source = PreparedSource()
        source._packets = iter([FramePacket(image, 42, 1.0, 1.02)])

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            code = main(
                [
                    "--aruco",
                    "--permitted-ids",
                    "7",
                    "--headless",
                    "--max-frames",
                    "1",
                    "--output-dir",
                    str(output),
                ],
                source_factory=lambda profile: source,
                display_factory=lambda **kw: self.fail("headless opened a window"),
                printer=lambda _: None,
            )
            self.assertEqual(code, 0)
            # Check marker 7 was saved without a pose because calibration is missing.
            record = json.loads((output / "frames.jsonl").read_text())
            self.assertEqual(record["sequence_number"], 42)
            (target,) = record["processors"][0]["data"]["results"]
            self.assertEqual(target["marker_id"], 7)
            self.assertEqual(target["failure_reason"], "MISSING_CALIBRATION")
            self.assertIsNone(target["pose"])
            # Check that reaching one frame completed the run with a matching run ID.
            manifest = json.loads((output / "run.json").read_text())
            self.assertEqual(manifest["stop_reason"], "frame_limit")
            self.assertEqual(manifest["frames_recorded"], 1)
            self.assertEqual(manifest["status"], "complete")
            self.assertIn("run_id", record)
            self.assertEqual(record["run_id"], manifest["run_id"])

        self.assertTrue(source.closed)

    def test_failure_manifest_and_cleanup_when_source_stops(self):
        # Fail the first camera read and check the saved status and cleanup.
        source, display = PreparedSource(), StopAfterTwoDisplay()
        source.read = lambda: (_ for _ in ()).throw(RuntimeError("camera disconnected"))

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
                main(
                    ["--output-dir", str(output)],
                    source_factory=lambda p: source,
                    display_factory=lambda **kw: display,
                    printer=lambda _: None,
                )

            manifest = json.loads((output / "run.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("camera disconnected", manifest["error"])

        self.assertTrue(source.closed and display.closed)

    def test_interrupt_manifest_and_cleanup(self):
        # Interrupt the first camera read and check the exit code and saved run status.
        source, display = PreparedSource(), StopAfterTwoDisplay()

        def interrupt():
            raise KeyboardInterrupt()

        source.read = interrupt

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            code = main(
                ["--output-dir", str(output)],
                source_factory=lambda p: source,
                display_factory=lambda **kw: display,
                printer=lambda _: None,
            )
            self.assertEqual(code, 130)
            self.assertEqual(
                json.loads((output / "run.json").read_text())["status"], "interrupted"
            )

        self.assertTrue(source.closed and display.closed)

    def test_headless_continuous_run_records_frames_until_interrupted(self):
        source = PreparedSource()
        read_packet = source.read

        def read_then_interrupt():
            # Supply two packets before simulating the operator pressing Ctrl+C.
            try:
                return read_packet()

            except StopIteration:
                raise KeyboardInterrupt()

        source.read = read_then_interrupt

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            code = main(
                ["--headless", "--max-frames", "0", "--output-dir", str(output)],
                source_factory=lambda profile: source,
                display_factory=lambda **kwargs: self.fail("Headless opened a window"),
                printer=lambda message: None,
            )
            self.assertEqual(code, 130)
            manifest = json.loads((output / "run.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["stop_reason"], "keyboard_interrupt")
            self.assertEqual(manifest["frames_recorded"], 2)
            records = [
                json.loads(line)
                for line in (output / "frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["sequence_number"] for record in records], [1, 2])

        self.assertTrue(source.closed)

    def test_invalid_options_do_not_open_camera(self):
        # Try invalid option combinations and fail the test if any one opens a camera.
        for options in (
            ["--headless"],
            ["--permitted-ids", "7"],
            ["--max-frames", "-1"],
            ["--max-processing-ms", "nan"],
            ["--calibration-mode", "oak-1080"],
        ):
            with self.subTest(options=options), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                main(
                    options,
                    source_factory=lambda p: self.fail("opened camera"),
                    printer=lambda _: None,
                )

    def test_existing_output_is_not_overwritten_or_camera_opened(self):
        # Supply an existing run folder and check that the command refuses to reuse it.
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            path.mkdir()

            with self.assertRaises(FileExistsError):
                main(
                    ["--output-dir", str(path)],
                    source_factory=lambda p: self.fail("opened camera"),
                )

    def test_wrong_calibration_device_closes_camera_before_display(self):
        # Supply calibration for another device and reject it before preview opens.
        source = PreparedSource()

        with TemporaryDirectory() as tmp:
            cal = Path(tmp) / "cal.json"
            cal.write_text(
                json.dumps(
                    dict(
                        camera_matrix=[[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                        distortion_coefficients=[0] * 5,
                        image_size_px=[1280, 720],
                        distortion_model="CameraModel.Perspective",
                        camera_socket="CAM_A",
                        device_mxid="WRONG",
                    )
                )
            )
            output = Path(tmp) / "run"

            with self.assertRaises(ValueError):
                main(
                    [
                        "--aruco",
                        "--permitted-ids",
                        "7",
                        "--calibration",
                        str(cal),
                        "--output-dir",
                        str(output),
                    ],
                    source_factory=lambda p: source,
                    display_factory=lambda **kw: self.fail("opened display"),
                )

            self.assertEqual(
                json.loads((output / "run.json").read_text())["status"], "failed"
            )

        self.assertTrue(source.closed)

    def test_finalisation_failure_does_not_hide_camera_failure(self):
        # Fail both the camera read and final save, then check which error is reported.
        from unittest.mock import patch
        from stream_session import StreamSession

        original_finish = StreamSession.finish

        def failing_finish(session, **kwargs):
            # Write the normal run summary, then raise a second failure from this save.
            original_finish(session, **kwargs)
            raise OSError("manifest finalisation failed")

        source = PreparedSource()
        source.read = lambda: (_ for _ in ()).throw(
            RuntimeError("original camera failure")
        )

        with TemporaryDirectory() as tmp, patch.object(
            StreamSession, "finish", failing_finish
        ):
            with self.assertRaisesRegex(
                RuntimeError, "original camera failure"
            ) as caught:
                main(
                    ["--output-dir", str(Path(tmp) / "run")],
                    source_factory=lambda p: source,
                    display_factory=lambda **kw: StopAfterTwoDisplay(),
                    printer=lambda _: None,
                )

            self.assertIn(
                "manifest finalisation failed", "\n".join(caught.exception.__notes__)
            )

        self.assertTrue(source.closed)
