"""Tests for the one-image gauge reader command."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


import cv2
import numpy as np
from gauge_types import FailureReason, GaugeReadResult
from read_gauge_image import main


class ReadGaugeImageCommandTests(unittest.TestCase):
    """Describe the command's observable input/output behaviour."""

    def test_saved_crop_writes_pivot_overlay_and_serializes_no_radius(self):
        source = (
            Path(__file__).parent
            / "fixtures"
            / "gauge_offcentre_frame_258.png"
        )
        original_bytes = source.read_bytes()
        messages = []

        # Exercise the real reader and image writer without changing the fixture.
        with tempfile.TemporaryDirectory() as folder:
            overlay_path = Path(folder) / "overlay.png"
            code = main(
                [str(source), "--overlay", str(overlay_path)],
                printer=messages.append,
            )
            overlay = cv2.imread(str(overlay_path))

        record = json.loads(messages[0])
        self.assertEqual(code, 0)
        self.assertTrue(record["valid"])
        self.assertIsNone(record["radius"])
        self.assertIsNone(record["failure_reason"])
        self.assertGreater(record["value_bar"], 1.8)
        self.assertLess(record["value_bar"], 2.5)
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertIsNotNone(overlay)

        # The saved annotation includes the measured pivot in source-crop pixels.
        x, y = np.rint(record["centre"]).astype(int)
        np.testing.assert_array_equal(overlay[y, x], (0, 180, 0))

    def test_unreadable_input_returns_one(self) -> None:
        """A missing or unreadable image is an input/output failure."""

        messages: list[str] = []

        code = main(
            ["missing.png"],
            # Match cv2.imread returning None when a file cannot be decoded.
            image_loader=lambda path: None,
            # Capture printed messages for the assertions below.
            printer=messages.append,
        )

        # Code 1 means an input or output error.
        self.assertEqual(code, 1)

        self.assertEqual(len(messages), 1)
        self.assertIn("could not read", messages[0].lower())

    @patch("read_gauge_image.read_gauge")
    def test_rejected_candidate_prints_json_and_returns_two(
        self,
        reader,
    ) -> None:
        """A CV rejection is different from an image input/output failure."""

        image = np.full(
            shape=(120, 120, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        # Isolate command handling with a readable image that has no accepted dial.
        reader.return_value = GaugeReadResult.failure(
            FailureReason.NO_DIAL,
        )

        messages: list[str] = []

        code = main(
            ["candidate.png"],
            image_loader=lambda path: image,
            printer=messages.append,
        )

        # Code 2 means the image loaded but the reader rejected it.
        self.assertEqual(code, 2)

        self.assertEqual(len(messages), 1)

        # Decode the printed result and check it reports NO_DIAL with no pressure.
        document = json.loads(messages[0])

        self.assertFalse(document["valid"])
        self.assertIsNone(document["value_bar"])
        self.assertEqual(document["failure_reason"], "NO_DIAL")

    @patch("read_gauge_image.read_gauge")
    def test_input_path_is_passed_to_reader_as_metadata(
        self,
        reader,
    ) -> None:
        """The structured result must retain its source-image identity."""

        image = np.full(
            shape=(120, 120, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        reader.return_value = GaugeReadResult.failure(
            FailureReason.NO_DIAL,
        )

        main(
            ["candidate.png"],
            image_loader=lambda path: image,
            printer=lambda message: None,
        )

        # Check that the command sends the exact decoded array to read_gauge.
        reader.assert_called_once()
        self.assertIs(
            reader.call_args.args[0],
            image,
        )

        # Check that read_gauge receives the input filename in its metadata.
        self.assertEqual(
            reader.call_args.kwargs,
            {
                "metadata": {
                    "input_image": "candidate.png",
                },
            },
        )

    @patch("read_gauge_image.read_gauge")
    def test_valid_candidate_writes_overlay_and_returns_zero(
        self,
        reader,
    ) -> None:
        """A valid reading may save its visual explanation on request."""

        image = np.full(
            shape=(120, 120, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        reader.return_value = GaugeReadResult.success(
            value_bar=5.0,
            angle_degrees=0.0,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(60.0, 60.0),
            tip=(60.0, 15.0),
            radius=50.0,
        )

        written_paths: list[str] = []

        code = main(
            [
                "candidate.png",
                "--overlay",
                "overlay.png",
            ],
            image_loader=lambda path: image,
            # Record the save path and report success without writing a file.
            image_writer=lambda path, rendered: (written_paths.append(path) or True),
            printer=lambda message: None,
        )

        # Code 0 means the reading and requested outputs succeeded.
        self.assertEqual(code, 0)
        self.assertEqual(
            written_paths,
            ["overlay.png"],
        )

    @patch("read_gauge_image.read_gauge")
    def test_overlay_write_failure_returns_one(
        self,
        reader,
    ) -> None:
        """Failure to save a requested artifact must be reported."""

        image = np.full(
            shape=(120, 120, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        reader.return_value = GaugeReadResult.success(
            value_bar=5.0,
            angle_degrees=0.0,
            dial_quality=0.8,
            needle_quality=0.7,
            centre=(60.0, 60.0),
            tip=(60.0, 15.0),
            radius=50.0,
        )

        messages: list[str] = []

        code = main(
            [
                "candidate.png",
                "--overlay",
                "overlay.png",
            ],
            image_loader=lambda path: image,
            # Simulate cv2.imwrite: return False to report a failed write.
            image_writer=lambda path, rendered: False,
            printer=messages.append,
        )

        # A failed overlay save is still an output error.
        self.assertEqual(code, 1)

        # The save error follows the JSON result.
        self.assertIn(
            "could not write",
            messages[-1].lower(),
        )
        self.assertIn(
            "overlay.png",
            messages[-1],
        )

    def test_script_help_displays_command_usage(self) -> None:
        """Executing the file directly must invoke its command interface."""

        script_path = Path(__file__).resolve().parents[1] / "read_gauge_image.py"

        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)

        self.assertIn(
            "usage:",
            completed.stdout.lower(),
        )
        self.assertIn(
            "--overlay",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
