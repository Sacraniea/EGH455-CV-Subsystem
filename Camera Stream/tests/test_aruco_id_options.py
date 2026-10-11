"""Check default and explicit ArUco marker selection through the stream commands."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from run_camera_stream import build_parser as camera_parser, main
from run_local_demo import build_parser as demo_parser, _sender_arguments
from stream_types import FramePacket, get_stream_profile
from test_run_camera_stream import PreparedSource


class ArUcoIdOptionTests(unittest.TestCase):
    def test_default_stream_accepts_dictionary_boundaries_and_an_id_other_than_seven(self):
        # These generated markers exercise real decoding without opening a camera.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        for marker_id in (0, 23, 99):
            with self.subTest(marker_id=marker_id), TemporaryDirectory() as temporary:
                image = np.full((600, 800, 3), 255, np.uint8)
                marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
                image[200:400, 300:500] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
                source = PreparedSource()
                source._packets = iter([FramePacket(image, 42, 1.0, 1.02)])
                output = Path(temporary) / "run"

                code = main(
                    [
                        "--aruco",
                        "--headless",
                        "--max-frames",
                        "1",
                        "--output-dir",
                        str(output),
                    ],
                    source_factory=lambda profile: source,
                    printer=lambda message: None,
                )

                self.assertEqual(code, 0)
                record = json.loads((output / "frames.jsonl").read_text())
                result, = record["processors"][0]["data"]["results"]
                self.assertEqual(result["marker_id"], marker_id)
                # Accept the ID while retaining the requirement for calibrated pose.
                self.assertEqual(result["failure_reason"], "MISSING_CALIBRATION")
                self.assertIsNone(result["pose"])
                self.assertTrue(source.closed)

    def test_demo_omits_restriction_by_default_and_preserves_an_explicit_list(self):
        # Load the adapter after discovery so its path setup cannot shadow stream tests.
        from stream_aruco import prepare_aruco

        for options, expected_ids in (
            ([], frozenset(range(100))),
            (["--permitted-ids", "7", "23"], frozenset({7, 23})),
        ):
            with self.subTest(options=options):
                demo_args = demo_parser().parse_args(options)
                sender_args = list(
                    map(str, _sender_arguments(demo_args, Path("trial")))
                )
                args = camera_parser().parse_args(sender_args)
                processor, metadata = prepare_aruco(
                    args,
                    get_stream_profile(args.profile),
                )

                self.assertEqual(processor.config.permitted_ids, expected_ids)
                self.assertEqual(metadata["permitted_ids"], sorted(expected_ids))


if __name__ == "__main__":
    unittest.main()
