"""Tests for the executable local camera-stream composition."""

import unittest

import numpy as np
import json

from pathlib import Path
from contextlib import chdir
from tempfile import TemporaryDirectory
from run_camera_stream import main
from stream_types import DeviceDetails, FramePacket


class PreparedSource:
    """Provide deterministic packets while resembling the real OAK source."""

    def __init__(self):
        # Prepare two four-by-six frames with arrival times 0.1 seconds apart.
        image = np.zeros(
            (4, 6, 3),
            dtype=np.uint8,
        )
        self._packets = iter(
            [
                FramePacket(
                    image=image.copy(),
                    sequence_number=1,
                    device_timestamp_s=0.980,
                    host_received_timestamp_s=1.000,
                ),
                FramePacket(
                    image=image.copy(),
                    sequence_number=2,
                    device_timestamp_s=1.080,
                    host_received_timestamp_s=1.100,
                ),
            ]
        )
        self.device_details = DeviceDetails(
            product_name="OAK-D-LITE",
            mxid="TEST-MXID",
            usb_speed="SUPER",
        )
        self.closed = False

    def read(self):
        return next(self._packets)

    def close(self):
        self.closed = True


class StopAfterTwoDisplay:
    """Represent the local window without opening a graphical interface."""

    def __init__(self):
        self.frames_shown = 0
        self.closed = False

    def show(self, packet):
        # Count each shown frame and stop the stream after the second one.
        self.frames_shown += 1
        return self.frames_shown < 2

    def close(self):
        self.closed = True


class RunCameraStreamTests(unittest.TestCase):
    """Define the command-level composition of the stream foundation."""

    def setUp(self):
        # Keep test runs out of the real diagnostics folder, including failed runs.
        directory = self.enterContext(TemporaryDirectory())
        self.enterContext(chdir(directory))

    def test_main_composes_selected_profile_source_and_local_display(self):
        """The command must run the selected profile through local display."""

        selected_profiles = []
        created_sources = []
        created_displays = []
        printed_messages = []

        def source_factory(profile):
            # Record the selected profile and each source created by the command.
            selected_profiles.append(profile)
            source = PreparedSource()
            created_sources.append(source)
            return source

        def display_factory(*, window_name):
            # Check the profile in the window name and retain the created display.
            self.assertEqual(
                window_name,
                "OAK-D Lite — 1080p30",
            )
            display = StopAfterTwoDisplay()
            created_displays.append(display)
            return display

        # Run the command with the prepared source, display and captured console output.
        exit_code = main(
            ["--profile", "1080p30"],
            source_factory=source_factory,
            display_factory=display_factory,
            printer=printed_messages.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            selected_profiles[0].name,
            "1080p30",
        )
        self.assertEqual(
            created_displays[0].frames_shown,
            2,
        )
        self.assertTrue(created_sources[0].closed)
        self.assertTrue(created_displays[0].closed)

        joined_output = "\n".join(printed_messages)
        self.assertIn("USB: SUPER", joined_output)
        self.assertIn("Received FPS: 10.000", joined_output)

    def test_successful_run_writes_json_diagnostic_record(self):
        """A completed trial must leave reproducible diagnostic evidence."""

        # Save this test's generated records inside a temporary directory.
        with TemporaryDirectory() as temporary_directory:
            # Change the working folder so the default diagnostics path is temporary.
            with chdir(temporary_directory):
                exit_code = main(
                    ["--profile", "1080p30"],
                    source_factory=lambda profile: PreparedSource(),
                    display_factory=(lambda *, window_name: StopAfterTwoDisplay()),
                    printer=lambda message: None,
                )

                # Find and read the completed acquisition record written by the command.
                record_paths = list(Path("diagnostics/runs").glob("*.json"))

                self.assertEqual(exit_code, 0)
                self.assertEqual(len(record_paths), 1)

                with record_paths[0].open(
                    "r",
                    encoding="utf-8",
                ) as record_file:
                    record = json.load(record_file)

                self.assertEqual(record["schema_version"], 1)
                self.assertEqual(record["status"], "complete")
                self.assertEqual(record["stop_reason"], "operator_q")

                self.assertEqual(
                    record["device"],
                    {
                        "product_name": "OAK-D-LITE",
                        "mxid": "TEST-MXID",
                        "usb_speed": "SUPER",
                    },
                )

                # Check that the record saved the requested 1080p profile settings.
                self.assertEqual(
                    record["profile"],
                    {
                        "name": "1080p30",
                        "requested_width": 1920,
                        "requested_height": 1080,
                        "requested_fps": 30.0,
                    },
                )

                # Check the supplied six-by-four frames and their measured timing.
                self.assertEqual(record["observed"]["frame_width"], 6)
                self.assertEqual(record["observed"]["frame_height"], 4)
                self.assertEqual(record["observed"]["frames_received"], 2)
                self.assertEqual(record["observed"]["sequence_gaps"], 0)
                self.assertAlmostEqual(
                    record["observed"]["received_fps"],
                    10.0,
                )
                self.assertAlmostEqual(
                    record["observed"]["latency_mean_ms"],
                    20.0,
                )
                self.assertAlmostEqual(
                    record["observed"]["latency_p95_ms"],
                    20.0,
                )

                # Check the recorded timing stage and excluded operations.
                self.assertEqual(
                    record["measurement"]["latency_boundary"],
                    "depthai_frame_timestamp_to_application_dequeue",
                )
                self.assertEqual(
                    record["measurement"]["latency_excludes"],
                    [
                        "bgr_conversion",
                        "local_display",
                        "computer_vision",
                        "network_transport",
                    ],
                )

                self.assertIn("started_at_utc", record)
                self.assertIn("finished_at_utc", record)
                self.assertIn("host", record)
                self.assertIn("software", record)

    def test_source_closes_when_display_construction_raises(self):
        """A pre-stream display failure must release the opened camera."""

        created_sources = []

        def source_factory(profile):
            source = PreparedSource()
            created_sources.append(source)
            return source

        def display_factory(*, window_name):
            # Raise while creating the display, before the frame loop starts.
            raise RuntimeError("display construction failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "display construction failed",
        ):
            main(
                ["--profile", "720p30"],
                source_factory=source_factory,
                display_factory=display_factory,
                printer=lambda message: None,
            )

        self.assertEqual(len(created_sources), 1)
        self.assertTrue(created_sources[0].closed)


if __name__ == "__main__":
    unittest.main()
