"""Check target options passed from the demo launcher to its sender."""

from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_local_demo


class ControlledChildren:
    def __init__(self, root):
        self.root = root
        self.running = []
        self.started = {}
        self.closed = False

    def start(self, name, script, arguments):
        # Record the exact child command without starting a camera or service.
        arguments = list(map(str, arguments))
        self.started[name] = (script, arguments)
        process = SimpleNamespace(returncode=0, poll=lambda: 0)
        self.running.append((name, process, None))

        if name == "sender":
            # Supply matching records so the launcher's real delivery check can finish.
            record = dict(
                run_id="controlled-demo",
                sequence_number=1,
                device_timestamp_s=1.0,
                host_received_timestamp_s=1.01,
                image_size_px=[1920, 1080],
                source_kind="synthetic" if "--synthetic" in arguments else "camera",
                processors=[],
            )

            for relative in (
                "sender/frames.jsonl",
                "mqtt.jsonl",
                "receiver/frames.jsonl",
            ):
                destination = self.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(record) + "\n")

        return process

    def ready(self, process, marker, seconds=10.0):
        pass

    def close(self):
        self.closed = True


class LocalDemoTargetTests(unittest.TestCase):
    def test_capture_interval_reaches_the_target_sender(self):
        started = self.run_demo([
            "--targets", "--capture-detections", "--capture-interval-s", "3.5",
        ])
        arguments = started["sender"][1]
        self.assertEqual(
            arguments[arguments.index("--capture-interval-s") + 1], "3.5"
        )

    def run_demo(self, options, *, caller_directory=None):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "trial"
        instances = []

        def create_children(root):
            children = ControlledChildren(root)
            instances.append(children)
            return children

        # Exercise parsing, command assembly and finalization without opening sockets.
        with chdir(caller_directory or Path.cwd()), patch(
            "run_local_demo.check_ports"
        ), patch(
            "run_local_demo.Children", side_effect=create_children
        ), redirect_stdout(StringIO()):
            code = run_local_demo.main(
                ["--output-dir", str(output), *options]
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(instances), 1)
        self.assertTrue(instances[0].closed)
        self.assertTrue(json.loads((output / "summary.json").read_text())["verified"])
        return instances[0].started

    def test_default_demo_keeps_aruco_and_network_without_enabling_targets(self):
        started = self.run_demo([])
        script, arguments = started["sender"]

        self.assertEqual(script, "run_camera_stream.py")
        self.assertIn("--synthetic", arguments)
        self.assertIn("--aruco", arguments)
        self.assertIn("--mqtt", arguments)
        self.assertIn("--video", arguments)
        self.assertIn("--headless", arguments)
        self.assertEqual(arguments[arguments.index("--profile") + 1], "1080p30")
        self.assertEqual(arguments[arguments.index("--max-frames") + 1], "900")
        self.assertNotIn("--targets", arguments)
        self.assertNotIn("--target-model", arguments)

        for _, child_arguments in started.values():
            self.assertNotIn("--idle-timeout-s", child_arguments)

    def test_targets_are_forwarded_only_to_sender_for_both_frame_sources(self):
        for source_options in ([], ["--camera"]):
            with self.subTest(source_options=source_options):
                started = self.run_demo(["--targets", *source_options])
                arguments = started["sender"][1]

                self.assertIn("--targets", arguments)
                self.assertIn("--aruco", arguments)
                self.assertEqual("--synthetic" in arguments, not source_options)
                self.assertNotIn("--target-model", arguments)

                for name in ("broker", "video-receiver", "mqtt-subscriber"):
                    self.assertNotIn("--targets", started[name][1])
                    self.assertNotIn("--target-model", started[name][1])

                receiver_arguments = started["video-receiver"][1]
                self.assertIn("--idle-timeout-s", receiver_arguments)
                self.assertEqual(
                    receiver_arguments[receiver_arguments.index("--idle-timeout-s") + 1],
                    "60",
                )

                for name in ("broker", "sender", "mqtt-subscriber"):
                    self.assertNotIn("--idle-timeout-s", started[name][1])

    def test_explicit_model_path_is_resolved_from_callers_directory(self):
        with TemporaryDirectory() as temporary:
            caller_directory = Path(temporary)
            model = caller_directory / "weights" / "model with spaces.pt"
            model.parent.mkdir()
            model.write_bytes(b"Unused model fixture")

            started = self.run_demo(
                ["--targets", "--target-model", "weights/model with spaces.pt"],
                caller_directory=caller_directory,
            )

        arguments = started["sender"][1]
        self.assertIn("--targets", arguments)
        self.assertEqual(
            arguments[arguments.index("--target-model") + 1],
            str(model.resolve()),
        )

    def test_model_override_without_targets_is_rejected_before_startup(self):
        errors = StringIO()

        with patch(
            "run_local_demo.check_ports",
            side_effect=AssertionError("Invalid options reached the port check"),
        ), patch(
            "run_local_demo.Children",
            side_effect=AssertionError("Invalid options started child processes"),
        ), redirect_stderr(errors), self.assertRaises(SystemExit) as stopped:
            run_local_demo.main(["--target-model", "weights/custom.pt"])

        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("--target-model requires --targets", errors.getvalue())

    def test_explicit_model_path_expands_a_quoted_home_directory(self):
        started = self.run_demo(
            ["--targets", "--target-model", "~/weights/model with spaces.pt"]
        )
        arguments = started["sender"][1]

        self.assertEqual(
            arguments[arguments.index("--target-model") + 1],
            str((Path.home() / "weights/model with spaces.pt").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
