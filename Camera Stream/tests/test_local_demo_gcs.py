"""Check that one launcher owns the stream and its GCS forwarders."""

import base64
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import socket
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import patch

import run_local_demo
from test_local_demo_targets import ControlledChildren


class ForwarderChildren(ControlledChildren):
    def close(self):
        # Model the normal Ctrl+C response of both long-running forwarders.
        for name, process, _ in self.running:
            if name in ("gcs-log-forwarder", "gcs-image-forwarder"):
                process.returncode = 130

        super().close()


class LocalDemoGCSTests(unittest.TestCase):
    def run_demo(self, options):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = (Path(temporary.name) / "trial").resolve()
        children = None

        def create_children(root):
            nonlocal children
            children = ForwarderChildren(root)
            return children

        # Check the real parser and launcher without opening services or a camera.
        with patch("run_local_demo.check_ports"), patch(
            "run_local_demo.Children", side_effect=create_children
        ), redirect_stdout(StringIO()):
            code = run_local_demo.main([
                "--output-dir", str(output), *options,
            ])

        self.assertEqual(code, 0)
        self.assertTrue(children.closed)
        summary = json.loads((output / "summary.json").read_text())
        self.assertTrue(summary["verified"], summary)
        return output, children.started

    def test_preview_uses_this_runs_broker_and_capture_folder(self):
        output, started = self.run_demo([
            "--gcs", "preview", "--mqtt-port", "18884",
        ])
        self.assertIn("gcs-log-forwarder", started)
        self.assertIn("gcs-image-forwarder", started)
        log_script, log_args = started["gcs-log-forwarder"]
        image_script, image_args = started["gcs-image-forwarder"]

        self.assertEqual(log_script, "run_gcs_forwarder.py")
        self.assertEqual(image_script, "run_gcs_image_forwarder.py")
        self.assertEqual(log_args[log_args.index("--local-port") + 1], "18884")
        self.assertEqual(
            Path(image_args[image_args.index("--capture-dir") + 1]),
            output / "sender" / "detections",
        )
        self.assertIn("--capture-detections", started["sender"][1])

        for arguments in (log_args, image_args):
            # Preview must leave external publishing disabled for both consumers.
            self.assertNotIn("--publish", arguments)

        # Subscribe before results appear; open photos after their sender starts.
        order = list(started)
        self.assertLess(order.index("gcs-log-forwarder"), order.index("sender"))
        self.assertLess(order.index("sender"), order.index("gcs-image-forwarder"))

    def test_publish_gives_both_forwarders_the_selected_destination(self):
        _, started = self.run_demo([
            "--gcs", "publish",
            "--gcs-broker", "  192.0.2.10  ",
            "--gcs-port", "1884",
        ])

        for name in ("gcs-log-forwarder", "gcs-image-forwarder"):
            self.assertIn(name, started)
            arguments = started[name][1]
            self.assertIn("--publish", arguments)
            self.assertEqual(arguments[arguments.index("--broker") + 1], "192.0.2.10")
            self.assertEqual(arguments[arguments.index("--port") + 1], "1884")

    def test_auto_without_address_keeps_both_forwarders_in_preview(self):
        _, started = self.run_demo(["--gcs", "auto"])

        for name in ("gcs-log-forwarder", "gcs-image-forwarder"):
            self.assertIn(name, started)
            self.assertNotIn("--publish", started[name][1])

    def test_positional_address_publishes_both_logs_and_photos(self):
        for mode in ([], ["--gcs", "auto"], ["--gcs", "publish", "--gcs-broker", "192.0.2.10"]):
            with self.subTest(mode=mode):
                output, started = self.run_demo([*mode, "  192.0.2.10  "])

                for name in ("gcs-log-forwarder", "gcs-image-forwarder"):
                    arguments = started[name][1]
                    self.assertIn("--publish", arguments)
                    self.assertEqual(
                        arguments[arguments.index("--broker") + 1],
                        "192.0.2.10",
                    )
                    self.assertEqual(arguments[arguments.index("--port") + 1], "1883")

                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["gcs_mode"], "publish")
                self.assertEqual(summary["gcs_broker"], "192.0.2.10")

    def test_explicit_preview_or_off_prevents_publishing_with_an_address(self):
        for mode in ("preview", "off"):
            with self.subTest(mode=mode):
                _, started = self.run_demo(["192.0.2.10", "--gcs", mode])

                for name in ("gcs-log-forwarder", "gcs-image-forwarder"):
                    if mode == "off":
                        self.assertNotIn(name, started)

                    else:
                        self.assertIn(name, started)
                        self.assertNotIn("--publish", started[name][1])

    def test_positional_address_rejects_conflicting_options_before_startup(self):
        for options in (
            ["192.0.2.10", "--gcs-broker", "192.0.2.11"],
            ["   "],
        ):
            with self.subTest(options=options), patch(
                "run_local_demo.Children"
            ) as children, redirect_stderr(StringIO()), self.assertRaises(
                SystemExit
            ) as stopped:
                run_local_demo.main(options)

            self.assertEqual(stopped.exception.code, 2)
            children.assert_not_called()

    def test_disabled_gcs_does_not_start_forwarders(self):
        _, started = self.run_demo([])
        self.assertNotIn("gcs-log-forwarder", started)
        self.assertNotIn("gcs-image-forwarder", started)

    def test_publish_without_a_destination_stops_before_starting_services(self):
        with patch("run_local_demo.Children") as children, redirect_stderr(
            StringIO()
        ), self.assertRaises(SystemExit) as stopped:
            run_local_demo.main(["--gcs", "publish", "--gcs-broker", "   "])

        self.assertEqual(stopped.exception.code, 2)
        children.assert_not_called()


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("amqtt", "vidgear", "paho")),
    "Optional network requirements are not installed",
)
class LocalDemoGCSIntegrationTests(unittest.TestCase):
    def test_synthetic_preview_keeps_annotations_and_frame_identity(self):
        self.run_preview(continuous=False)

    @unittest.skipIf(sys.platform == "win32", "This test sends a POSIX interrupt")
    def test_continuous_preview_stops_every_service_on_ctrl_c(self):
        self.run_preview(continuous=True)

    def run_preview(self, *, continuous):
        # Reserve different loopback ports so this test leaves normal services alone.
        reservations = [socket.socket() for _ in range(3)]

        try:
            for reservation in reservations:
                reservation.bind(("127.0.0.1", 0))

            ports = [reservation.getsockname()[1] for reservation in reservations]

        finally:
            for reservation in reservations:
                reservation.close()

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "trial"
            command = [
                sys.executable, "-B", str(Path(run_local_demo.__file__)),
                "--profile", "720p30",
                "--max-frames", "0" if continuous else "90",
                "--gcs", "preview",
                "--output-dir", str(output),
                "--mqtt-port", str(ports[0]),
                "--video-port", str(ports[1]),
                "--http-port", str(ports[2]),
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

            try:
                if continuous:
                    # Stop only after both forwarders have handled a real detection.
                    deadline = monotonic() + 45

                    while process.poll() is None and monotonic() < deadline:
                        log_path = output / "gcs-logs.jsonl"
                        video_path = output / "receiver" / "frames.jsonl"
                        photos = output / "sender" / "detections"

                        if (
                            log_path.exists() and log_path.stat().st_size
                            and video_path.exists() and video_path.stat().st_size
                            and any(photos.glob("*.preview.json"))
                        ):
                            break

                        sleep(0.1)

                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)

                details, _ = process.communicate(timeout=60)

            finally:
                # Give the launcher a chance to close its children if the test fails.
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)

                    try:
                        process.communicate(timeout=15)

                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()

            self.assertEqual(process.returncode, 130 if continuous else 0, details)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted" if continuous else "complete")

            if not continuous:
                self.assertTrue(summary["verified"], summary)

            # Both previews must refer to a frame that this sender actually recorded.
            frames = run_local_demo.records(output / "sender" / "frames.jsonl")
            identities = {
                (frame["run_id"], frame["sequence_number"])
                for frame in frames
            }
            logs = run_local_demo.records(output / "gcs-logs.jsonl")
            self.assertTrue(logs, "The log forwarder did not prepare any messages")

            for entry in logs:
                self.assertEqual(entry["delivery"], "preview")
                self.assertEqual(entry["topic"], "gcs/drone_log")
                self.assertIn((entry["run_id"], entry["sequence_number"]), identities)
                self.assertTrue(entry["payload"]["action"])
                self.assertTrue(entry["payload"]["content"])

            captures = output / "sender" / "detections"
            previews = list(captures.glob("*.preview.json"))
            self.assertTrue(previews, "The image forwarder did not prepare a photo")

            for preview in previews:
                entry = json.loads(preview.read_text())
                event = json.loads((captures / entry["sidecar"]).read_text())
                annotated = captures / event["annotated_image"]["image_file"]
                original = captures / event["image_file"]
                self.assertEqual(entry["delivery"], "preview", entry)
                self.assertFalse(entry["publish_attempted"])
                self.assertEqual(entry["topic"], "gcs/detection")
                self.assertIn((entry["run_id"], entry["sequence_number"]), identities)
                self.assertEqual(entry["event_id"], event["event_id"])
                self.assertEqual(
                    base64.b64decode(entry["payload"]["image"]),
                    annotated.read_bytes(),
                )
                self.assertNotEqual(annotated.read_bytes(), original.read_bytes())

            # A successful preview must stop every child and release its listeners.
            for name in ("gcs-log-forwarder", "gcs-image-forwarder"):
                self.assertIn(summary["child_exit_codes"][name], (0, 130), summary)

            self.assertTrue(
                all(code is not None for code in summary["child_exit_codes"].values()),
                summary,
            )
            run_local_demo.check_ports(ports)


if __name__ == "__main__":
    unittest.main()
