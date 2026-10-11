import json
import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from run_local_demo import check_ports, verify_delivery, complete_trial


class LocalDemoTests(unittest.TestCase):
    def test_occupied_loopback_port_is_refused(self):
        # Occupy a local port and check that the demo's port check refuses it.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))

            with self.assertRaisesRegex(OSError, "already in use"):
                check_ports([listener.getsockname()[1]])

    def test_duplicate_ports_are_refused(self):
        # Assign the same port to two services and check that validation rejects it.
        with self.assertRaises(ValueError):
            check_ports([5555, 5555])

    def test_http_preflight_checks_the_requested_interface(self):
        # Find three free local ports, then check HTTP on a different host address.
        sockets = [socket.socket() for _ in range(3)]

        try:
            for sock in sockets:
                sock.bind(("127.0.0.1", 0))

            ports = [sock.getsockname()[1] for sock in sockets]

        finally:
            for sock in sockets:
                sock.close()

        # Check that binding the unassigned HTTP address fails despite the free ports.
        with self.assertRaisesRegex(OSError, "203.0.113.42"):
            check_ports(ports, http_host="203.0.113.42")

    def test_verification_matches_actual_timestamps_and_all_mqtt_results(self):
        # Write matching sender, MQTT and video records and compare their identities.
        record = dict(
            run_id="trial",
            sequence_number=1,
            device_timestamp_s=2.0,
            host_received_timestamp_s=2.01,
            image_size_px=[1280, 720],
            source_kind="synthetic",
            processors=[],
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sender").mkdir()
            (root / "receiver").mkdir()

            for path in ("sender/frames.jsonl", "mqtt.jsonl", "receiver/frames.jsonl"):
                (root / path).write_text(json.dumps(record) + "\n")

            self.assertTrue(verify_delivery(root)["verified"])
            # Repeat an identical MQTT record and check that delivery still verifies.
            (root / "mqtt.jsonl").write_text((json.dumps(record) + "\n") * 2)
            self.assertTrue(verify_delivery(root)["verified"])
            # Change only the video timestamp and check that verification now fails.
            record["device_timestamp_s"] = 3.0
            (root / "receiver/frames.jsonl").write_text(json.dumps(record) + "\n")
            self.assertFalse(verify_delivery(root)["verified"])

    def test_polling_waits_for_partial_record_but_final_verification_is_strict(self):
        # Compare a partial MQTT write while polling, then during final verification.
        record = dict(
            run_id="trial",
            sequence_number=1,
            device_timestamp_s=2.0,
            host_received_timestamp_s=2.01,
            image_size_px=[1280, 720],
            source_kind="synthetic",
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sender").mkdir()
            (root / "receiver").mkdir()

            for path in ("sender/frames.jsonl", "receiver/frames.jsonl"):
                (root / path).write_text(json.dumps(record) + "\n")

            # Write a partial line; polling ignores it but the final read raises.
            (root / "mqtt.jsonl").write_text(json.dumps(record)[:10])
            self.assertFalse(verify_delivery(root, allow_partial=True)["verified"])

            with self.assertRaises(json.JSONDecodeError):
                verify_delivery(root)

            # Finish the MQTT line and check that the three records now match.
            (root / "mqtt.jsonl").write_text(json.dumps(record) + "\n")
            self.assertTrue(verify_delivery(root, allow_partial=True)["verified"])

    def test_final_check_includes_trailing_records_and_receiver_exit_failures(self):
        # Verify records written during shutdown and reject a failed receiver exit.
        from types import SimpleNamespace

        record = dict(
            run_id="trial",
            sequence_number=1,
            device_timestamp_s=2.0,
            host_received_timestamp_s=2.01,
            image_size_px=[1280, 720],
            source_kind="synthetic",
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sender").mkdir()
            (root / "receiver").mkdir()

            for path in ("sender/frames.jsonl", "mqtt.jsonl", "receiver/frames.jsonl"):
                (root / path).write_text(json.dumps(record) + "\n")

            process = SimpleNamespace(returncode=0)

            class LateReceiver:
                running = [("video-receiver", process, None)]

                def close(self):
                    # Append the receiver's final frame when the trial closes it.
                    with (root / "receiver/frames.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")

            final = complete_trial(root, LateReceiver())
            self.assertEqual(final["video_frames"], 2)
            self.assertTrue(final["verified"])
            # Repeat shutdown with a failed receiver exit and check the trial fails.
            process.returncode = 1
            failed = complete_trial(root, LateReceiver())
            self.assertFalse(failed["verified"])
            self.assertIn("video-receiver", failed["error"])
            (root / "receiver/frames.jsonl").write_text("")
            self.assertFalse(verify_delivery(root)["verified"])


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("amqtt", "vidgear", "paho")),
    "Optional network requirements are not installed",
)
class LocalDemoIntegrationTests(unittest.TestCase):
    def test_real_broker_video_and_subscriber_delivery_and_cleanup(self):
        self.run_demo(expect_success=True)

    def test_failed_sender_stops_its_broker_and_receivers(self):
        self.run_demo(expect_success=False)

    def test_explicit_http_host_is_used_for_the_browser_url(self):
        self.run_demo(expect_success=True, explicit_http_host=True)

    def test_selected_profile_reaches_sender_and_received_video(self):
        for profile, image_size in (("720p30", [1280, 720]), ("1080p30", [1920, 1080])):
            with self.subTest(profile=profile):
                self.run_demo(
                    expect_success=True,
                    profile=profile,
                    image_size=image_size,
                )

    def test_capture_option_reaches_sender_and_preserves_frame_identity(self):
        self.run_demo(expect_success=True, capture=True)

    def run_demo(
        self,
        *,
        expect_success,
        explicit_http_host=False,
        profile=None,
        image_size=None,
        capture=False,
    ):
        # Run a local demo and check its saved records, selected options and cleanup.
        # Reserve three sockets together to obtain a different port for each service.
        reservations = []

        try:
            for _ in range(3):
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                reservations.append(sock)

            ports = [sock.getsockname()[1] for sock in reservations]

        finally:
            for sock in reservations:
                sock.close()

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "trial"
            # Build a 30-frame command using the temporary folder and chosen ports.
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve().parents[1] / "run_local_demo.py"),
                "--max-frames",
                "30",
                "--output-dir",
                str(output),
                "--mqtt-port",
                str(ports[0]),
                "--video-port",
                str(ports[1]),
                "--http-port",
                str(ports[2]),
            ]

            if not expect_success:
                # Request an ID outside this marker dictionary to make the sender fail.
                command.extend(["--permitted-ids", "100"])

            if explicit_http_host:
                command.extend(["--http-host", "localhost"])

            if profile:
                command.extend(["--profile", profile])

            if capture:
                command.append("--capture-detections")

            # Run the demo and read the summary produced after its services shut down.
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=45.0,
            )
            self.assertTrue(
                (output / "summary.json").exists(), completed.stdout + completed.stderr
            )
            summary = json.loads((output / "summary.json").read_text())
            details = completed.stdout + completed.stderr + "\n" + json.dumps(summary)

            if expect_success:
                # Check all MQTT results, received video frames and the preview URL.
                self.assertEqual(completed.returncode, 0, details)
                self.assertTrue(summary["verified"], details)
                self.assertEqual(summary["mqtt_unique_frames"], 30)
                self.assertGreater(summary["matched_video_and_mqtt_frames"], 0)
                self.assertTrue((output / "receiver/latest.jpg").exists())
                self.assertEqual(
                    summary["preview_url"], f"http://127.0.0.1:{ports[2]}/stream.mjpeg"
                )
                receiver = json.loads((output / "receiver.ready.json").read_text())
                self.assertEqual(receiver["url"], summary["preview_url"])

                if capture:
                    # Match the saved detection event to its frame in the sender's log.
                    events = list((output / "sender/detections").glob("*.event.json"))
                    self.assertEqual(len(events), 1)
                    event = json.loads(events[0].read_text())
                    frames = [
                        json.loads(line)
                        for line in (output / "sender/frames.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    self.assertEqual(
                        event["record"],
                        next(
                            frame
                            for frame in frames
                            if frame["sequence_number"] == event["sequence_number"]
                        ),
                    )
                    self.assertTrue(
                        (output / "sender/detections" / event["image_file"]).exists()
                    )

                if image_size is not None:
                    # Check the profile size in every record and the received image.
                    import cv2

                    manifest = json.loads((output / "sender/run.json").read_text())
                    self.assertEqual(manifest["metadata"]["profile"]["name"], profile)

                    for path in (
                        "sender/frames.jsonl",
                        "mqtt.jsonl",
                        "receiver/frames.jsonl",
                    ):
                        frames = [
                            json.loads(line)
                            for line in (output / path).read_text().splitlines()
                        ]
                        self.assertTrue(frames, path)
                        self.assertTrue(
                            all(
                                frame["image_size_px"] == image_size for frame in frames
                            ),
                            path,
                        )

                    image = cv2.imread(str(output / "receiver/latest.jpg"))
                    self.assertEqual(list(image.shape[:2][::-1]), image_size)

            else:
                # Check that the invalid sender settings produced a failed trial.
                self.assertEqual(completed.returncode, 1, details)
                self.assertEqual(summary["status"], "failed")

            self.assertTrue(
                all(code is not None for code in summary["child_exit_codes"].values()),
                details,
            )
            # Check that all three ports can be opened again after the demo exits.
            check_ports(ports)
