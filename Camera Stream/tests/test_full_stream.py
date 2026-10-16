"""Check the complete path with generated frames and a private local broker."""

import base64
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from urllib.request import urlopen
import unittest

import cv2
import numpy as np
import paho.mqtt.client as mqtt


DIRECTORY = Path(__file__).resolve().parents[1]
BROKER = '''
import asyncio, sys
from amqtt.broker import Broker
async def main():
    broker = Broker({
        "listeners": {"default": {"type": "tcp", "bind": "127.0.0.1:" + sys.argv[1]}},
        "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}},
    })
    await broker.start()
    print("ready", flush=True)
    await asyncio.sleep(60)
asyncio.run(main())
'''


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_until(check, timeout = 10):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if check():
            return
        sleep(0.05)
    raise AssertionError("Timed out waiting for the local test service")


class FullStreamTests(unittest.TestCase):
    def test_synthetic_video_logs_and_annotated_photo_share_the_saved_run(self):
        broker_port, http_port, video_port = free_port(), free_port(), free_port()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            messages = []
            subscribed = Event()
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.on_connect = lambda client, userdata, flags, reason, properties: client.subscribe("gcs/#", qos=1)
            client.on_subscribe = lambda *args: subscribed.set()
            client.on_message = lambda client, userdata, message: messages.append(
                (message.topic, json.loads(message.payload))
            )
            broker = stream = None
            with (root / "broker.log").open("w+") as broker_log, (root / "stream.log").open("w+") as stream_log:
                try:
                    broker = subprocess.Popen(
                        [sys.executable, "-u", "-c", BROKER, str(broker_port)],
                        stdout = broker_log,
                        stderr = subprocess.STDOUT,
                    )
                    wait_until(lambda: "ready" in (root / "broker.log").read_text())
                    client.connect("127.0.0.1", broker_port)
                    client.loop_start()
                    self.assertTrue(subscribed.wait(5))
                    stream = subprocess.Popen([
                        sys.executable, "-B", "-u", str(DIRECTORY / "run_subsystem.py"),
                        "127.0.0.1", "--gcs-port", str(broker_port),
                        "--source", "synthetic", "--no-targets", "--profile", "480p20",
                        "--max-frames", "90", "--http-port", str(http_port),
                        "--video-port", str(video_port), "--output-dir", str(root / "run"),
                    ], stdout = stream_log, stderr = subprocess.STDOUT)
                    wait_until(lambda: "Browser stream:" in (root / "stream.log").read_text())

                    # Read one complete multipart image rather than only checking HTTP 200.
                    with urlopen(f"http://127.0.0.1:{http_port}/stream.mjpeg", timeout=10) as response:
                        self.assertIn("multipart", response.headers["Content-Type"])
                        self.assertEqual(response.readline().strip(), b"--frame")
                        headers = {}
                        while True:
                            line = response.readline().strip()
                            if not line:
                                break
                            key, value = line.decode().split(":", 1)
                            headers[key] = value.strip()
                        jpeg = response.read(int(headers["Content-Length"]))
                        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        self.assertEqual(decoded.shape[:2], (480, 640))

                    self.assertEqual(stream.wait(timeout=20), 0, (root / "stream.log").read_text())
                    wait_until(lambda: any(topic == "gcs/detection" for topic, _ in messages), timeout=2)
                    summary = json.loads((root / "run/run.json").read_text())
                    records = [json.loads(line) for line in (root / "run/frames.jsonl").read_text().splitlines()]
                    self.assertEqual(summary["frames_recorded"], 90)
                    self.assertEqual(headers["X-Run-ID"], summary["run_id"])
                    shown = next(record for record in records if record["sequence_number"] == int(headers["X-Frame-Sequence"]))
                    self.assertEqual(shown["processors"][0]["processor"], "aruco")
                    self.assertTrue(any(result["valid"] for result in shown["processors"][0]["data"]["results"]))

                    logs = [payload for topic, payload in messages if topic == "gcs/drone_log"]
                    self.assertTrue(logs)
                    self.assertEqual(set(logs[0]), {"action", "content"})
                    receipts = [json.loads(line) for line in (root / "run/gcs-logs.jsonl").read_text().splitlines()]
                    self.assertTrue(any(entry["status"] == "broker_acknowledged" for entry in receipts))
                    photos = [payload for topic, payload in messages if topic == "gcs/detection"]
                    self.assertEqual(len(photos), 1)
                    self.assertEqual(set(photos[0]), {"type", "description", "mime", "image", "timestamp"})
                    event = json.loads(next((root / "run/detections").glob("*.event.json")).read_text())
                    annotation = root / "run/detections" / event["annotated_image"]["image_file"]
                    self.assertEqual(base64.b64decode(photos[0]["image"]), annotation.read_bytes())
                    self.assertEqual(event["run_id"], summary["run_id"])
                    self.assertIn(event["record"], records)
                    original = cv2.imread(str(root / "run/detections" / event["image_file"]))
                    annotated = cv2.imread(str(annotation))
                    self.assertEqual(original.shape, annotated.shape)
                    self.assertFalse(np.array_equal(original, annotated))
                finally:
                    if stream is not None and stream.poll() is None:
                        stream.terminate()
                        stream.wait(timeout=15)
                    client.disconnect()
                    client.loop_stop()
                    if broker is not None and broker.poll() is None:
                        broker.terminate()
                        broker.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
