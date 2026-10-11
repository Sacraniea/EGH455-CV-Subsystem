"""Check continuous subscribers without opening a camera or network connection."""

import contextlib
import io
import json
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_gcs_forwarder
import run_mqtt_subscriber


class SubscriberRuntimeTests(unittest.TestCase):
    def mqtt_client(self, *, payload=None):
        # Acknowledge setup immediately and optionally deliver one local frame record.
        client = Mock()
        mqtt = SimpleNamespace(
            Client=Mock(return_value=client),
            CallbackAPIVersion=SimpleNamespace(VERSION2=2),
            MQTTv311=4,
            MQTT_ERR_SUCCESS=0,
        )

        def start():
            client.on_subscribe(
                client, None, 1, [SimpleNamespace(is_failure=False)], None
            )

            if payload is not None:
                client.on_message(
                    client, None, SimpleNamespace(payload=json.dumps(payload).encode())
                )

            return 0

        client.loop_start.side_effect = start
        return mqtt, client

    def test_continuous_subscription_can_still_stop_at_message_count(self):
        mqtt, client = self.mqtt_client(payload={"run_id": "continuous", "processors": []})
        args = run_mqtt_subscriber.build_parser().parse_args([
            "--timeout-s", "0", "--max-messages", "1",
        ])
        records = []

        with (
            patch("run_mqtt_subscriber.load_mqtt", return_value=mqtt),
            patch("run_mqtt_subscriber.monotonic", side_effect=AssertionError("No deadline")),
        ):
            count = run_mqtt_subscriber.receive(
                args, on_record=records.append, printer=lambda *a, **k: None
            )

        self.assertEqual(count, 1)
        self.assertEqual(records[0]["run_id"], "continuous")
        self.assertEqual(client.connect_timeout, 3.0)
        client.disconnect.assert_called_once()
        client.loop_stop.assert_called_once()

    def test_continuous_subscription_waits_for_ctrl_c_and_closes(self):
        mqtt, client = self.mqtt_client()
        messages = Mock()
        messages.get.side_effect = [Empty, KeyboardInterrupt]

        with (
            patch("run_mqtt_subscriber.load_mqtt", return_value=mqtt),
            patch("run_mqtt_subscriber.Queue", return_value=messages),
            patch("run_mqtt_subscriber.monotonic", side_effect=AssertionError("No deadline")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = run_mqtt_subscriber.main(["--timeout-s", "0"])

        self.assertEqual(code, 130)
        self.assertEqual(messages.get.call_count, 2)
        self.assertEqual(messages.get.call_args.kwargs["timeout"], 0.1)
        client.disconnect.assert_called_once()
        client.loop_stop.assert_called_once()

    def test_continuous_subscription_still_limits_setup_to_three_seconds(self):
        mqtt, client = self.mqtt_client()
        client.loop_start.side_effect = None
        client.loop_start.return_value = 0
        subscribed, failed = Mock(), Mock()
        subscribed.wait.return_value = False
        args = run_mqtt_subscriber.build_parser().parse_args(["--timeout-s", "0"])

        with (
            patch("run_mqtt_subscriber.load_mqtt", return_value=mqtt),
            patch("run_mqtt_subscriber.Event", side_effect=[subscribed, failed]),
            self.assertRaisesRegex(TimeoutError, "acknowledgement timed out"),
        ):
            run_mqtt_subscriber.receive(args)

        subscribed.wait.assert_called_once_with(3.0)
        client.disconnect.assert_called_once()
        client.loop_stop.assert_called_once()

    def test_finite_runtime_keeps_its_deadline_and_unmet_count_error(self):
        for count in (0, 1):
            with self.subTest(max_messages=count):
                mqtt, client = self.mqtt_client()
                args = run_mqtt_subscriber.build_parser().parse_args([
                    "--timeout-s", "0.5", "--max-messages", str(count),
                ])

                with (
                    patch("run_mqtt_subscriber.load_mqtt", return_value=mqtt),
                    patch("run_mqtt_subscriber.monotonic", side_effect=[100.0, 100.5]),
                ):
                    if count:
                        with self.assertRaisesRegex(TimeoutError, "Received 0/1"):
                            run_mqtt_subscriber.receive(args, printer=lambda *a, **k: None)

                    else:
                        self.assertEqual(
                            run_mqtt_subscriber.receive(args, printer=lambda *a, **k: None),
                            0,
                        )

                self.assertEqual(client.connect_timeout, 0.5)
                client.disconnect.assert_called_once()
                client.loop_stop.assert_called_once()

    def test_negative_and_nonfinite_runtime_rejected_before_mqtt_setup(self):
        for timeout in (-1, float("inf"), -float("inf"), float("nan")):
            with self.subTest(timeout=timeout):
                args = run_mqtt_subscriber.build_parser().parse_args([])
                args.timeout_s = timeout

                with (
                    patch("run_mqtt_subscriber.load_mqtt") as load,
                    self.assertRaises(ValueError),
                ):
                    run_mqtt_subscriber.receive(args)

                load.assert_not_called()


class GcsContinuousRuntimeTests(unittest.TestCase):
    def test_zero_runtime_reaches_subscription_and_returns_interrupt_status(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "logs.jsonl"

            with patch("run_gcs_forwarder.receive", side_effect=KeyboardInterrupt) as receive:
                code = run_gcs_forwarder.main(
                    ["--timeout-s", "0", "--output", str(output)],
                    printer=lambda *a, **k: None,
                )

            self.assertEqual(code, 130)
            self.assertEqual(receive.call_args.args[0].timeout_s, 0.0)
            self.assertEqual(output.read_text(), "")

    def test_invalid_runtime_is_rejected_before_subscription(self):
        for timeout in ("-1", "inf", "-inf", "nan"):
            with self.subTest(timeout=timeout), patch("run_gcs_forwarder.receive") as receive:
                code = run_gcs_forwarder.main(
                    [f"--timeout-s={timeout}"], printer=lambda *a, **k: None
                )

                self.assertEqual(code, 1)
                receive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
