"""Bound MQTT callback races in subprocesses so a regression cannot hang the suite."""

from pathlib import Path
import subprocess
import sys
import unittest


STREAM_DIRECTORY = Path(__file__).resolve().parents[1]

# These scenarios use Paho's real QoS 1 publish and PUBACK paths. No network loop
# or socket is opened; only the arrival and ordering of a broker ACK are supplied.
SCENARIOS = r'''
import struct
import sys
from threading import Event, Thread

import paho.mqtt.client as mqtt
import MQTT_config


PAYLOAD = {"action": "Valve detected", "content": "Closed valve"}


def acknowledge(client, mid):
    client._in_packet["remaining_length"] = 2
    client._in_packet["packet"] = struct.pack("!H", mid)
    client._handle_pubackcomp("PUBACK")


def lock_order_scenario():
    publish_entered = Event()
    ack_has_message_lock = Event()
    receipts = []
    errors = []

    class ObservedClient(mqtt.Client):
        observe_publish = False

        def publish(self, *args, **kwargs):
            if self.observe_publish:
                publish_entered.set()
            return super().publish(*args, **kwargs)

    client = ObservedClient(mqtt.CallbackAPIVersion.VERSION2)
    client._cv_log_receipts = MQTT_config.LogReceipts(receipts.append)
    client.on_publish = MQTT_config._on_publish
    MQTT_config.publish_log(client, PAYLOAD, event_id="run:1")
    first_mid = receipts[0]["mid"]
    client.observe_publish = True

    def receive_first_ack():
        try:
            # Paho holds this mutex when it calls on_publish. Arrange a second
            # publish at that exact point, without replacing either Paho path.
            with client._out_message_mutex:
                ack_has_message_lock.set()
                if not publish_entered.wait(1):
                    raise AssertionError("Second publish never reached Paho")
                acknowledge(client, first_mid)
        except BaseException as error:
            errors.append(repr(error))

    def publish_second_log():
        try:
            MQTT_config.publish_log(client, PAYLOAD, event_id="run:2")
        except BaseException as error:
            errors.append(repr(error))

    ack_thread = Thread(target=receive_first_ack)
    ack_thread.start()
    assert ack_has_message_lock.wait(1), "ACK thread did not acquire Paho mutex"
    publish_thread = Thread(target=publish_second_log)
    publish_thread.start()

    # The parent kills this entire child after two seconds if the locks deadlock.
    ack_thread.join()
    publish_thread.join()
    assert not errors, errors
    statuses = [(item["event_id"], item["status"]) for item in receipts]
    assert statuses == [
        ("run:1", "queued"),
        ("run:1", "broker_acknowledged"),
        ("run:2", "queued"),
    ], statuses
    MQTT_config.close_mqtt_client(client)
    assert receipts[-1]["event_id"] == "run:2", receipts
    assert receipts[-1]["status"] == "delivery_unconfirmed_on_stop", receipts
    print("LOCK_ORDER_OK")


def early_ack_scenario():
    receipts = []

    class ImmediateAckClient(mqtt.Client):
        def publish(self, *args, **kwargs):
            info = super().publish(*args, **kwargs)
            # Exercise the earliest possible ACK: before the caller learns mid.
            acknowledge(self, info.mid)
            return info

    client = ImmediateAckClient(mqtt.CallbackAPIVersion.VERSION2)
    client._cv_log_receipts = MQTT_config.LogReceipts(receipts.append)
    client.on_publish = MQTT_config._on_publish
    MQTT_config.publish_log(client, PAYLOAD, event_id="run:8")
    MQTT_config.close_mqtt_client(client)

    assert [item["status"] for item in receipts] == [
        "queued", "broker_acknowledged",
    ], receipts
    assert all(item["event_id"] == "run:8" for item in receipts), receipts
    assert all(item["payload"] == PAYLOAD for item in receipts), receipts
    assert len({item["mid"] for item in receipts}) == 1, receipts
    print("EARLY_ACK_OK")


if sys.argv[1] == "lock-order":
    lock_order_scenario()
else:
    early_ack_scenario()
'''


class MQTTLockingTests(unittest.TestCase):
    def run_scenario(self, scenario, expected_marker):
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-c", SCENARIOS, scenario],
                cwd = STREAM_DIRECTORY,
                capture_output = True,
                text = True,
                timeout = 2,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"MQTT {scenario} deadlocked; its subprocess was stopped")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(expected_marker, result.stdout)

    def test_ack_of_previous_log_cannot_deadlock_next_real_paho_publish(self):
        self.run_scenario("lock-order", "LOCK_ORDER_OK")

    def test_ack_before_publish_returns_retains_identity_and_queued_then_ack_order(self):
        self.run_scenario("early-ack", "EARLY_ACK_OK")


if __name__ == "__main__":
    unittest.main()
