"""Keep GCS delivery outcomes tied to the frame that produced the message."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import paho.mqtt.client as mqtt
import MQTT_config


class MQTTReceiptTests(unittest.TestCase):
    def connected_client(self, events):
        client = Mock()
        client.loop_start.return_value = mqtt.MQTT_ERR_SUCCESS
        client.publish.return_value = SimpleNamespace(rc = mqtt.MQTT_ERR_SUCCESS, mid = 42)
        with patch.object(MQTT_config.mqtt, "Client", return_value = client):
            try:
                MQTT_config.create_mqtt_client("example.invalid", on_event = events.append)
            except TypeError as error:
                self.fail(f"MQTT needs frame-linked receipts: {error}")
        return client

    def test_publish_ack_and_shutdown_distinguish_delivery_outcomes(self):
        events = []
        client = self.connected_client(events)
        payload = {"action": "Valve detected", "content": "Closed valve"}
        MQTT_config.publish_log(client, payload, event_id = "run:4")
        self.assertEqual(events[-1]["status"], "queued")
        self.assertEqual(events[-1]["event_id"], "run:4")

        client.on_publish(client, None, 42, SimpleNamespace(is_failure = False), None)
        self.assertEqual(events[-1]["status"], "broker_acknowledged")
        self.assertEqual(events[-1]["event_id"], "run:4")
        self.assertEqual(events[-1]["payload"], payload)

        client.publish.return_value.mid = 43
        MQTT_config.publish_log(client, payload, event_id = "run:5")
        MQTT_config.close_mqtt_client(client)
        self.assertEqual(events[-1]["status"], "delivery_unconfirmed_on_stop")
        self.assertEqual(events[-1]["event_id"], "run:5")

    def test_full_queue_is_not_recorded_as_queued_or_acknowledged(self):
        events = []
        client = self.connected_client(events)
        client.publish.return_value.rc = mqtt.MQTT_ERR_QUEUE_SIZE
        MQTT_config.publish_log(client, {"action": "Gauge detected", "content": "Pressure unavailable"}, event_id = "run:9")
        self.assertEqual(events[-1]["status"], "not_queued")
        MQTT_config.close_mqtt_client(client)
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
