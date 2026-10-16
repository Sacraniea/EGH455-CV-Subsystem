"""Connect to the GCS broker and publish formatted detection logs."""

import json
from threading import Lock
import paho.mqtt.client as mqtt


class LogReceipts:
    """Match MQTT message numbers to their source frame until the broker replies."""

    def __init__(self, on_event):
        self.lock = Lock()
        self.pending = {}
        self.publishing = 0
        self.early_acknowledgements = {}
        self.on_event = on_event

    def record(self, status, details):
        if self.on_event is not None:
            self.on_event({"status": status, **details})


def _receipts(client):
    receipts = getattr(client, "_cv_log_receipts", None)
    return receipts if isinstance(receipts, LogReceipts) else None


def _on_connect(client, userdata, flags, reason_code, properties):
    """Report whether the broker accepted our connection."""

    # Only report a successful connection after the broker accepts us.
    if reason_code.is_failure:
        print(f"MQTT connection rejected: {reason_code}")

    else:
        print(f"MQTT connected to {client.host}:{client.port}")


def _on_connect_fail(client, userdata):
    """Report an unsuccessful attempt to reach the broker."""

    # Paho will retry using the delay configured when creating the client.
    print(
        f"MQTT could not reach {client.host}:{client.port}; "
        "retrying."
    )


def _on_disconnect(client, userdata, flags, reason_code, properties):
    """Report when the connection closes."""

    if reason_code.is_failure:
        print(f"MQTT connection lost: {reason_code}; reconnecting.")

    else:
        print("MQTT disconnected.")


def _on_publish(client, userdata, mid, reason_code, properties):
    """Report when the broker acknowledges one of our messages."""

    # With QoS 1, this confirms broker receipt, not storage in the GCS database.
    receipts = _receipts(client)
    if receipts is not None:
        with receipts.lock:
            details = receipts.pending.pop(mid, None)
            status = "broker_rejected" if reason_code.is_failure else "broker_acknowledged"
            if details is not None:
                receipts.record(status, details)
            elif receipts.publishing:
                # Remember an acknowledgement that beat publish() returning its number.
                receipts.early_acknowledgements[mid] = status

    if reason_code.is_failure:
        print(f"MQTT broker rejected message {mid}: {reason_code}")
    else:
        print(f"MQTT broker acknowledged message {mid}.")


def publish_log(client, payload, *, event_id = None):
    """Queue one action/content message and keep its source identity locally."""

    receipts = _receipts(client)
    details = {"event_id": event_id, "payload": payload}

    try:
        message = json.dumps(payload, allow_nan = False)
    except (TypeError, ValueError) as error:
        if receipts is not None:
            receipts.record("not_queued", {**details, "error": str(error)})
        print(f"MQTT log not queued: invalid JSON ({error}).")
        return

    if receipts is not None:
        with receipts.lock:
            receipts.publishing += 1

    # Do not hold our receipt lock during publish: Paho uses its own lock while
    # delivering acknowledgements, and the opposite lock order would deadlock.
    try:
        info = client.publish(
            topic = "gcs/drone_log",
            payload = message,
            qos = 1,
            retain = False,
        )
    except (OSError, ValueError) as error:
        if receipts is not None:
            with receipts.lock:
                receipts.publishing -= 1
                receipts.early_acknowledgements.clear()
                receipts.record("delivery_unconfirmed", {**details, "error": str(error)})
        print(f"MQTT log delivery unconfirmed: {error}")
        return

    details["mid"] = info.mid
    queued = info.rc in (mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_NO_CONN)
    if receipts is not None:
        with receipts.lock:
            receipts.publishing -= 1
            early_acknowledgement = receipts.early_acknowledgements.pop(info.mid, None)
            receipts.record("queued" if queued else "not_queued", details)
            if queued:
                if early_acknowledgement is not None:
                    receipts.record(early_acknowledgement, details)
                else:
                    receipts.pending[info.mid] = details

    if info.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"MQTT queued {info.mid}: {payload['action']} | {payload['content']}")
    elif info.rc == mqtt.MQTT_ERR_NO_CONN:
        print(f"MQTT message {info.mid} queued, awaiting reconnection.")
    elif info.rc == mqtt.MQTT_ERR_QUEUE_SIZE:
        print("MQTT log not queued: outgoing queue is full.")
    else:
        print(f"MQTT publish error: {mqtt.error_string(info.rc)}")


def create_mqtt_client(host, port = 1883, *, on_event = None):
    """Start a background connection to the configured GCS broker."""

    # Use the current callback interface and our existing MQTT 3.1.1 protocol.
    client = mqtt.Client(
        callback_api_version = mqtt.CallbackAPIVersion.VERSION2,
        protocol = mqtt.MQTTv311,
        reconnect_on_failure = True,
    )

    client._cv_log_receipts = LogReceipts(on_event)

    # Tell Paho which functions should report connection changes.
    client.on_connect = _on_connect
    client.on_connect_fail = _on_connect_fail
    client.on_disconnect = _on_disconnect
    client.on_publish = _on_publish

    # Limit each socket connection attempt and space out repeated attempts.
    client.connect_timeout = 3.0
    client.reconnect_delay_set(
        min_delay = 1,
        max_delay = 30,
    )

    # Limit queued acknowledgement-based messages if delivery falls behind.
    client.max_queued_messages_set(20)

    try:
        # Set the destination without making the camera wait for a connection.
        client.connect_async(
            host,
            port,
            keepalive = 30,
        )

        # Start the background thread that handles MQTT network traffic.
        result = client.loop_start()

        if result != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT network loop could not start: {result}")

    except BaseException:
        # Release resources if setup fails or is interrupted.
        close_mqtt_client(client)
        raise

    return client


def close_mqtt_client(client):
    """Disconnect and stop the MQTT background thread."""

    try:
        client.disconnect()

    finally:
        # Stop the network thread even if disconnecting raises an error.
        client.loop_stop()

        # Preserve uncertain delivery when stopping before the broker replies.
        receipts = _receipts(client)
        if receipts is not None:
            with receipts.lock:
                for details in receipts.pending.values():
                    receipts.record("delivery_unconfirmed_on_stop", details)
                receipts.pending.clear()
