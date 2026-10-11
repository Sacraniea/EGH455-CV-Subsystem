"""Send JSON messages to MQTT and wait for the broker to acknowledge them."""

from __future__ import annotations

from ipaddress import ip_address
import json
import re
from math import isfinite
from threading import Event


def _validate_host(host: str) -> None:
    """Accept a bare ASCII DNS name or IP literal, never a URL or host:port."""
    error = "MQTT host must be a plain hostname or IP address without URI, port, path or wildcard"

    # Reject empty hosts, non-ASCII text and characters used in URLs or wildcards.
    if (
        not isinstance(host, str)
        or not host
        or not host.isascii()
        or any(char.isspace() or char in "/\\%#?*+[]@" for char in host)
    ):
        raise ValueError(error)

    try:
        address = ip_address(host)

    except ValueError:
        # If it isn't an IP address, check that it is a valid hostname instead.
        name = host[:-1] if host.endswith(".") else host

        if (
            len(name) > 253
            or all(char in "0123456789." for char in name)
            or any(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                is None
                for label in name.split(".")
            )
        ):
            raise ValueError(error) from None

    else:
        # Reject unspecified, multicast and broadcast IPs as broker destinations.
        if (
            address.is_unspecified
            or address.is_multicast
            or host == "255.255.255.255"
        ):
            raise ValueError(
                "MQTT host must identify one broker, not a wildcard or broadcast address"
            )


def _validate_endpoint(host: str, port: int, topic: str, *, allow_remote: bool) -> None:
    # Validate a remote broker when enabled; otherwise require a loopback address.
    if allow_remote:
        _validate_host(host)

    elif host not in ("127.0.0.1", "localhost"):
        raise ValueError("MQTT host must be 127.0.0.1 or localhost for the local demo")

    # Reject a port unless it is an integer in the valid network-port range.
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65535
    ):
        raise ValueError("MQTT port must be an integer from 1 to 65535")

    # Require one concrete, nonempty topic rather than a wildcard subscription.
    if (
        not isinstance(topic, str)
        or not topic
        or any(char in topic for char in ("#", "+", "\0"))
    ):
        raise ValueError("MQTT topic must be nonempty without wildcards or NUL")

    # Reject topic text that exceeds MQTT's length limit after UTF-8 encoding.
    if len(topic.encode("utf-8")) > 65535:
        raise ValueError("MQTT topic is too long")


def validate_local_endpoint(host: str, port: int, topic: str) -> None:
    """Keep the demo subscriber and default publisher on the local machine."""
    _validate_endpoint(host, port, topic, allow_remote=False)


def positive_timeout(value: float) -> float:
    # Parse the timeout in seconds and reject zero, negative or non-finite values.
    value = float(value)

    if not isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")

    return value


def load_mqtt():
    """Leave paho optional for existing camera commands that do not use MQTT."""

    try:
        import paho.mqtt.client as mqtt

    except ImportError as error:
        raise RuntimeError(
            "MQTT requires paho-mqtt==2.1.0 in the stream environment"
        ) from error

    return mqtt


class PublishNotStartedError(ConnectionError):
    """The payload was never handed to the MQTT client and can be retried."""


class MqttResultPublisher:
    """Send one record at a time, with a time limit for each acknowledgement."""

    def __init__(
        self,
        host="127.0.0.1",
        port=18883,
        topic="taip/local/results",
        connect_timeout_s=3.0,
        publish_timeout_s=2.0,
        *,
        allow_remote=False,
    ):
        _validate_endpoint(host, port, topic, allow_remote=allow_remote)
        host = "127.0.0.1" if host == "localhost" else host
        connect_timeout_s = positive_timeout(connect_timeout_s)
        self._publish_timeout_s = positive_timeout(publish_timeout_s)
        self._topic = topic
        self._closed = False
        self._connected = Event()
        self._handshake = Event()
        self._connection_error = None
        self._mqtt = load_mqtt()
        # Create an MQTT 3.1.1 client with automatic reconnection disabled.
        self._client = self._mqtt.Client(
            callback_api_version=self._mqtt.CallbackAPIVersion.VERSION2,
            protocol=self._mqtt.MQTTv311,
            reconnect_on_failure=False,
        )
        self._client.connect_timeout = connect_timeout_s
        self._client.on_connect = self._on_connect
        self._client.on_connect_fail = self._on_connect_fail
        self._client.on_disconnect = self._on_disconnect

        try:
            # Start connecting in the background so setup can enforce a timeout.
            self._client.connect_async(host, port, keepalive=30)
            result = self._client.loop_start()

            if result != self._mqtt.MQTT_ERR_SUCCESS:
                raise ConnectionError(f"MQTT network loop could not start: {result}")

            # Wait for the connection callback; raise if the broker takes too long.
            if not self._handshake.wait(connect_timeout_s):
                raise TimeoutError(
                    f"MQTT connect acknowledgement timed out after {connect_timeout_s:g}s"
                )

            if self._connection_error or not self._connected.is_set():
                # Reject setup if the callback reported refusal or disconnection.
                raise ConnectionError(
                    self._connection_error or "MQTT disconnected during connection"
                )

        except BaseException:
            # Release the connection even if setup is interrupted with Ctrl+C.
            self.close()
            raise

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        # Save a broker rejection, or mark the connection ready for publishing.
        if reason_code.is_failure:
            self._connection_error = f"MQTT connection rejected: {reason_code}"

        else:
            self._connected.set()

        # Wake the waiting caller for both an accepted and a rejected connection.
        self._handshake.set()

    def _on_connect_fail(self, client, userdata):
        # Record the connection failure and release the waiting setup call.
        self._connection_error = "MQTT broker connection failed"
        self._handshake.set()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        # Prevent further publishing and record an unexpected lost connection.
        self._connected.clear()

        if not self._closed:
            self._connection_error = f"MQTT broker disconnected: {reason_code}"

        self._handshake.set()

    def publish(self, record: dict) -> None:
        # Reject a closed or disconnected publisher before handing over any payload.
        # PublishNotStartedError tells the image queue it can retry this event.
        if self._closed or not self._connected.is_set():
            raise PublishNotStartedError(
                self._connection_error or "MQTT publisher is closed or disconnected"
            )

        if not isinstance(record, dict):
            raise TypeError("MQTT result must be a JSON object")

        # Convert the record to compact JSON and reject NaN or infinite values.
        payload = json.dumps(record, allow_nan=False, separators=(",", ":"))
        # Publish on the configured topic with acknowledgement, without retaining it.
        info = self._client.publish(
            self._topic,
            payload=payload,
            qos=1,
            retain=False,
        )

        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(
                f"MQTT publish failed before acknowledgement: {info.rc}"
            )

        try:
            # Wait up to the configured limit for the broker to acknowledge this send.
            info.wait_for_publish(timeout=self._publish_timeout_s)
            acknowledged = info.is_published()

        except (RuntimeError, ValueError) as error:
            raise ConnectionError(
                f"MQTT publish failed before acknowledgement: {error}"
            ) from error

        if not acknowledged:
            # Report a lost connection or timeout if no acknowledgement arrived.
            # The payload may already have arrived, so this is not a safe retry signal.
            if not self._connected.is_set():
                raise ConnectionError(
                    self._connection_error or "MQTT disconnected before acknowledgement"
                )

            raise TimeoutError(
                f"MQTT publish acknowledgement timed out after {self._publish_timeout_s:g}s"
            )

    def close(self) -> None:
        # Disconnect once and stop the client's background network thread.
        if self._closed:
            return

        self._closed = True
        self._connected.clear()

        try:
            self._client.disconnect()

        finally:
            # Don't leave the MQTT background thread running after we close.
            self._client.loop_stop()
