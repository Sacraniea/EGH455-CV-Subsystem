"""Send annotated JPEG frames and their capture metadata over local NetGear sockets."""

from __future__ import annotations

from ipaddress import IPv4Address

import numpy as np

from stream_types import FramePacket


def _load_netgear():
    """Keep optional network dependencies out of ordinary local capture."""
    from vidgear.gears import NetGear

    return NetGear


def _local_endpoint(host: str, port: int) -> tuple[str, str]:
    # Return a loopback IP and valid port as strings for NetGear's constructor.
    if host == "localhost":
        host = "127.0.0.1"

    try:
        address = IPv4Address(host)

    except (ValueError, TypeError) as error:
        raise ValueError("video transport requires an IPv4 loopback address") from error

    if not address.is_loopback:
        # Reject off-machine video destinations; the HTTP bridge handles LAN viewing.
        raise ValueError("video transport requires an IPv4 loopback address")

    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("video port must be an integer between 1 and 65535")

    return str(address), str(port)


class NetGearVideoSender:
    """Publish annotated BGR packets without waiting for a local receiver."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        address, network_port = _local_endpoint(host, port)
        self._closed = False
        # Connect a publisher that sends without waiting for receiver acknowledgements.
        self._transport = _load_netgear()(
            address=address,
            port=network_port,
            protocol="tcp",
            pattern=2,
            receive_mode=False,
            logging=False,
            # Enable the mode that carries frame metadata alongside each JPEG.
            multiserver_mode=True,
            jpeg_compression="BGR",
            jpeg_compression_quality=90,
        )

    def send(
        self, packet: FramePacket, *, run_id: str, source_kind: str = "camera"
    ) -> None:
        """Send the supplied image and its original capture identity together."""

        if self._closed:
            raise RuntimeError("video sender is closed")

        # Copy the frame's identity and source dimensions into the outgoing metadata.
        height, width = packet.image.shape[:2]
        metadata = {
            "schema_version": 1,
            "producer": "camera_stream",
            "source": "depthai" if source_kind == "camera" else source_kind,
            "source_kind": source_kind,
            "run_id": run_id,
            "sequence_number": packet.sequence_number,
            "device_timestamp_s": packet.device_timestamp_s,
            "host_received_timestamp_s": packet.host_received_timestamp_s,
            "image_size_px": [width, height],
        }
        # Send the annotated frame and metadata together; do not wait for an ACK.
        self._transport.send(packet.image, message=metadata)

    def close(self) -> None:
        """Send NetGear's best-effort termination message and release the socket."""

        if not self._closed:
            self._closed = True
            self._transport.close()


class NetGearVideoReceiver:
    """Receive local frames in a dedicated process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        *,
        subscriber_timeout_s: int = 10,
    ):
        address, network_port = _local_endpoint(host, port)

        if type(subscriber_timeout_s) is not int or subscriber_timeout_s <= 0:
            raise ValueError("subscriber_timeout_s must be a positive integer")

        self._closed = False
        # Bind a local subscriber that receives JPEG images and their frame metadata.
        self._transport = _load_netgear()(
            address=address,
            port=[network_port],
            protocol="tcp",
            pattern=2,
            receive_mode=True,
            logging=False,
            multiserver_mode=True,
            # Stop receiving after this idle time, including before the first frame.
            subscriber_timeout=subscriber_timeout_s,
            jpeg_compression="BGR",
            jpeg_compression_quality=90,
        )

    def receive(self) -> tuple[dict, np.ndarray] | None:
        """Return one metadata/image pair, or None on termination or idle timeout."""

        if self._closed:
            return None

        received = self._transport.recv()

        if received is None:
            # Return None for either sender termination or receiver idle timeout.
            return None

        # Extract metadata and pixels from NetGear's (port, metadata, image) tuple.
        _, metadata, image = received

        if metadata["image_size_px"] != [image.shape[1], image.shape[0]]:
            # Reject an image whose actual dimensions disagree with its frame record.
            raise ValueError("received video dimensions do not match its metadata")

        return metadata, image

    def close(self) -> None:
        """Release the receiver and its exclusive process-wide ZeroMQ context."""

        if not self._closed:
            self._closed = True
            # Stop the waiting thread by destroying this process's ZeroMQ context.
            # Other ZeroMQ sockets must live in another process or they close too.
            self._transport.close(kill=True)
