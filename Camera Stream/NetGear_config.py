from vidgear.gears import NetGear

def create_sender(host="127.0.0.1", port=5555):
    """Create a NetGear sender for annotated frames."""
    return NetGear(
        address=host,
        port=str(port),
        protocol="tcp",
        pattern=2,
        receive_mode=False,
        logging=False,
        multiserver_mode=True,
        jpeg_compression=False,
    )


def build_frame_record(packet, results, *, run_id, source_kind):
    """Combine this frame's identity and completed detection results."""

    # Read the original image dimensions for the frame record.
    height, width = packet.image.shape[:2]

    # Identify the capture source using the existing record format.
    if source_kind == "camera":
        source = "depthai"

    else:
        source = source_kind

    # Convert each detector's saved result into its dictionary representation.
    processor_records = []

    for result in results:
        processor_records.append(
            result.to_record()
        )

    # Keep the detections attached to the frame and run that produced them.
    return {
        "schema_version": 1,
        "producer": "camera_stream",
        "source": source,
        "source_kind": source_kind,
        "run_id": run_id,
        "event_id": f"{run_id}:{packet.sequence_number}",
        "timestamp_basis": (
            "generated_host_time"
            if source_kind == "synthetic"
            else "depthai_capture_and_host_arrival"
        ),
        "sequence_number": packet.sequence_number,
        "device_timestamp_s": packet.device_timestamp_s,
        "host_received_timestamp_s": packet.host_received_timestamp_s,
        "image_size_px": [width, height],
        "processors": processor_records,
    }


def send_frame(sender, overlay, record):
    """Send the annotated image with its matching frame record."""

    # The record was prepared from the same frame used to draw this overlay.
    sender.send(
        overlay,
        message = record,
    )

def create_receiver(host="127.0.0.1", port=5555):
    """Create a NetGear receiver for annotated frames."""
    return NetGear(
        address=host,
        port=[str(port)],
        protocol="tcp",
        pattern=2,
        receive_mode=True,
        multiserver_mode=True,
        jpeg_compression=False,
        subscriber_timeout=60,
        logging=False,
    )

def receive_frame(receiver):

    # Wait for the next frame and its ID information
    received = receiver.recv()

    if received is None:
        # Let the calling loop know that the receiver has been closed or timed out.
        return None

    # Multi-server mode returns the sending port metadata and image.
    _, metadata, image = received

    #Capture image dimensions to compare with the metadata to ensure they match
    width, height = image.shape[1], image.shape[0]

    # Check the image dimensions against the frame record to avoid misinterpretation.
    if metadata["image_size_px"] != [width, height]:
        # Reject an image whose actual dimensions disagree with its frame record.
        raise ValueError("received video dimensions do not match its metadata")

    return metadata, image

def close_receiver(receiver):
    """Close the NetGear receiver and release its socket."""
    receiver.close(kill=True)