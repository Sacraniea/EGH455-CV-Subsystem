import argparse
import json
import os
from pathlib import Path
import signal
import sys

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Thread
from time import monotonic, sleep
from NetGear_config import create_receiver, receive_frame, close_receiver

import cv2


class LatestFrame:
    """Store the most recent frame and its arrival time"""

    def __init__(self):

        # Coordinate access to the frame and its metadata between the receiver and any waiting consumers.
        self.condition = Condition()

        # No image is available until the first frame arrives, so these are None until then.
        self.jpeg = None
        self.metadata = None
        self.received_at = None

        # Count updates so each waiting consumer can detect a new frame.
        self.generation = 0
        self.closed = False

    def publish(self, metadata, image):

        # Record when this reciever begins processing the frame.
        received_at = monotonic()

        # Encode the image, once to share with every waiting consumer.
        success, encoded = cv2.imencode(
            ".jpg", # JPEG compression format for the image
            image, # Image to encode
            [cv2.IMWRITE_JPEG_QUALITY, 90] # OpenCV parameter for JPEG quality, currently 90% of maximum quality.
        )

        # Stop if the encoding failed, which should not happen for a valid BGR image.
        if not success:
            raise RuntimeError("Failed to encode the image as JPEG")

        # Convert the encoded image to bytes for sharing with waiting consumers.
        jpeg = encoded.tobytes()
        frame_metadata = dict(metadata)

        with self.condition:

            # Stop if the receiver has been closed and no more frames will arrive.
            if self.closed:
                return

            # Replace the image an dits matching information together
            self.jpeg = jpeg
            self.metadata = frame_metadata
            self.received_at = received_at
            self.generation += 1

            # Notify any waiting consumers that a new frame is available.
            self.condition.notify_all()

    def close(self):
        # Let any waiting consumers know that no more frames will arrive.
        with self.condition:
            self.closed = True
            self.condition.notify_all()

## BROWSER REQUEST HANDLING ##

def handler_for(store):

    class StreamHandler(BaseHTTPRequestHandler):
        """Serve the most recent frame to a browser client"""

        def do_GET(self):
            try:
                if self.path not in ("/", "/stream.mjpeg"):
                    self.send_error(404, "Use /stream.mjpeg to view the camera stream")
                    return

                # Stop a connection that has stalled
                self.connection.settimeout(5.0)

                # Notify the browser that this is a multipart MJPEG stream.
                self.send_response(200)
                self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")

                # Prevent cached images and ask the browser to not buffer the stream, so it can display each frame as it arrives.
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")


                self.end_headers()

                # On start a browser has not received any frames yet, so it has no generation number.
                last_generation = 0

                while True:

                    # Keep the existing debug viewer at roughly ten frames per second.
                    sleep(0.1)

                    with store.condition:
                        # Sleep until a new frame arrived or the receiver was closed.
                        store.condition.wait_for(
                            lambda: (store.closed or store.generation != last_generation)
                        )

                        # Stop if the receiver has been closed and no more frames will arrive.
                        if store.closed:
                            return

                        # Mark the update as seen by this browser client.
                        last_generation = store.generation

                        # Avoid sending an image left over from a stalled stream.
                        if monotonic() - store.received_at > 2.0:
                            continue

                        # Take this image and its matching information for sending to the browser.
                        jpeg = store.jpeg
                        metadata = dict(store.metadata)


                    #  Describe the image and its metadata in the HTTP headers for the browser to use.
                    headers = (
                        "--frame\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n"
                        f"X-Run-ID: {metadata['run_id']}\r\n"
                        f"X-Frame-Sequence: {metadata['sequence_number']}\r\n"
                        "\r\n"
                    ).encode("ascii")

                    # Send the headers and the image to the browser.
                    self.wfile.write(headers)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            except (
                BrokenPipeError,
                ConnectionResetError,
                TimeoutError,
            ):
                # Stop if the browser closed the connection or the network timed out.
                return

        def log_message(self, *args):
            # Keep routine browser requests out of the terminal
            pass

    return StreamHandler

def interrupt(signum, frame):
    # SIGTERM and Windows CTRL_BREAK follow the same cleanup path as Ctrl+C.
    raise KeyboardInterrupt


def main(argv = None):
    parser = argparse.ArgumentParser(
        description = "Receive local NetGear frames and serve an MJPEG browser stream.",
    )
    parser.add_argument(
        "--http-host",
        default = "127.0.0.1",
        help = "This computer's IPv4 address, or 0.0.0.0 for all interfaces.",
    )
    parser.add_argument(
        "--http-port",
        type = int,
        default = 8080,
        help = "Browser stream port; defaults to 8080.",
    )
    parser.add_argument(
        "--video-host",
        default = "127.0.0.1",
        help = "NetGear listener address; defaults to loopback.",
    )
    parser.add_argument(
        "--video-port",
        type = int,
        default = 5555,
        help = "NetGear listener port; defaults to 5555.",
    )
    parser.add_argument("--ready-file", type = Path, help = argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for option in ("http_port", "video_port"):
        if not 1 <= getattr(args, option) <= 65535:
            parser.error(f"--{option.replace('_', '-')} must be between 1 and 65535")

    store = LatestFrame()
    try:
        server = ThreadingHTTPServer(
            (args.http_host, args.http_port),
            handler_for(store),
        )
    except OSError as error:
        parser.exit(
            2,
            f"Receiver error: cannot listen on {args.http_host}:{args.http_port}: "
            f"{error}\nUse an address assigned to this computer and a free port.\n",
        )

    receiver = None
    old_handlers = {}
    ready_written = False
    with server:
        server_thread = Thread(
            target = server.serve_forever,
            daemon = True,
        )
        server_thread.start()
        try:
            for name in ("SIGTERM", "SIGBREAK"):
                if hasattr(signal, name):
                    signum = getattr(signal, name)
                    old_handlers[signum] = signal.signal(signum, interrupt)

            # NetGear binds its socket before returning the receiver instance.
            receiver = create_receiver(
                host = args.video_host,
                port = args.video_port,
            )
            if args.ready_file is not None:
                ready_state = {
                    "pid": os.getpid(),
                    "http_host": args.http_host,
                    "http_port": args.http_port,
                    "video_host": args.video_host,
                    "video_port": args.video_port,
                }
                temporary = args.ready_file.with_suffix(".tmp")
                temporary.write_text(json.dumps(ready_state), encoding = "utf-8")
                temporary.replace(args.ready_file)
                ready_written = True

            if args.http_host == "0.0.0.0":
                print(f"HTTP listening on all IPv4 interfaces, port {args.http_port}.")
                print(
                    "Use this computer's LAN IP in the GCS video URL: "
                    f"http://<LAN-IP>:{args.http_port}/stream.mjpeg"
                )
            else:
                print(f"Browser stream: http://{args.http_host}:{args.http_port}/stream.mjpeg")
            print("Waiting for the sender. Press Ctrl+C to stop.", flush = True)
            while True:
                received = receive_frame(receiver)
                if received is None:
                    print("Reception ended or timed out.")
                    break
                metadata, image = received
                store.publish(metadata, image)
            return 0
        except KeyboardInterrupt:
            print("\nReceiver stopped.")
            return 0
        except Exception as error:
            print(
                f"Receiver error at video {args.video_host}:{args.video_port}: {error}",
                file = sys.stderr,
            )
            return 1
        finally:
            if ready_written:
                args.ready_file.unlink(missing_ok = True)
            store.close()
            server.shutdown()
            server_thread.join()
            if receiver is not None:
                close_receiver(receiver)
            for signum, previous in old_handlers.items():
                signal.signal(signum, previous)


if __name__ == "__main__":
    sys.exit(main())
