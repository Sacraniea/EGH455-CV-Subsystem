"""Foreground MQTT broker for local experiments; never listens on the LAN."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import signal


def broker_config(port):
    # Validate the port and return an anonymous broker configuration for this computer.
    if (
        type(port) is not int
        or not 1 <= port <= 65535
    ):
        raise ValueError("Port must be between 1 and 65535")

    return {
        "listeners": {
            "default": {
                "type": "tcp",
                # Bind the MQTT listener to loopback so LAN devices cannot connect here.
                "bind": f"127.0.0.1:{port}",
                "max_connections": 16,
            }
        },
        "plugins": {
            "amqtt.plugins.authentication.AnonymousAuthPlugin": {
                "allow_anonymous": True
            }
        },
    }


async def serve(port, ready_file):
    # Start the local broker, signal readiness and wait for a shutdown request.
    from amqtt.broker import Broker

    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Wake the event loop for Ctrl+C, termination and Windows Ctrl+Break.
    # signal.signal works on Windows, where loop.add_signal_handler is unavailable.
    def request_stop(signum, frame):
        loop.call_soon_threadsafe(stopped.set)

    stop_signals = [signal.SIGINT, signal.SIGTERM]

    if hasattr(signal, "SIGBREAK"):
        stop_signals.append(signal.SIGBREAK)

    previous_handlers = {}

    try:
        for sig in stop_signals:
            previous_handlers[sig] = signal.signal(sig, request_stop)

        broker = Broker(broker_config(port))
        await broker.start()

        try:
            if ready_file:
                # Write readiness only after the broker opens its listening port.
                with ready_file.open("x", encoding="utf-8") as file:
                    json.dump({"broker": "amqtt", "host": "127.0.0.1", "port": port}, file)

            print(
                f"Local MQTT broker listening on 127.0.0.1:{port}; Ctrl+C stops it.",
                flush=True,
            )
            await stopped.wait()

        finally:
            # Shut down the broker and release its listening port when the wait ends.
            await broker.shutdown()

    finally:
        # Do not leave handlers pointing at a loop that is about to close.
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    # Read the broker options and run its asynchronous startup/shutdown loop.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18883)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    broker_config(args.port)
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(serve(args.port, args.ready_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
