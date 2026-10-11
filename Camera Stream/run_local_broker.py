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

    # Turn Ctrl+C and termination signals into the event that ends the wait below.
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)

    broker = Broker(broker_config(port))
    await broker.start()

    try:
        if ready_file:
            # Write the readiness file after the broker has opened its listening port.
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
