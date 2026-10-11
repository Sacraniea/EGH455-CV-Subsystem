"""Portable shutdown tests; fake signals and transports never contact a broker."""

import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import run_local_broker
import run_video_receiver


class FakeSignals:
    """Model Windows signal delivery without changing the host's handlers."""

    SIGINT = 2
    SIGTERM = 15
    SIGBREAK = 21
    SIG_IGN = object()

    @staticmethod
    def default_int_handler(signum, frame):
        raise KeyboardInterrupt

    def __init__(self):
        self.handlers = {
            sig: self.default_int_handler
            for sig in (self.SIGINT, self.SIGTERM, self.SIGBREAK)
        }
        self.original = self.handlers.copy()

    def signal(self, signum, handler):
        previous = self.handlers[signum]
        self.handlers[signum] = handler
        return previous

    def emit(self, signum):
        handler = self.handlers[signum]
        if handler is not self.SIG_IGN:
            handler(signum, None)


class BrokerShutdownSignalsTests(unittest.TestCase):
    def test_ctrl_break_stops_broker_and_restores_handlers_without_unix_loop_api(self):
        signals = FakeSignals()
        broker = SimpleNamespace(start=AsyncMock(), shutdown=AsyncMock())

        async def started():
            signals.emit(signals.SIGBREAK)

        broker.start.side_effect = started

        async def run(ready):
            # Windows event loops cannot install asyncio signal handlers.
            loop = asyncio.get_running_loop()
            with patch.object(
                loop,
                "add_signal_handler",
                side_effect=AssertionError("Unix-only signal API was used"),
            ):
                await asyncio.wait_for(run_local_broker.serve(18883, ready), 1.0)

        with TemporaryDirectory() as directory:
            ready = Path(directory) / "broker.ready.json"
            with patch.object(run_local_broker, "signal", signals), patch.dict(
                "sys.modules", {"amqtt.broker": SimpleNamespace(Broker=lambda config: broker)}
            ), redirect_stdout(StringIO()):
                asyncio.run(run(ready))

            self.assertEqual(json.loads(ready.read_text())["port"], 18883)

        broker.shutdown.assert_awaited_once()
        self.assertEqual(signals.handlers, signals.original)

    def test_broker_restores_handlers_when_startup_fails(self):
        signals = FakeSignals()
        broker = SimpleNamespace(
            start=AsyncMock(side_effect=RuntimeError("startup failed")),
            shutdown=AsyncMock(),
        )

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(
                loop,
                "add_signal_handler",
                side_effect=AssertionError("Unix-only signal API was used"),
            ):
                await run_local_broker.serve(18883, None)

        with patch.object(run_local_broker, "signal", signals), patch.dict(
            "sys.modules", {"amqtt.broker": SimpleNamespace(Broker=lambda config: broker)}
        ), self.assertRaisesRegex(RuntimeError, "startup failed"):
            asyncio.run(run())

        broker.shutdown.assert_not_awaited()
        self.assertEqual(signals.handlers, signals.original)

    def test_broker_restores_handlers_when_shutdown_fails(self):
        signals = FakeSignals()
        broker = SimpleNamespace(
            start=AsyncMock(),
            shutdown=AsyncMock(side_effect=OSError("shutdown failed")),
        )

        async def started():
            signals.emit(signals.SIGBREAK)

        broker.start.side_effect = started

        async def run():
            loop = asyncio.get_running_loop()
            with patch.object(
                loop,
                "add_signal_handler",
                side_effect=AssertionError("Unix-only signal API was used"),
            ):
                await asyncio.wait_for(run_local_broker.serve(18883, None), 1.0)

        with patch.object(run_local_broker, "signal", signals), patch.dict(
            "sys.modules", {"amqtt.broker": SimpleNamespace(Broker=lambda config: broker)}
        ), redirect_stdout(StringIO()), self.assertRaisesRegex(OSError, "shutdown failed"):
            asyncio.run(run())

        broker.shutdown.assert_awaited_once()
        self.assertEqual(signals.handlers, signals.original)


class ReceiverShutdownSignalsTests(unittest.TestCase):
    def test_ctrl_break_during_cleanup_preserves_evidence_and_restores_handlers(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                self.run_receiver(interrupted=interrupted)

    def test_cleanup_failure_still_restores_ctrl_break_handler_and_saves_status(self):
        self.run_receiver(cleanup_error=OSError("close failed"))

    def run_receiver(self, *, interrupted=False, cleanup_error=None):
        signals = FakeSignals()
        source = Mock(spec=["receive", "close"])
        source.receive.side_effect = [
            ({"run_id": "signal-test", "sequence_number": 1}, np.zeros((8, 8, 3), np.uint8)),
            KeyboardInterrupt() if interrupted else None,
        ]
        source.close.side_effect = cleanup_error
        server = Mock(spec=["serve_forever", "shutdown", "server_close", "daemon_threads"])
        # A supervisor interrupt can arrive after the receiver already began cleanup.
        server.shutdown.side_effect = lambda: signals.emit(signals.SIGBREAK)
        thread = Mock(spec=["start", "is_alive"])
        thread.is_alive.return_value = True

        with TemporaryDirectory() as directory:
            output = Path(directory) / "receiver"
            with patch.object(run_video_receiver, "signal", signals), patch.object(
                run_video_receiver, "ThreadingHTTPServer", return_value=server
            ), patch.object(
                run_video_receiver,
                "threading",
                SimpleNamespace(Thread=lambda **kwargs: thread, Condition=threading.Condition),
            ), patch.dict(
                "sys.modules", {"stream_video": SimpleNamespace(NetGearVideoReceiver=lambda **kwargs: source)}
            ), redirect_stdout(StringIO()):
                try:
                    if cleanup_error is not None:
                        with self.assertRaisesRegex(OSError, "close failed"):
                            run_video_receiver.main(["--output-dir", str(output)])
                    else:
                        code = run_video_receiver.main(["--output-dir", str(output)])
                        self.assertEqual(code, 130 if interrupted else 0)
                except KeyboardInterrupt:
                    self.fail("Ctrl+Break interrupted cleanup before receiver evidence was saved")

            status = json.loads((output / "status.json").read_text())
            expected = "failed" if cleanup_error is not None else "interrupted" if interrupted else "ended"
            self.assertEqual(status["status"], expected)
            self.assertEqual(status["frames_received"], 1)
            self.assertTrue((output / "latest.jpg").read_bytes().startswith(b"\xff\xd8"))

        source.close.assert_called_once()
        server.server_close.assert_called_once()
        self.assertEqual(signals.handlers, signals.original)
        with self.assertRaises(KeyboardInterrupt):
            signals.emit(signals.SIGBREAK)


if __name__ == "__main__":
    unittest.main()
