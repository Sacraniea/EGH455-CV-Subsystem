"""Exercise the launcher with real, camera-free child processes."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_subsystem

RECEIVER = '''
import argparse, json, signal, sys, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--ready-file')
parser.add_argument('--http-host')
parser.add_argument('--http-port')
parser.add_argument('--video-host')
parser.add_argument('--video-port')
args = parser.parse_args()
root = Path(__file__).parent
(root / 'receiver_args.json').write_text(json.dumps(vars(args)))
def stop(signum, frame):
    (root / 'receiver_stopped').write_text(str(signum))
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, stop)
MODE
'''
SENDER = '''
import json, signal, sys, time
from pathlib import Path
root = Path(__file__).parent
args = json.loads((root / 'receiver_args.json').read_text())
assert Path(args['ready_file']).exists(), 'sender started before receiver readiness'
(root / 'sender_args.json').write_text(json.dumps(sys.argv[1:]))
def stop(signum, frame):
    (root / 'sender_stopped').write_text(str(signum))
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, stop)
MODE
'''

class SubsystemCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.output, self.errors = StringIO(), StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(redirect_stderr(self.errors))
        self.enterContext(patch.object(run_subsystem, 'SCRIPT_DIR', self.root))
        self.enterContext(patch.object(run_subsystem, 'RUNS_DIR', self.root / 'runs'))

    def children(self, receiver=None, sender='sys.exit(0)'):
        if receiver is None:
            receiver = "Path(args.ready_file).write_text('{}')\nwhile True: time.sleep(0.05)"
        (self.root / 'run_receiver.py').write_text(RECEIVER.replace('MODE', receiver))
        (self.root / 'run_stream.py').write_text(SENDER.replace('MODE', sender))

    def test_ready_receiver_precedes_sender_and_custom_ports_are_shared(self):
        self.children()
        result = run_subsystem.main([
            '192.0.2.20', '--source', 'synthetic', '--no-targets', '--max-frames', '3',
            '--http-port', '18081', '--video-port', '15556', '--profile', '480p20',
        ])
        self.assertEqual(result, 0)
        receiver = json.loads((self.root / 'receiver_args.json').read_text())
        self.assertEqual(receiver['http_port'], '18081')
        self.assertEqual(receiver['video_port'], '15556')
        self.assertEqual(receiver['video_host'], '127.0.0.1')
        sender = json.loads((self.root / 'sender_args.json').read_text())
        self.assertEqual(sender[:9], ['192.0.2.20', '--source', 'synthetic', '--no-targets',
                                     '--max-frames', '3', '--profile', '480p20', '--receiver-host'])
        self.assertEqual(sender[sender.index('--receiver-host') + 1], '127.0.0.1')
        self.assertEqual(sender[sender.index('--video-port') + 1], '15556')
        run_root = Path(sender[sender.index('--output-dir') + 1])
        self.assertTrue(run_root.is_dir())
        self.assertTrue((run_root / 'receiver.log').exists())
        self.assertTrue((self.root / 'receiver_stopped').exists())

    def test_receiver_failure_before_ready_never_launches_sender(self):
        self.children(receiver="print('Address already in use', flush=True)\nsys.exit(7)")
        self.assertEqual(run_subsystem.main([]), 7)
        self.assertFalse((self.root / 'sender_args.json').exists())
        self.assertIn('receiver.log', self.errors.getvalue())
        log = next((self.root / 'runs').glob('*/receiver.log'))
        self.assertIn('Address already in use', log.read_text())

    def test_receiver_readiness_timeout_stops_its_child(self):
        self.children(receiver='while True: time.sleep(0.05)')
        self.assertNotEqual(run_subsystem.main(['--startup-timeout', '0.2']), 0)
        self.assertFalse((self.root / 'sender_args.json').exists())
        self.assertTrue((self.root / 'receiver_stopped').exists())
        self.assertIn('ready', self.errors.getvalue().lower())

    def test_sender_error_is_propagated_and_receiver_is_stopped(self):
        self.children(sender='sys.exit(5)')
        self.assertEqual(run_subsystem.main([]), 5)
        self.assertTrue((self.root / 'receiver_stopped').exists())

    def test_receiver_error_while_streaming_stops_sender(self):
        self.children(receiver="Path(args.ready_file).write_text('{}')\ntime.sleep(0.4)\nsys.exit(9)",
                      sender='while True: time.sleep(0.05)')
        self.assertEqual(run_subsystem.main([]), 9)
        self.assertTrue((self.root / 'sender_stopped').exists())

    def test_invalid_port_is_rejected_before_children_start(self):
        self.children()
        with self.assertRaises(SystemExit) as stopped:
            run_subsystem.main(['--video-port', '0'])
        self.assertEqual(stopped.exception.code, 2)
        self.assertFalse((self.root / 'receiver_args.json').exists())

    def test_duplicate_output_directory_is_not_reused(self):
        self.children()
        run_root = self.root / 'already-used'
        run_root.mkdir()
        (run_root / 'run.json').write_text('existing evidence')
        self.assertNotEqual(run_subsystem.main(['--output-dir', str(run_root)]), 0)
        self.assertEqual((run_root / 'run.json').read_text(), 'existing evidence')
        self.assertFalse((self.root / 'receiver_args.json').exists())

    def test_ctrl_c_stops_both_children_without_hanging(self):
        if sys.platform == 'win32':
            self.skipTest('POSIX signal integration; Windows cleanup uses CTRL_BREAK_EVENT')
        import signal
        import time
        self.children(sender='while True: time.sleep(0.05)')
        code = (
            f"import sys; sys.path.insert(0, {str(Path(run_subsystem.__file__).parent)!r}); "
            "import run_subsystem; from pathlib import Path; "
            f"run_subsystem.SCRIPT_DIR = Path({str(self.root)!r}); "
            f"run_subsystem.RUNS_DIR = Path({str(self.root / 'runs')!r}); "
            "sys.exit(run_subsystem.main([]))"
        )
        child = subprocess.Popen([sys.executable, '-B', '-c', code], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / 'sender_args.json').exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    self.fail('Launcher never started sender')
                time.sleep(0.03)
            child.send_signal(signal.SIGINT)
            child.communicate(timeout=7)
            self.assertEqual(child.returncode, 130)
            self.assertTrue((self.root / 'sender_stopped').exists())
            self.assertTrue((self.root / 'receiver_stopped').exists())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


    def test_stop_child_escalates_for_a_process_that_ignores_stop_signals(self):
        if sys.platform == 'win32':
            self.skipTest('POSIX ignored-signal escalation')
        import signal
        import time
        child_ready = self.root / 'unresponsive.ready'
        code = (
            'import signal, time; from pathlib import Path; '
            'signal.signal(signal.SIGINT, signal.SIG_IGN); '
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            f"Path({str(child_ready)!r}).write_text('ready'); "
            'time.sleep(30)'
        )
        child = subprocess.Popen([sys.executable, '-B', '-c', code],
                                 **run_subsystem.child_options())
        try:
            deadline = time.monotonic() + 5
            while not child_ready.exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    self.fail('Unresponsive child did not start')
                time.sleep(0.03)
            run_subsystem.stop_child(child)
            self.assertEqual(child.returncode, -signal.SIGKILL)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)



    def test_open_browser_waits_for_ready_and_uses_local_wildcard_url(self):
        self.children()
        def open_ready_url(url):
            receiver = json.loads((self.root / 'receiver_args.json').read_text())
            self.assertTrue(Path(receiver['ready_file']).is_file())
            self.assertEqual(url, 'http://127.0.0.1:18080/stream.mjpeg')
            return True
        with patch('webbrowser.open', side_effect = open_ready_url) as opened:
            result = run_subsystem.main([
                '--open-browser', '--http-host', '0.0.0.0', '--http-port', '18080',
            ])
        self.assertEqual(result, 0)
        opened.assert_called_once()
        sender = json.loads((self.root / 'sender_args.json').read_text())
        self.assertNotIn('--open-browser', sender)

    def test_receiver_failure_never_opens_browser(self):
        self.children(receiver='sys.exit(7)')
        with patch('webbrowser.open') as opened:
            self.assertEqual(run_subsystem.main(['--open-browser']), 7)
        opened.assert_not_called()

    def test_browser_failure_does_not_stop_the_stream(self):
        self.children()
        with patch('webbrowser.open', side_effect = OSError('No desktop session')) as opened:
            result = run_subsystem.main(['--open-browser'])
        self.assertEqual(result, 0)
        opened.assert_called_once()
        self.assertTrue((self.root / 'sender_args.json').exists())
        self.assertIn('http://127.0.0.1:8080/stream.mjpeg', self.errors.getvalue())

    def test_default_listener_ports_match_the_subsystem_commands(self):
        self.children()
        self.assertEqual(run_subsystem.main([]), 0)
        receiver = json.loads((self.root / 'receiver_args.json').read_text())
        sender = json.loads((self.root / 'sender_args.json').read_text())
        self.assertEqual(receiver['http_port'], '8080')
        self.assertEqual(receiver['video_port'], '5555')
        self.assertEqual(sender[sender.index('--video-port') + 1], '5555')


if __name__ == '__main__':
    unittest.main()
