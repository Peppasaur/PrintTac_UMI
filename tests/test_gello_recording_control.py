import json
import threading
import unittest
import urllib.request
from pathlib import Path

from scripts.gello_polymetis_tcp_teleop import TeleopRecordingControlServer


class FakeRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_recording = False
        self.episode_dir = None
        self.episode_count = 0
        self.frame_idx = 0
        self.discard_count = 0

    def start_episode(self):
        with self.lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.episode_count += 1
            self.episode_dir = Path("/tmp/fake-episode")

    def stop_episode(self):
        with self.lock:
            self.is_recording = False
            self.episode_dir = None

    def discard_episode(self):
        with self.lock:
            self.is_recording = False
            self.episode_dir = None
            self.discard_count += 1


class TeleopRecordingControlServerTests(unittest.TestCase):
    def setUp(self):
        self.recorder = FakeRecorder()
        self.server = TeleopRecordingControlServer(self.recorder, "127.0.0.1", 0)
        self.server.start()
        self.base_url = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.stop()

    def _request(self, path, method="GET"):
        request = urllib.request.Request(f"{self.base_url}{path}", method=method)
        with urllib.request.urlopen(request, timeout=1.0) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def test_start_and_stop_match_keyboard_recorder_actions(self):
        self.assertEqual(self._request("/status")["state"], "idle")

        started = self._request("/start", method="POST")
        self.assertEqual(started["action"], "start")
        self.assertEqual(started["state"], "recording")
        self.assertEqual(started["episode_count"], 1)

        stopped = self._request("/stop", method="POST")
        self.assertEqual(stopped["action"], "stop")
        self.assertEqual(stopped["state"], "idle")

    def test_discard_stops_and_deletes_the_active_episode(self):
        self._request("/start", method="POST")

        discarded = self._request("/discard", method="POST")

        self.assertEqual(discarded["action"], "discard")
        self.assertEqual(discarded["state"], "idle")
        self.assertEqual(self.recorder.discard_count, 1)


if __name__ == "__main__":
    unittest.main()
