import unittest
from unittest import mock

import numpy as np

from scripts import gello_polymetis_tcp_teleop as teleop


class FakeIPhoneCamera:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.frame[0, 0] = [10, 20, 30]
        self.__class__.instances.append(self)

    def open(self):
        return self.frame.copy()

    def read(self):
        return self.frame.copy()

    def close(self):
        self.closed = True


class GelloIPhoneRecordingCameraTests(unittest.TestCase):
    def setUp(self):
        FakeIPhoneCamera.instances.clear()

    def test_iphone_backend_uses_default_phone_and_returns_rgb(self):
        with mock.patch.object(teleop, "IPhoneUDPCamera", FakeIPhoneCamera):
            camera = teleop.RecordingCamera(
                backend="iphone",
                source="auto",
                require_color=False,
                iphone_video_port=6000,
                iphone_combined_port=6001,
                iphone_registration_port=6002,
            )
            camera.open()
            rgb = camera.read_rgb()
            instance = FakeIPhoneCamera.instances[0]
            camera.close()

        self.assertEqual(instance.kwargs["phone_ip"], "172.20.10.1")
        self.assertEqual(instance.kwargs["video_port"], 6000)
        self.assertEqual(instance.kwargs["combined_port"], 6001)
        self.assertEqual(instance.kwargs["registration_port"], 6002)
        np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
        self.assertTrue(instance.closed)


if __name__ == "__main__":
    unittest.main()
