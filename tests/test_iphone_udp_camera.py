import time
import unittest

from reactive_diffusion_policy.real_world.iphone_udp_camera import (
    IPhoneFrameAssembly,
    IPhoneNALAssembly,
    IPhoneUDPCamera,
    VIDEO_PACKET_HEADER_V1,
    VIDEO_PACKET_HEADER_V2,
)


class IPhoneUDPCameraTests(unittest.TestCase):
    def test_reassembles_nalus_in_annexb_order(self):
        frame = IPhoneFrameAssembly(
            frame_id=7,
            capture_timestamp=time.time(),
            nalu_count=2,
            is_keyframe=True,
            created_at=time.monotonic(),
            last_update_at=time.monotonic(),
        )
        frame.nalus[0] = IPhoneNALAssembly(
            total_fragments=2,
            fragments={1: b"b", 0: b"a"},
        )
        frame.nalus[1] = IPhoneNALAssembly(
            total_fragments=1,
            fragments={0: b"c"},
        )

        self.assertTrue(frame.is_complete())
        self.assertEqual(
            frame.to_annexb(),
            b"\x00\x00\x00\x01ab\x00\x00\x00\x01c",
        )

    def test_parses_apv1_and_apv2_packets(self):
        camera = IPhoneUDPCamera()
        apv1 = VIDEO_PACKET_HEADER_V1.pack(
            b"APV1", 1, 1, 0, 12, 123.5, 0, 1, 1, 2
        ) + b"v1"
        apv2 = VIDEO_PACKET_HEADER_V2.pack(
            b"APV2",
            2,
            0,
            0,
            13,
            124.5,
            1,
            2,
            0,
            1,
            1000.0,
            1000.0,
            640.0,
            360.0,
            1280,
            720,
        ) + b"v2"

        self.assertEqual(camera._parse_video_packet(apv1)[1:], (12, 123.5, 0, 1, 1, 2, b"v1"))
        self.assertEqual(camera._parse_video_packet(apv2)[1:], (13, 124.5, 1, 2, 0, 1, b"v2"))

    def test_rejects_invalid_fragment_indices(self):
        camera = IPhoneUDPCamera()
        packet = VIDEO_PACKET_HEADER_V1.pack(
            b"APV1", 1, 0, 0, 12, 123.5, 0, 1, 2, 2
        ) + b"bad"

        self.assertIsNone(camera._parse_video_packet(packet))


if __name__ == "__main__":
    unittest.main()
