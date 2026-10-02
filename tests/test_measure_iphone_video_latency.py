import struct

from scripts.measure_iphone_video_latency import (
    FrameAssembly,
    VIDEO_HEADER_V1,
    extract_udp_payload,
    parse_pose_send_timestamp,
    parse_video_fragment,
)


def test_parses_apv1_fragment_and_assembles_complete_frame():
    packet_a = VIDEO_HEADER_V1.pack(
        b"APV1", 1, 1, 0, 7, 1_785_000_000.25, 0, 1, 0, 2
    ) + b"first"
    packet_b = VIDEO_HEADER_V1.pack(
        b"APV1", 1, 1, 0, 7, 1_785_000_000.25, 0, 1, 1, 2
    ) + b"second"
    first = parse_video_fragment(packet_a)
    second = parse_video_fragment(packet_b)

    assert first is not None and second is not None
    frame = FrameAssembly(7, first.capture_timestamp, first.nalu_count, 10.0, 10.0)
    frame.add(first, 10.0)
    assert not frame.is_complete
    frame.add(second, 10.01)
    assert frame.is_complete


def test_parses_phone_send_timestamp_from_apm1_prefix():
    packet = struct.pack("<4sHHI16sd", b"APM1", 1, 0, 12, bytes(16), 1_785_000_001.5)

    assert parse_pose_send_timestamp(packet) == 1_785_000_001.5


def test_extracts_udp_payload_from_ethernet_ipv4_packet():
    payload = b"APV1-test"
    udp = struct.pack("!HHHH", 50000, 5560, 8 + len(payload), 0) + payload
    ipv4 = bytes([0x45, 0]) + struct.pack("!H", 20 + len(udp)) + bytes(4) + bytes([64, 17]) + bytes(10)
    ethernet = bytes(12) + struct.pack("!H", 0x0800)

    assert extract_udp_payload(ethernet + ipv4 + udp) == (5560, payload)
