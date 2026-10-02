#!/usr/bin/env python3
"""Passively measure ARPoseStreamer iPhone UDP video latency and frame loss.

The script sends only the PC_HELLO registration packet required by the phone
gateway.  It does not import the robot environment or issue robot commands.
"""

from __future__ import annotations

import argparse
import select
import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np


VIDEO_HEADER_V1 = struct.Struct("<4sBBHIdHHHH")
VIDEO_HEADER_V2 = struct.Struct("<4sBBHIdHHHHffffHH")
POSE_PREFIX = struct.Struct("<4sHHI16sd")
VIDEO_MAGIC_TO_HEADER = {b"APV1": VIDEO_HEADER_V1, b"APV2": VIDEO_HEADER_V2}


@dataclass(frozen=True)
class VideoFragment:
    frame_id: int
    capture_timestamp: float
    nalu_index: int
    nalu_count: int
    fragment_index: int
    fragment_count: int
    payload: bytes
    is_keyframe: bool


@dataclass
class NaluAssembly:
    fragment_count: int
    fragments: dict[int, bytes] = field(default_factory=dict)

    def add(self, fragment_index: int, payload: bytes) -> None:
        if fragment_index not in self.fragments:
            self.fragments[fragment_index] = payload

    @property
    def is_complete(self) -> bool:
        return len(self.fragments) == self.fragment_count


@dataclass
class FrameAssembly:
    frame_id: int
    capture_timestamp: float
    nalu_count: int
    first_received_at: float
    last_received_at: float
    nalus: dict[int, NaluAssembly] = field(default_factory=dict)

    def add(self, fragment: VideoFragment, received_at: float) -> None:
        self.last_received_at = received_at
        nalu = self.nalus.get(fragment.nalu_index)
        if nalu is None:
            nalu = NaluAssembly(fragment.fragment_count)
            self.nalus[fragment.nalu_index] = nalu
        elif nalu.fragment_count != fragment.fragment_count:
            nalu.fragment_count = max(nalu.fragment_count, fragment.fragment_count)
        nalu.add(fragment.fragment_index, fragment.payload)

    @property
    def is_complete(self) -> bool:
        return len(self.nalus) == self.nalu_count and all(
            nalu.is_complete for nalu in self.nalus.values()
        )


def parse_video_fragment(packet: bytes) -> VideoFragment | None:
    if len(packet) < 6:
        return None
    header = VIDEO_MAGIC_TO_HEADER.get(packet[:4])
    if header is None or len(packet) < header.size:
        return None
    values = header.unpack_from(packet)
    if header is VIDEO_HEADER_V1:
        _, version, flags, _, frame_id, capture, nalu_index, nalu_count, fragment_index, fragment_count = values
        if version != 1:
            return None
    else:
        _, version, flags, _, frame_id, capture, nalu_index, nalu_count, fragment_index, fragment_count, *_ = values
        if version != 2:
            return None
    if (
        nalu_count <= 0
        or fragment_count <= 0
        or not 0 <= nalu_index < nalu_count
        or not 0 <= fragment_index < fragment_count
    ):
        return None
    return VideoFragment(
        frame_id=int(frame_id),
        capture_timestamp=float(capture),
        nalu_index=int(nalu_index),
        nalu_count=int(nalu_count),
        fragment_index=int(fragment_index),
        fragment_count=int(fragment_count),
        payload=packet[header.size :],
        is_keyframe=bool(flags & 0x01),
    )


def parse_pose_send_timestamp(packet: bytes) -> float | None:
    if len(packet) < POSE_PREFIX.size or packet[:4] != b"APM1":
        return None
    magic, version, _flags, _sequence, _session, phone_send_unix = POSE_PREFIX.unpack_from(packet)
    if magic != b"APM1" or version != 1:
        return None
    return float(phone_send_unix)


def extract_udp_payload(ethernet_frame: bytes) -> tuple[int, bytes] | None:
    """Return ``(destination_port, payload)`` from an Ethernet IPv4 UDP frame."""
    if len(ethernet_frame) < 14:
        return None
    offset = 14
    ether_type = struct.unpack_from("!H", ethernet_frame, 12)[0]
    while ether_type in (0x8100, 0x88A8):
        if len(ethernet_frame) < offset + 4:
            return None
        ether_type = struct.unpack_from("!H", ethernet_frame, offset + 2)[0]
        offset += 4
    if ether_type != 0x0800 or len(ethernet_frame) < offset + 20:
        return None
    version_and_ihl = ethernet_frame[offset]
    ip_header_size = (version_and_ihl & 0x0F) * 4
    if version_and_ihl >> 4 != 4 or ip_header_size < 20:
        return None
    if len(ethernet_frame) < offset + ip_header_size + 8:
        return None
    if ethernet_frame[offset + 9] != socket.IPPROTO_UDP:
        return None
    udp_offset = offset + ip_header_size
    _source_port, destination_port, udp_length, _checksum = struct.unpack_from(
        "!HHHH", ethernet_frame, udp_offset
    )
    if udp_length < 8 or len(ethernet_frame) < udp_offset + udp_length:
        return None
    return int(destination_port), ethernet_frame[udp_offset + 8 : udp_offset + udp_length]


def percentile_summary(values: list[float], label: str) -> str:
    if not values:
        return f"{label}=no samples"
    data = np.asarray(values, dtype=np.float64)
    return (
        f"{label}: n={len(data)}, p50={np.percentile(data, 50):.1f}ms, "
        f"p95={np.percentile(data, 95):.1f}ms, p99={np.percentile(data, 99):.1f}ms, "
        f"max={data.max():.1f}ms"
    )


def run(args: argparse.Namespace) -> int:
    capture_socket = None
    video_socket = None
    pose_socket = None
    hello_socket = None
    if args.capture_interface:
        if not hasattr(socket, "AF_PACKET"):
            raise RuntimeError("--capture-interface requires Linux AF_PACKET support")
        capture_socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
        capture_socket.bind((args.capture_interface, 0))
        capture_socket.setblocking(False)
        mode = f"passive packet capture on {args.capture_interface}"
    else:
        video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        pose_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for sock in (video_socket, pose_socket):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setblocking(False)
        try:
            video_socket.bind((args.bind_host, args.video_port))
            pose_socket.bind((args.bind_host, args.combined_port))
        except OSError as exc:
            video_socket.close()
            pose_socket.close()
            raise RuntimeError(f"Could not bind UDP measurement sockets: {exc}") from exc
        hello_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mode = "registered UDP receiver"

    hello_payload = f"PC_HELLO,1,{args.combined_port},{args.video_port}\n".encode("ascii")
    assemblies: dict[int, FrameAssembly] = {}
    complete_raw_ms: list[float] = []
    pose_raw_ms: list[float] = []
    completed_capture_times: list[float] = []
    completed = 0
    incomplete_dropped = 0
    malformed_packets = 0
    received_video_packets = 0
    duplicate_or_old_frames = 0
    missing_frame_ids = 0
    last_completed_frame_id: int | None = None
    next_hello = 0.0
    deadline = time.monotonic() + args.duration

    print(
        f"Measuring ({mode}): video_port={args.video_port}, "
        f"pose_port={args.combined_port}, duration={args.duration:.1f}s"
    )
    try:
        while time.monotonic() < deadline:
            now_monotonic = time.monotonic()
            if hello_socket is not None and args.phone_ip and now_monotonic >= next_hello:
                hello_socket.sendto(hello_payload, (args.phone_ip, args.registration_port))
                next_hello = now_monotonic + args.hello_interval

            sockets = [capture_socket] if capture_socket is not None else [video_socket, pose_socket]
            readable, _, _ = select.select(sockets, [], [], 0.05)
            for sock in readable:
                packet, _address = sock.recvfrom(65535)
                received_at = time.time()
                if sock is capture_socket:
                    extracted = extract_udp_payload(packet)
                    if extracted is None:
                        continue
                    destination_port, packet = extracted
                elif sock is pose_socket:
                    destination_port = args.combined_port
                else:
                    destination_port = args.video_port
                if destination_port == args.combined_port:
                    pose_timestamp = parse_pose_send_timestamp(packet)
                    if pose_timestamp is not None:
                        pose_raw_ms.append((received_at - pose_timestamp) * 1000.0)
                    continue
                if destination_port != args.video_port:
                    continue

                received_video_packets += 1
                fragment = parse_video_fragment(packet)
                if fragment is None:
                    malformed_packets += 1
                    continue
                if last_completed_frame_id is not None and fragment.frame_id <= last_completed_frame_id:
                    duplicate_or_old_frames += 1
                    continue
                frame = assemblies.get(fragment.frame_id)
                if frame is None:
                    frame = FrameAssembly(
                        frame_id=fragment.frame_id,
                        capture_timestamp=fragment.capture_timestamp,
                        nalu_count=fragment.nalu_count,
                        first_received_at=received_at,
                        last_received_at=received_at,
                    )
                    assemblies[fragment.frame_id] = frame
                frame.add(fragment, received_at)
                if frame.is_complete:
                    assemblies.pop(frame.frame_id, None)
                    if last_completed_frame_id is not None and frame.frame_id > last_completed_frame_id + 1:
                        missing_frame_ids += frame.frame_id - last_completed_frame_id - 1
                    last_completed_frame_id = frame.frame_id
                    completed += 1
                    completed_capture_times.append(frame.capture_timestamp)
                    complete_raw_ms.append((received_at - frame.capture_timestamp) * 1000.0)

            wall_now = time.time()
            stale_ids = [
                frame_id
                for frame_id, frame in assemblies.items()
                if wall_now - frame.last_received_at >= args.stale_frame_seconds
            ]
            for frame_id in stale_ids:
                assemblies.pop(frame_id, None)
                incomplete_dropped += 1
    finally:
        for sock in (capture_socket, video_socket, pose_socket, hello_socket):
            if sock is not None:
                sock.close()

    reference_raw_ms = min(pose_raw_ms) if pose_raw_ms else (min(complete_raw_ms) if complete_raw_ms else None)
    excess_video_ms = (
        [max(0.0, value - reference_raw_ms) for value in complete_raw_ms]
        if reference_raw_ms is not None
        else []
    )
    capture_dt_ms = np.diff(completed_capture_times) * 1000.0 if len(completed_capture_times) > 1 else []
    duration = max(args.duration, 1e-9)
    print("\nResults")
    print(f"video_packets={received_video_packets}, malformed={malformed_packets}")
    print(
        f"complete_frames={completed}, fps={completed / duration:.2f}, "
        f"missing_frame_ids={missing_frame_ids}, incomplete_dropped={incomplete_dropped}, "
        f"old_or_duplicate_packets={duplicate_or_old_frames}"
    )
    print(percentile_summary(pose_raw_ms, "pose raw capture-to-PC"))
    print(percentile_summary(complete_raw_ms, "video raw capture-to-complete-frame"))
    print(percentile_summary(excess_video_ms, "video excess above pose minimum"))
    print(percentile_summary(list(capture_dt_ms), "completed video capture interval"))
    if reference_raw_ms is not None:
        source = "pose" if pose_raw_ms else "video"
        print(f"clock/link reference={reference_raw_ms:.1f}ms ({source} minimum raw delay)")
    if not pose_raw_ms:
        print("Warning: no pose packets received; video excess latency uses its own minimum.")
    return 0 if completed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--phone-ip", default="172.20.10.1")
    parser.add_argument("--video-port", type=int, default=5560)
    parser.add_argument("--combined-port", type=int, default=5558)
    parser.add_argument("--registration-port", type=int, default=5559)
    parser.add_argument("--hello-interval", type=float, default=2.0)
    parser.add_argument("--stale-frame-seconds", type=float, default=0.20)
    parser.add_argument(
        "--capture-interface",
        help=(
            "Passively sniff this Linux network interface instead of binding UDP ports. "
            "Requires CAP_NET_RAW and does not send PC_HELLO."
        ),
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.hello_interval <= 0 or args.stale_frame_seconds <= 0:
        parser.error("duration, hello interval, and stale frame seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
