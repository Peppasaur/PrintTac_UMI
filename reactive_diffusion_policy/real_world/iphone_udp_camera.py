"""ARPoseStreamer APV1/APV2 UDP H.264 camera receiver."""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from select import select
from typing import Dict

import cv2
import numpy as np

try:
    import av
except Exception as exc:  # pragma: no cover - depends on the runtime environment
    av = None
    AV_IMPORT_ERROR = exc
else:
    AV_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

VIDEO_PACKET_HEADER_V1 = struct.Struct("<4sBBHIdHHHH")
VIDEO_PACKET_HEADER_V2 = struct.Struct("<4sBBHIdHHHHffffHH")
VIDEO_MAGIC_V1 = b"APV1"
VIDEO_MAGIC_V2 = b"APV2"
VIDEO_VERSION_V1 = 1
VIDEO_VERSION_V2 = 2
FRAME_STALE_SECONDS = 0.20
MAX_INFLIGHT_FRAMES = 8


def bgr_color_score(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return 0.0
    frame_f = frame.astype(np.float32)
    return float(
        max(
            np.mean(np.abs(frame_f[..., 0] - frame_f[..., 1])),
            np.mean(np.abs(frame_f[..., 1] - frame_f[..., 2])),
            np.mean(np.abs(frame_f[..., 0] - frame_f[..., 2])),
        )
    )


@dataclass
class IPhoneNALAssembly:
    total_fragments: int
    fragments: Dict[int, bytes] = field(default_factory=dict)

    def is_complete(self):
        return len(self.fragments) == self.total_fragments


@dataclass
class IPhoneFrameAssembly:
    frame_id: int
    capture_timestamp: float
    nalu_count: int
    is_keyframe: bool
    created_at: float
    last_update_at: float
    nalus: Dict[int, IPhoneNALAssembly] = field(default_factory=dict)

    def is_complete(self):
        if len(self.nalus) != self.nalu_count:
            return False
        return all(nalu.is_complete() for nalu in self.nalus.values())

    def to_annexb(self):
        chunks = []
        for nalu_index in range(self.nalu_count):
            assembly = self.nalus.get(nalu_index)
            if assembly is None or not assembly.is_complete():
                raise ValueError(f"iPhone frame {self.frame_id} is incomplete")
            payload = b"".join(
                assembly.fragments[index]
                for index in range(assembly.total_fragments)
            )
            chunks.append(b"\x00\x00\x00\x01" + payload)
        return b"".join(chunks)


class IPhoneUDPCamera:
    """Receive ARPoseStreamer UDP video and expose the latest BGR frame."""

    def __init__(
        self,
        source="auto",
        bind_host="0.0.0.0",
        video_port=5560,
        combined_port=5558,
        phone_ip="",
        registration_port=5559,
        startup_timeout=5.0,
        read_timeout=1.0,
        hello_interval=2.0,
        require_color=True,
        color_threshold=1.5,
    ):
        self.source = str(source)
        self.bind_host = str(bind_host)
        self.video_port = int(video_port)
        self.combined_port = int(combined_port)
        self.phone_ip = str(phone_ip or "")
        if not self.phone_ip and self.source not in ("", "auto", "none", "None"):
            self.phone_ip = self.source
        self.registration_port = int(registration_port)
        self.startup_timeout = float(startup_timeout)
        self.read_timeout = float(read_timeout)
        self.hello_interval = float(hello_interval)
        self.require_color = bool(require_color)
        self.color_threshold = float(color_threshold)

        self.video_socket = None
        self.hello_socket = None
        self.thread = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_id = None
        self.latest_frame_time = None
        self.receiver_error = None
        self.decoder = None
        self.frames = {}
        self.waiting_for_keyframe = True
        self.decoded_frames = 0
        self.dropped_frames = 0
        self.decode_errors = 0
        self.received_packets = 0
        self.unsupported_packets = 0

    @staticmethod
    def _create_decoder():
        if av is None:
            return None
        return av.CodecContext.create("h264", "r")

    def open(self):
        if av is None:
            raise RuntimeError(
                "The iPhone camera backend requires PyAV to decode the "
                "ARPoseStreamer H.264 stream. For /usr/bin/python3 install it "
                "with `sudo apt-get install python3-av`. "
                f"PyAV import failed: {AV_IMPORT_ERROR}"
            )

        self.close()
        self.stop_event.clear()
        self.receiver_error = None
        self.latest_frame = None
        self.latest_frame_id = None
        self.frames = {}
        self.waiting_for_keyframe = True
        self.received_packets = 0
        self.unsupported_packets = 0
        self.decoded_frames = 0
        self.dropped_frames = 0
        self.decode_errors = 0
        self.decoder = self._create_decoder()

        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.video_socket.bind((self.bind_host, self.video_port))
        except OSError as exc:
            self.video_socket.close()
            self.video_socket = None
            raise RuntimeError(
                "Could not bind the iPhone video UDP port "
                f"{self.bind_host}:{self.video_port}. Another iPhone receiver "
                "may already be running."
            ) from exc
        self.video_socket.setblocking(False)

        if self.phone_ip:
            self.hello_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.hello_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.hello_socket.bind((self.bind_host, self.combined_port))
            except OSError as exc:
                logger.warning(
                    "Could not bind iPhone combined socket on %s:%d (%s); "
                    "using an ephemeral PC_HELLO source port",
                    self.bind_host,
                    self.combined_port,
                    exc,
                )
                self.hello_socket.bind((self.bind_host, 0))
        else:
            logger.warning(
                "iPhone phone IP is empty; passively listening on %s:%d "
                "without PC_HELLO registration",
                self.bind_host,
                self.video_port,
            )

        self.thread = threading.Thread(
            target=self._receive_loop,
            name="iphone-udp-camera",
            daemon=True,
        )
        self.thread.start()
        self._send_hello()

        deadline = time.monotonic() + self.startup_timeout
        with self.condition:
            while (
                self.latest_frame is None
                and self.receiver_error is None
                and time.monotonic() < deadline
            ):
                self.condition.wait(timeout=0.05)
            if self.receiver_error is not None:
                error = self.receiver_error
                self.close()
                raise RuntimeError(f"iPhone camera receiver failed: {error}")
            if self.latest_frame is None:
                stats = self.stats_text()
                self.close()
                raise RuntimeError(
                    "Timed out waiting for iPhone video on "
                    f"{self.bind_host}:{self.video_port}; {stats}"
                )
            first_frame = self.latest_frame.copy()

        color_score = bgr_color_score(first_frame)
        if self.require_color and color_score < self.color_threshold:
            self.close()
            raise RuntimeError(
                f"iPhone frame color_score={color_score:.3f} is below "
                f"threshold={self.color_threshold:.3f}"
            )
        return first_frame

    def read(self):
        deadline = time.monotonic() + self.read_timeout
        with self.condition:
            while self.latest_frame is None and self.receiver_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))
            if self.latest_frame is None:
                if self.receiver_error is not None:
                    raise RuntimeError(
                        f"iPhone camera receiver failed: {self.receiver_error}"
                    )
                raise RuntimeError(
                    f"Timed out reading iPhone frame after {self.read_timeout:.3f}s"
                )
            return self.latest_frame.copy()

    def close(self):
        self.stop_event.set()
        for current_socket in (self.video_socket, self.hello_socket):
            if current_socket is not None:
                try:
                    current_socket.close()
                except OSError:
                    pass
        self.video_socket = None
        self.hello_socket = None
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
        self.decoder = None

    def stats_text(self):
        return (
            f"received_packets={self.received_packets}, "
            f"decoded_frames={self.decoded_frames}, "
            f"dropped_frames={self.dropped_frames}, "
            f"decode_errors={self.decode_errors}, "
            f"unsupported_packets={self.unsupported_packets}"
        )

    def _receive_loop(self):
        next_hello = 0.0
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.phone_ip and now >= next_hello:
                    self._send_hello()
                    next_hello = now + self.hello_interval
                try:
                    readable, _, _ = select([self.video_socket], [], [], 0.05)
                except (OSError, TypeError, ValueError):
                    break
                for current_socket in readable:
                    try:
                        packet, _address = current_socket.recvfrom(65535)
                    except OSError:
                        continue
                    self._handle_video_packet(packet)
                self._prune_stale_frames(time.monotonic())
        except Exception as exc:
            with self.condition:
                self.receiver_error = exc
                self.condition.notify_all()

    def _send_hello(self):
        if not self.phone_ip or self.hello_socket is None:
            return
        hello = f"PC_HELLO,1,{self.combined_port},{self.video_port}\n".encode(
            "ascii"
        )
        try:
            self.hello_socket.sendto(
                hello,
                (self.phone_ip, self.registration_port),
            )
        except OSError as exc:
            logger.warning("Failed to send iPhone PC_HELLO: %s", exc)

    def _parse_video_packet(self, packet):
        if len(packet) < 6:
            return None
        magic = packet[:4]
        version = packet[4]
        if magic == VIDEO_MAGIC_V1 and version == VIDEO_VERSION_V1:
            header = VIDEO_PACKET_HEADER_V1
        elif magic == VIDEO_MAGIC_V2 and version == VIDEO_VERSION_V2:
            header = VIDEO_PACKET_HEADER_V2
        else:
            self.unsupported_packets += 1
            return None
        if len(packet) < header.size:
            return None
        try:
            values = header.unpack_from(packet)
        except struct.error:
            return None

        if header is VIDEO_PACKET_HEADER_V1:
            (
                _magic,
                _version,
                flags,
                _reserved,
                frame_id,
                capture_timestamp,
                nalu_index,
                nalu_count,
                fragment_index,
                fragment_count,
            ) = values
        else:
            (
                _magic,
                _version,
                flags,
                _reserved,
                frame_id,
                capture_timestamp,
                nalu_index,
                nalu_count,
                fragment_index,
                fragment_count,
                _fx,
                _fy,
                _cx,
                _cy,
                _image_width,
                _image_height,
            ) = values
        if fragment_count <= 0 or nalu_count <= 0:
            return None
        if not (0 <= nalu_index < nalu_count):
            return None
        if not (0 <= fragment_index < fragment_count):
            return None
        return (
            flags,
            frame_id,
            capture_timestamp,
            nalu_index,
            nalu_count,
            fragment_index,
            fragment_count,
            packet[header.size :],
        )

    def _handle_video_packet(self, packet):
        self.received_packets += 1
        parsed = self._parse_video_packet(packet)
        if parsed is None:
            return
        (
            flags,
            frame_id,
            capture_timestamp,
            nalu_index,
            nalu_count,
            fragment_index,
            fragment_count,
            payload,
        ) = parsed
        if self.latest_frame_id is not None and frame_id <= self.latest_frame_id:
            return

        now = time.monotonic()
        frame = self.frames.get(frame_id)
        if frame is None:
            frame = IPhoneFrameAssembly(
                frame_id=frame_id,
                capture_timestamp=capture_timestamp,
                nalu_count=nalu_count,
                is_keyframe=bool(flags & 0x01),
                created_at=now,
                last_update_at=now,
            )
            self.frames[frame_id] = frame
        else:
            frame.last_update_at = now

        nalu = frame.nalus.get(nalu_index)
        if nalu is None:
            nalu = IPhoneNALAssembly(total_fragments=fragment_count)
            frame.nalus[nalu_index] = nalu
        elif nalu.total_fragments != fragment_count:
            nalu.total_fragments = max(nalu.total_fragments, fragment_count)
        if fragment_index not in nalu.fragments:
            nalu.fragments[fragment_index] = payload

        if frame.is_complete():
            self.frames.pop(frame_id, None)
            self._decode_frame(frame)
        self._trim_inflight_frames()

    def _decode_frame(self, frame):
        if self.decoder is None:
            return
        if self.waiting_for_keyframe and not frame.is_keyframe:
            return
        try:
            annexb = frame.to_annexb()
            if frame.is_keyframe:
                self.decoder = self._create_decoder()
                self.waiting_for_keyframe = False
            decoded_frames = self._decode_annexb_packet(annexb)
            decoded_any = False
            for decoded_frame in decoded_frames:
                decoded_any = True
                rgb = decoded_frame.to_ndarray(format="rgb24")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with self.condition:
                    self.latest_frame = bgr
                    self.latest_frame_id = frame.frame_id
                    self.latest_frame_time = time.time()
                    self.decoded_frames += 1
                    self.condition.notify_all()
            self.waiting_for_keyframe = not decoded_any and frame.is_keyframe
        except Exception as exc:
            self.decode_errors += 1
            self.waiting_for_keyframe = True
            self.decoder = self._create_decoder()
            logger.warning(
                "iPhone H.264 decode error on frame %d: %s",
                frame.frame_id,
                exc,
            )

    def _decode_annexb_packet(self, annexb):
        packet = av.Packet(annexb)
        decoded_frames = self.decoder.decode(packet)
        if decoded_frames:
            return decoded_frames
        fallback_frames = []
        for parsed_packet in self.decoder.parse(annexb):
            fallback_frames.extend(self.decoder.decode(parsed_packet))
        return fallback_frames

    def _trim_inflight_frames(self):
        if len(self.frames) <= MAX_INFLIGHT_FRAMES:
            return
        stale = sorted(
            self.frames.items(),
            key=lambda item: item[1].created_at,
        )[:-MAX_INFLIGHT_FRAMES]
        for frame_id, _frame in stale:
            self.frames.pop(frame_id, None)
            self.dropped_frames += 1

    def _prune_stale_frames(self, now):
        stale_ids = [
            frame_id
            for frame_id, frame in self.frames.items()
            if now - frame.last_update_at >= FRAME_STALE_SECONDS
        ]
        for frame_id in stale_ids:
            self.frames.pop(frame_id, None)
            self.dropped_frames += 1
