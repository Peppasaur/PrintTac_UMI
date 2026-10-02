import argparse
import os.path
from os import path as osp

import mvsdk
import time
import tqdm
import cv2
from collections import deque
import threading
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn
from loguru import logger
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
from third_party.mvcam.vcamera import vCameraSystem


class vCameraSever:
    def __init__(self, host_ip, port, camera_id=0, fps=30, buffer_size=60, video_size=(1280, 800)):
        self.camera = self.get_camera(camera_id)
        self.camera_id = camera_id

        self.host_ip = host_ip
        self.port = port

        self.fps = fps
        self.capture_interval_time = 1.0 / fps
        self.capture_buffer = deque(maxlen=buffer_size)

        self.app = FastAPI()
        self.setup_routes()

        self.video_size = video_size
        self.video_writer = None
        self.record_event = threading.Event()
        self.stop_event = threading.Event()
        self.mutex = threading.Lock()
    
    def get_camera(self, camera_id):
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) - 1 < camera_id:
            camera = None
        else:
            cam_sys = vCameraSystem()
            camera = cam_sys[camera_id]

        if camera is None:
            raise ValueError('No Mindvision camera found!')
    
        return camera

    def setup_routes(self):
        @self.app.get('/get_capture')
        async def get_capture():
            with self.mutex:
                capture_buffer_size = len(self.capture_buffer)
                if capture_buffer_size == 0:
                    capture = None
                else:
                    capture = self.capture_buffer.popleft()
            logger.info(f'Rest capture buffer size: {capture_buffer_size}')
            return Response(content=capture, media_type="application/octet-stream")

        @self.app.get('/peek_latest_capture')
        async def peek_latest_capture():
            with self.mutex:
                capture_buffer_size = len(self.capture_buffer)
                if capture_buffer_size == 0:
                    capture = None
                else:
                    capture = self.capture_buffer[-1]
            return Response(content=capture, media_type="application/octet-stream")

        @self.app.post('/start_recording/{video_path:path}')
        async def start_recording(video_path):
            if not osp.exists(osp.dirname(video_path)):
                os.makedirs(osp.dirname(video_path))
            video_path = ".".join(video_path.split('.')[:-1]) + f'_camera{self.camera_id}.mp4'
            if osp.exists(video_path):
                video_path = ".".join(video_path.split('.')[:-1]) + f'{time.strftime("_%Y%m%d_%H%M%S")}.mp4'
                logger.warning(f'Video path already exists, save to {video_path}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            with self.mutex:
                self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.video_size)
            self.record_event.set()
            logger.info(f'Start recording to {video_path}')
            return Response(content='Start recording', media_type="text/plain")

        @self.app.post('/stop_recording')
        async def stop_recording():
            self.record_event.clear()
            with self.mutex:
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
            logger.info(f'Stop recording')
            return Response(content='Stop recording', media_type="text/plain")

    def run(self):
        capture_thread = threading.Thread(target=self.capture, args=(self.stop_event,), daemon=True)
        try:
            capture_thread.start()
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            self.stop_event.set()
            capture_thread.join()

    def capture(self, stop_event):
        with self.camera as c:
            logger.info(f'Start camera capture')
            with tqdm.tqdm(range(1)) as pbar:
                start_t = time.time()
                cnt = 0
                while not stop_event.is_set():
                    capture_start_t = time.time()
                    cnt += 1
                    img = c.read()
                    img = img[:, :, ::-1]  # RGB -> BGR

                    video_img = cv2.resize(img, self.video_size)
                    with self.mutex:
                        if self.record_event.is_set():
                            self.video_writer.write(video_img)

                    img = cv2.imencode('.jpg', img)[1].tobytes()
                    with self.mutex:
                        self.capture_buffer.append(img)
                    cur_time = time.time()
                    precise_sleep(max(0., self.capture_interval_time - (cur_time - capture_start_t)))
                    pbar.set_description(f"fps={cnt / (cur_time - start_t)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_ip', type=str, default='192.168.2.187')
    parser.add_argument('--port', type=int, default=8102)
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--video_size', type=str, default='1920,1080')
    args = parser.parse_args()

    vcamera_server = vCameraSever(host_ip=args.host_ip,
                                  port=args.port,
                                  camera_id=args.camera_id,
                                  fps=args.fps,
                                  video_size=tuple(map(int, args.video_size.split(','))))
    vcamera_server.run()