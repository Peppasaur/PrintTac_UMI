#!/usr/bin/env python3
"""Record FR3 camera + magnet overlay without sending robot commands."""

import argparse
import os
import sys
import time

from loguru import logger

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import (
    FrankaPolymetisEnv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open the same FrankaPolymetisEnv observation path used by eval, "
            "record the magnet overlay, and never send TCP or gripper commands."
        )
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir", default="data/eval_outputs/franka_polymetis/magnet_only")
    parser.add_argument("--robot-server-host", default="127.0.0.1")
    parser.add_argument("--robot-server-port", type=int, default=8092)
    parser.add_argument("--dataset-path", default="dataset/traj_rdp10d_command_downsample2")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--camera-backend", default="realsense")
    parser.add_argument("--camera-source", default="auto")
    parser.add_argument("--camera-zed-view", default="left")
    parser.add_argument("--camera-zed-resolution", default="HD720")
    parser.add_argument("--camera-zed-depth-mode", default="NEURAL")
    parser.add_argument("--camera-flip", action="store_true")
    parser.add_argument("--magnet-port", default="/dev/ttyACM0")
    parser.add_argument("--magnet-baudrate", type=int, default=115200)
    parser.add_argument("--magnet-samples-per-frame", type=int, default=8)
    parser.add_argument("--no-magnet-reader-subtract-baseline", action="store_true")
    parser.add_argument("--magnet-normalize-to-first-frame", action="store_true")
    parser.add_argument("--no-magnet-filter-abnormal-readings", action="store_true")
    parser.add_argument("--magnet-abnormal-abs-threshold", type=float, default=5000.0)
    parser.add_argument("--recording-fps", type=float, default=12.0)
    parser.add_argument("--plot-width", type=int, default=720)
    parser.add_argument("--plot-window-sec", type=float, default=10.0)
    parser.add_argument("--recording-image-width", type=int, default=None)
    parser.add_argument("--recording-image-height", type=int, default=None)
    parser.add_argument("--http-timeout", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.warning(
        "Magnet-only recording: this script does not call env.reset(), "
        "move_to_dataset_start(), move_tcp, or move_gripper."
    )
    env = FrankaPolymetisEnv(
        transforms=None,
        robot_server_ip=args.robot_server_host,
        robot_server_port=args.robot_server_port,
        dataset_path=args.dataset_path,
        image_shape=(3, args.image_height, args.image_width),
        max_fps=args.fps,
        move_to_start_on_reset=False,
        save_processed_image=True,
        enable_policy_recording=True,
        policy_recording_output_dir=args.output_dir,
        policy_recording_fps=args.recording_fps,
        policy_recording_image_width=args.recording_image_width,
        policy_recording_image_height=args.recording_image_height,
        policy_recording_plot_width=args.plot_width,
        policy_recording_plot_window_sec=args.plot_window_sec,
        tcp_move_timeout=0.05,
        http_timeout=args.http_timeout,
        enable_magnet=True,
        magnet_required=True,
        magnet_port=args.magnet_port,
        magnet_baudrate=args.magnet_baudrate,
        magnet_samples_per_frame=args.magnet_samples_per_frame,
        magnet_tactile_key="left_gripper1_marker_offset_emb",
        magnet_tactile_dim=15,
        magnet_reader_subtract_baseline=not args.no_magnet_reader_subtract_baseline,
        magnet_normalize_to_first_frame=args.magnet_normalize_to_first_frame,
        magnet_filter_abnormal_readings=not args.no_magnet_filter_abnormal_readings,
        magnet_abnormal_abs_threshold=args.magnet_abnormal_abs_threshold,
        camera_backend=args.camera_backend,
        camera_source=args.camera_source,
        camera_zed_view=args.camera_zed_view,
        camera_zed_resolution=args.camera_zed_resolution,
        camera_zed_depth_mode=args.camera_zed_depth_mode,
        camera_flip=args.camera_flip,
        gripper_obs_mode="commanded",
        ignore_gripper_commands=True,
    )

    try:
        if hasattr(env, "prepare_policy_start"):
            env.prepare_policy_start()
        video_path = env.start_policy_recording(episode_idx=0, policy_normalizer=None)
        logger.info(f"Recording magnet-only overlay for {args.duration:.2f}s: {video_path}")
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        env.stop_policy_recording()
    finally:
        env.close()


if __name__ == "__main__":
    main()
