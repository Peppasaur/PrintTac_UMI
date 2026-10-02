#!/usr/bin/env python3
"""Command one gripper width and record camera + magnet data.

This script does not run policy inference and does not send TCP commands. It
only opens the same real eval observation stack, sends one physical gripper
width command, then records the policy-recording magnet trace/overlay.
"""

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
        description="Close/open gripper to a specified width and record magnet readings."
    )
    parser.add_argument("--gripper-width-mm", type=float, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.5,
        help="Wait after commanding gripper before starting the recording.",
    )
    parser.add_argument("--output-dir", default="data/eval_outputs/franka_polymetis/gripper_magnet")
    parser.add_argument("--robot-server-host", default="127.0.0.1")
    parser.add_argument("--robot-server-port", type=int, default=8092)
    parser.add_argument("--dataset-path", default="dataset/traj_rdp10d_command_downsample5")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--camera-backend", default="iphone")
    parser.add_argument("--camera-source", default="auto")
    parser.add_argument("--camera-preprocess-mode", default="square_crop")
    parser.add_argument("--camera-square-crop-bottom-rows", type=int, default=0)
    parser.add_argument("--camera-zed-view", default="left")
    parser.add_argument("--camera-zed-resolution", default="HD720")
    parser.add_argument("--camera-zed-depth-mode", default="NEURAL")
    parser.add_argument("--camera-iphone-bind-host", default="0.0.0.0")
    parser.add_argument("--camera-iphone-video-port", type=int, default=5560)
    parser.add_argument("--camera-iphone-combined-port", type=int, default=5558)
    parser.add_argument("--camera-iphone-phone-ip", default="")
    parser.add_argument("--camera-iphone-registration-port", type=int, default=5559)
    parser.add_argument("--camera-iphone-startup-timeout", type=float, default=5.0)
    parser.add_argument("--camera-iphone-read-timeout", type=float, default=1.0)
    parser.add_argument("--camera-iphone-hello-interval", type=float, default=2.0)
    parser.add_argument("--camera-flip", action="store_true")
    parser.add_argument("--gripper-stroke", type=float, default=0.085)
    parser.add_argument("--grasp-force", type=float, default=20.0)
    parser.add_argument("--gripper-velocity", type=float, default=0.08)
    parser.add_argument("--magnet-port", default="/dev/ttyACM0")
    parser.add_argument("--magnet-baudrate", type=int, default=115200)
    parser.add_argument("--magnet-samples-per-frame", type=int, default=8)
    parser.add_argument("--magnet-sensor-order", default="4,1,2,3")
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
    parser.add_argument("--gripper-http-timeout", type=float, default=5.0)
    return parser.parse_args()


def parse_sensor_order(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.strip("[]").split(",")]


def main():
    args = parse_args()
    gripper_width_m = float(args.gripper_width_mm) / 1000.0
    if gripper_width_m < 0.0 or gripper_width_m > float(args.gripper_stroke):
        raise ValueError(
            "--gripper-width-mm must be within gripper stroke: "
            f"0..{float(args.gripper_stroke) * 1000.0:.3f} mm, "
            f"got {float(args.gripper_width_mm):.3f} mm"
        )

    logger.warning(
        "This script sends exactly one gripper width command and never sends "
        "TCP commands or policy actions."
    )
    env = FrankaPolymetisEnv(
        transforms=None,
        robot_server_ip=args.robot_server_host,
        robot_server_port=args.robot_server_port,
        dataset_path=args.dataset_path,
        image_shape=(3, args.image_height, args.image_width),
        max_fps=args.fps,
        gripper_stroke=args.gripper_stroke,
        grasp_force=args.grasp_force,
        gripper_velocity=args.gripper_velocity,
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
        gripper_http_timeout=args.gripper_http_timeout,
        enable_magnet=True,
        magnet_required=True,
        magnet_port=args.magnet_port,
        magnet_baudrate=args.magnet_baudrate,
        magnet_samples_per_frame=args.magnet_samples_per_frame,
        magnet_tactile_key="left_gripper1_marker_offset_emb",
        magnet_tactile_dim=15,
        magnet_sensor_order=parse_sensor_order(args.magnet_sensor_order),
        magnet_reader_subtract_baseline=not args.no_magnet_reader_subtract_baseline,
        magnet_normalize_to_first_frame=args.magnet_normalize_to_first_frame,
        magnet_filter_abnormal_readings=not args.no_magnet_filter_abnormal_readings,
        magnet_abnormal_abs_threshold=args.magnet_abnormal_abs_threshold,
        camera_backend=args.camera_backend,
        camera_source=args.camera_source,
        camera_preprocess_mode=args.camera_preprocess_mode,
        camera_square_crop_bottom_rows=args.camera_square_crop_bottom_rows,
        camera_zed_view=args.camera_zed_view,
        camera_zed_resolution=args.camera_zed_resolution,
        camera_zed_depth_mode=args.camera_zed_depth_mode,
        camera_iphone_bind_host=args.camera_iphone_bind_host,
        camera_iphone_video_port=args.camera_iphone_video_port,
        camera_iphone_combined_port=args.camera_iphone_combined_port,
        camera_iphone_phone_ip=args.camera_iphone_phone_ip,
        camera_iphone_registration_port=args.camera_iphone_registration_port,
        camera_iphone_startup_timeout=args.camera_iphone_startup_timeout,
        camera_iphone_read_timeout=args.camera_iphone_read_timeout,
        camera_iphone_hello_interval=args.camera_iphone_hello_interval,
        camera_flip=args.camera_flip,
        gripper_obs_mode="commanded",
        ignore_gripper_commands=False,
        ignore_policy_gripper_commands=True,
    )

    try:
        if hasattr(env, "prepare_policy_start"):
            env.prepare_policy_start()
        logger.info(f"Commanding gripper width={args.gripper_width_mm:.3f} mm")
        env.send_gripper_width_m_direct(gripper_width_m)
        if args.settle_sec > 0.0:
            time.sleep(float(args.settle_sec))
        video_path = env.start_policy_recording(episode_idx=0, policy_normalizer=None)
        logger.info(
            f"Recording magnet trace for {args.duration:.2f}s after gripper command: "
            f"{video_path}"
        )
        deadline = time.monotonic() + float(args.duration)
        while time.monotonic() < deadline:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        env.stop_policy_recording()
    finally:
        env.close()


if __name__ == "__main__":
    main()
