#!/usr/bin/env python3
"""Move the left Robotiq gripper to a target physical width in millimeters."""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "width_mm",
        type=float,
        help="Target finger opening in millimeters (0=closed, 85=fully open).",
    )
    parser.add_argument(
        "--robot-server",
        default="http://127.0.0.1:8092",
        help="RDP Franka HTTP server URL (default: http://127.0.0.1:8092).",
    )
    parser.add_argument("--max-width-mm", type=float, default=85.0)
    parser.add_argument("--velocity", type=float, default=0.05, help="Motion speed in m/s.")
    parser.add_argument("--force", type=float, default=20.0, help="Force limit in N.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--before-delay-sec",
        type=float,
        default=5.0,
        help="Record magnet data for this long before sending the command (default: 5).",
    )
    parser.add_argument(
        "--after-duration-sec",
        type=float,
        default=10.0,
        help="Record magnet data for this long after sending the command (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/eval_outputs/franka_polymetis/gripper_magnet",
        help="Directory for the normalized magnet CSV record.",
    )
    parser.add_argument(
        "--last-normalized-output",
        default=None,
        help=(
            "Optional text path for the final normalized magnet frame. "
            "Defaults to <csv-stem>_last_normalized.txt."
        ),
    )
    parser.add_argument("--magnet-port", default="/dev/ttyACM0")
    parser.add_argument("--magnet-baudrate", type=int, default=115200)
    parser.add_argument("--magnet-samples-per-frame", type=int, default=8)
    parser.add_argument(
        "--magnet-sensor-order",
        default="4,1,2,3",
        help="One-based sensor permutation, matching eval.sh (default: 4,1,2,3).",
    )
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=0.02,
        help="CSV polling interval (default: 0.02 seconds).",
    )
    parser.add_argument(
        "--magnet-startup-timeout-sec",
        type=float,
        default=5.0,
        help="Maximum wait for the first valid magnet sample.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request without moving the gripper.",
    )
    return parser.parse_args()


def command_gripper_width(
    width_mm,
    robot_server="http://127.0.0.1:8092",
    max_width_mm=85.0,
    velocity=0.05,
    force=20.0,
    timeout=10.0,
    dry_run=False,
):
    width_mm = float(width_mm)
    max_width_mm = float(max_width_mm)
    velocity = float(velocity)
    force = float(force)
    if not 0.0 <= width_mm <= max_width_mm:
        raise ValueError(
            f"width_mm must be within [0, {max_width_mm:g}], got {width_mm:g}"
        )
    if velocity <= 0.0:
        raise ValueError(f"velocity must be positive, got {velocity:g}")
    if force <= 0.0:
        raise ValueError(f"force must be positive, got {force:g}")

    url = f"{robot_server.rstrip('/')}/move_gripper/left"
    payload = {
        "width": width_mm / 1000.0,
        "velocity": velocity,
        "force_limit": force,
    }
    print(f"Robotiq target: {width_mm:.3f} mm")
    print(f"POST {url} {payload}")
    if dry_run:
        return None

    response = requests.post(url, json=payload, timeout=float(timeout))
    response.raise_for_status()
    try:
        result = response.json()
    except ValueError:
        result = response.text
    print(f"Server response: {result}")
    return result


def _csv_fieldnames(sensor_count):
    fields = ["timestamp", "elapsed", "phase", "magnet_sample_count"]
    for prefix in ("raw", "normalized"):
        for sensor_idx in range(sensor_count):
            for axis in ("x", "y", "z"):
                fields.append(f"{prefix}_s{sensor_idx + 1}_{axis}")
    return fields


def _latest_magnet_sample(reader, sensor_order):
    recent = reader.get_recent_samples()
    timestamps = recent["magnet_timestamp_ns"]
    valid = timestamps > 0
    if not valid.any():
        return None
    sample_idx = int(valid.nonzero()[0][-1])
    xyz = recent["magnet_xyz"][sample_idx][sensor_order]
    return int(timestamps[sample_idx]), xyz.astype("float32", copy=False)


def write_last_normalized_record(
    path,
    timestamp,
    elapsed,
    phase,
    sample_count,
    normalized_xyz,
):
    """Write one human-readable final normalized magnet frame."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_xyz = normalized_xyz.reshape(-1, 3)
    with output_path.open("w") as text_file:
        text_file.write("Final normalized magnet change\n")
        text_file.write(f"timestamp={float(timestamp):.9f}\n")
        text_file.write(f"elapsed={float(elapsed):.6f} s\n")
        text_file.write(f"phase={phase}\n")
        text_file.write(f"magnet_sample_count={int(sample_count)}\n")
        for sensor_idx, values in enumerate(normalized_xyz, start=1):
            text_file.write(
                f"S{sensor_idx}: x={float(values[0]):.6f} "
                f"y={float(values[1]):.6f} z={float(values[2]):.6f}\n"
            )
    return output_path


def record_gripper_width_with_magnet(
    width_mm,
    robot_server="http://127.0.0.1:8092",
    max_width_mm=85.0,
    velocity=0.05,
    force=20.0,
    timeout=10.0,
    before_delay_sec=5.0,
    after_duration_sec=10.0,
    output_dir="../data/eval_outputs/franka_polymetis/gripper_magnet",
    last_normalized_output=None,
    magnet_port="/dev/ttyACM0",
    magnet_baudrate=115200,
    magnet_samples_per_frame=8,
    magnet_sensor_order="4,1,2,3",
    sample_interval_sec=0.02,
    magnet_startup_timeout_sec=5.0,
    dry_run=False,
):
    """Record baseline-relative magnet XYZ before and after one gripper command."""
    width_mm = float(width_mm)
    before_delay_sec = float(before_delay_sec)
    after_duration_sec = float(after_duration_sec)
    sample_interval_sec = float(sample_interval_sec)
    magnet_startup_timeout_sec = float(magnet_startup_timeout_sec)
    if before_delay_sec < 0.0 or after_duration_sec < 0.0:
        raise ValueError("before_delay_sec and after_duration_sec must be non-negative")
    if sample_interval_sec <= 0.0:
        raise ValueError("sample_interval_sec must be positive")
    if magnet_startup_timeout_sec <= 0.0:
        raise ValueError("magnet_startup_timeout_sec must be positive")
    if dry_run:
        return command_gripper_width(
            width_mm,
            robot_server=robot_server,
            max_width_mm=max_width_mm,
            velocity=velocity,
            force=force,
            timeout=timeout,
            dry_run=True,
        )

    from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import (
        _MagnetometerReader,
        _parse_magnet_sensor_order,
    )

    sensor_order = _parse_magnet_sensor_order(magnet_sensor_order, used_sensor_count=4)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"gripper_{width_mm:g}mm_{stamp}_magnet.csv"
    last_normalized_path = (
        output_path.with_name(f"{output_path.stem}_last_normalized.txt")
        if last_normalized_output is None
        else Path(last_normalized_output).expanduser()
    )
    reader = _MagnetometerReader(
        port=magnet_port,
        baudrate=magnet_baudrate,
        samples_per_frame=magnet_samples_per_frame,
        used_sensor_count=4,
        subtract_baseline=False,
    )
    reader.start()
    baseline = None
    last_timestamp_ns = 0
    row_count = 0
    start = time.monotonic()
    start_wall = time.time()
    command_sent = False
    last_normalized_record = None
    fieldnames = _csv_fieldnames(sensor_order.size)
    print(f"Recording normalized magnet data to {output_path}")
    print(f"Waiting {before_delay_sec:.3f}s before commanding {width_mm:.3f} mm...")
    try:
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()
            deadline = start + before_delay_sec + after_duration_sec
            first_sample_deadline = start + magnet_startup_timeout_sec
            while time.monotonic() < deadline:
                now = time.monotonic()
                if not command_sent and now >= start + before_delay_sec:
                    command_gripper_width(
                        width_mm,
                        robot_server=robot_server,
                        max_width_mm=max_width_mm,
                        velocity=velocity,
                        force=force,
                        timeout=timeout,
                    )
                    command_sent = True
                    print("Gripper command sent; continuing magnet recording.")

                latest = _latest_magnet_sample(reader, sensor_order)
                if latest is not None and latest[0] > last_timestamp_ns:
                    timestamp_ns, raw_xyz = latest
                    if baseline is None:
                        baseline = raw_xyz.copy()
                    normalized_xyz = raw_xyz - baseline
                    timestamp = timestamp_ns / 1e9
                    row = {
                        "timestamp": timestamp,
                        "elapsed": timestamp - start_wall,
                        "phase": "before_command" if not command_sent else "after_command",
                        "magnet_sample_count": reader.sample_count,
                    }
                    for prefix, values in (("raw", raw_xyz), ("normalized", normalized_xyz)):
                        for sensor_idx, sensor_values in enumerate(values):
                            for axis, value in zip(("x", "y", "z"), sensor_values):
                                row[f"{prefix}_s{sensor_idx + 1}_{axis}"] = float(value)
                    writer.writerow(row)
                    csv_file.flush()
                    last_timestamp_ns = timestamp_ns
                    last_normalized_record = {
                        "timestamp": timestamp,
                        "elapsed": timestamp - start_wall,
                        "phase": "before_command" if not command_sent else "after_command",
                        "sample_count": reader.sample_count,
                        "normalized_xyz": normalized_xyz.copy(),
                    }
                    row_count += 1
                if baseline is None and time.monotonic() >= first_sample_deadline:
                    raise RuntimeError(
                        f"No valid magnet sample received from {magnet_port} "
                        f"within {magnet_startup_timeout_sec:.1f}s"
                    )
                time.sleep(sample_interval_sec)
    except KeyboardInterrupt:
        print("Interrupted; keeping the partial CSV record.")
    finally:
        if not command_sent:
            print("Recording ended before the delay elapsed; no gripper command was sent.")
        reader.stop()
        if last_normalized_record is not None:
            write_last_normalized_record(
                last_normalized_path,
                **last_normalized_record,
            )
            print(f"Saved final normalized magnet frame to {last_normalized_path}")
    print(f"Saved {row_count} normalized magnet samples to {output_path}")
    return output_path


def main():
    args = parse_args()
    record_gripper_width_with_magnet(
        width_mm=args.width_mm,
        robot_server=args.robot_server,
        max_width_mm=args.max_width_mm,
        velocity=args.velocity,
        force=args.force,
        timeout=args.timeout,
        before_delay_sec=args.before_delay_sec,
        after_duration_sec=args.after_duration_sec,
        output_dir=args.output_dir,
        last_normalized_output=args.last_normalized_output,
        magnet_port=args.magnet_port,
        magnet_baudrate=args.magnet_baudrate,
        magnet_samples_per_frame=args.magnet_samples_per_frame,
        magnet_sensor_order=args.magnet_sensor_order,
        sample_interval_sec=args.sample_interval_sec,
        magnet_startup_timeout_sec=args.magnet_startup_timeout_sec,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
