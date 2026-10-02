#!/usr/bin/env python3
"""Open and close a Robotiq gripper directly over its serial connection."""

import argparse
import math
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Control only a Robotiq gripper over serial. No robot-arm or "
            "Polymetis/HTTP connection is created."
        )
    )
    parser.add_argument(
        "--port",
        default="auto",
        help="Serial port such as /dev/ttyUSB1, or auto (default: auto).",
    )
    parser.add_argument("--device-id", type=int, default=9, help="Modbus device ID.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of close/open cycles.")
    parser.add_argument(
        "--open-width-mm",
        type=float,
        default=None,
        help="Target opening in millimeters. Defaults to fully open.",
    )
    parser.add_argument(
        "--max-width-mm",
        type=float,
        default=85.0,
        help="Physical fully-open width used for millimeter calibration (default: 85).",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=2.0,
        help="Wait after closing and after opening (default: 2 seconds).",
    )
    parser.add_argument(
        "--start-delay-sec",
        type=float,
        default=3.0,
        help="Safety delay before the first close command (default: 3 seconds).",
    )
    parser.add_argument(
        "--reset-wait-sec",
        type=float,
        default=1.0,
        help="Wait between reset and activation (default: 1 second).",
    )
    parser.add_argument("--speed", type=int, default=255, help="Motion speed in [0, 255].")
    parser.add_argument("--force", type=int, default=100, help="Closing force in [0, 255].")
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Activate without resetting first; use only when the gripper is already initialized.",
    )
    return parser.parse_args()


def _validate_settings(
    cycles,
    hold_sec,
    start_delay_sec,
    reset_wait_sec,
    speed,
    force,
    open_width_mm,
    max_width_mm,
):
    if int(cycles) <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}")
    for name, value in (
        ("hold_sec", hold_sec),
        ("start_delay_sec", start_delay_sec),
        ("reset_wait_sec", reset_wait_sec),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    for name, value in (("speed", speed), ("force", force)):
        if not 0 <= int(value) <= 255:
            raise ValueError(f"{name} must be within [0, 255], got {value}")
    max_width_mm = float(max_width_mm)
    if not math.isfinite(max_width_mm) or max_width_mm <= 0.0:
        raise ValueError(f"max_width_mm must be positive and finite, got {max_width_mm}")
    if open_width_mm is not None:
        open_width_mm = float(open_width_mm)
        if not math.isfinite(open_width_mm) or not 0.0 <= open_width_mm <= max_width_mm:
            raise ValueError(
                "open_width_mm must be within "
                f"[0, {max_width_mm:g}], got {open_width_mm}"
            )


def _load_gripper_factory():
    try:
        from pyrobotiqgripper import RobotiqGripper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyrobotiqgripper is not installed in this Python environment. "
            "On this machine, run the script with: conda run -n franka python"
        ) from exc
    return RobotiqGripper


def run_open_close(
    port="auto",
    device_id=9,
    cycles=1,
    hold_sec=2.0,
    start_delay_sec=3.0,
    reset_wait_sec=1.0,
    speed=255,
    force=100,
    open_width_mm=None,
    max_width_mm=85.0,
    skip_reset=False,
    gripper_factory=None,
    sleep_fn=time.sleep,
):
    """Connect to the gripper and execute blocking close/open cycles."""
    _validate_settings(
        cycles=cycles,
        hold_sec=hold_sec,
        start_delay_sec=start_delay_sec,
        reset_wait_sec=reset_wait_sec,
        speed=speed,
        force=force,
        open_width_mm=open_width_mm,
        max_width_mm=max_width_mm,
    )
    gripper_factory = gripper_factory or _load_gripper_factory()

    print(f"Connecting to Robotiq gripper on {port}...")
    gripper = gripper_factory(com_port=str(port), device_id=int(device_id))
    try:
        if not skip_reset:
            print("Resetting gripper...")
            gripper.reset()
            sleep_fn(float(reset_wait_sec))

        print("Activating gripper...")
        # Reset is handled explicitly above so it happens exactly once.
        gripper.activate(reset=False)

        if open_width_mm is not None:
            # Use the nominal Robotiq 0..255 travel endpoints. Passing explicit
            # endpoints prevents calibrate_bit() from moving through another
            # full open/close cycle.
            gripper.calibrate_bit(openbit=0, closebit=255)
            gripper.calibrate_mm(closemm=0.0, openmm=float(max_width_mm))

        if start_delay_sec > 0.0:
            print(f"Starting first close command in {float(start_delay_sec):g} seconds...")
            sleep_fn(float(start_delay_sec))

        for cycle_idx in range(int(cycles)):
            print(f"Cycle {cycle_idx + 1}/{int(cycles)}: closing...")
            gripper.close(speed=int(speed), force=int(force), wait=True)
            sleep_fn(float(hold_sec))

            if open_width_mm is None:
                print(f"Cycle {cycle_idx + 1}/{int(cycles)}: opening fully...")
                gripper.open(speed=int(speed), force=int(force), wait=True)
            else:
                print(
                    f"Cycle {cycle_idx + 1}/{int(cycles)}: "
                    f"opening to {float(open_width_mm):g} mm..."
                )
                gripper.move_mm(
                    float(open_width_mm),
                    speed=int(speed),
                    force=int(force),
                    wait=True,
                )
            sleep_fn(float(hold_sec))

        final_width = "fully open" if open_width_mm is None else f"{float(open_width_mm):g} mm"
        print(f"Finished. Gripper opening is {final_width}.")
    finally:
        disconnect = getattr(gripper, "disconnect", None)
        if callable(disconnect):
            disconnect()


def main():
    args = parse_args()
    try:
        run_open_close(
            port=args.port,
            device_id=args.device_id,
            cycles=args.cycles,
            hold_sec=args.hold_sec,
            start_delay_sec=args.start_delay_sec,
            reset_wait_sec=args.reset_wait_sec,
            speed=args.speed,
            force=args.force,
            open_width_mm=args.open_width_mm,
            max_width_mm=args.max_width_mm,
            skip_reset=args.skip_reset,
        )
    except KeyboardInterrupt:
        print("Interrupted; serial connection closed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Make sure no Polymetis gripper server is using the same serial port.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
