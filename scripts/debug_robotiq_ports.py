#!/usr/bin/env python3
import argparse
import glob
import os
import time

from polymetis.robot_client.robotiq_gripper.third_party.robotiq_2finger_grippers.robotiq_2f_gripper import (
    Robotiq2FingerGripper,
)


def unique_existing(paths):
    result = []
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        key = (path, real)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def default_ports():
    candidates = []
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/serial/by-path/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    return unique_existing(candidates)


def status_string(gripper):
    return (
        f"gACT={gripper.gACT}, gSTA={gripper.gSTA}, gGTO={gripper.gGTO}, "
        f"gOBJ={gripper.gOBJ}, gFLT={gripper.gFLT}, width={gripper.get_pos():.4f}m"
    )


def probe_port(port, stroke, attempts, delay):
    print(f"== {port} ==")
    try:
        gripper = Robotiq2FingerGripper(comport=port, stroke=stroke)
    except Exception as exc:
        print(f"open_error: {exc}")
        return False

    print(f"init_success={bool(gripper.init_success)}")
    if not gripper.init_success:
        return False

    for attempt in range(1, attempts + 1):
        try:
            ok = bool(gripper.getStatus())
        except Exception as exc:
            print(f"status_attempt={attempt} error={exc}")
            ok = False
        if ok:
            print(f"status_attempt={attempt} ok=True {status_string(gripper)}")
            return True
        print(f"status_attempt={attempt} ok=False")
        time.sleep(delay)
    return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Probe Robotiq serial ports without activating or moving the gripper. "
            "A usable port should report ok=True for getStatus()."
        )
    )
    parser.add_argument("ports", nargs="*", help="Ports to probe. Defaults to common tty/by-id paths.")
    parser.add_argument("--stroke", type=float, default=0.085)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    ports = unique_existing(args.ports) if args.ports else default_ports()
    if not ports:
        print("No candidate serial ports found.")
        return 1

    ok_ports = []
    for port in ports:
        if probe_port(port, stroke=args.stroke, attempts=args.attempts, delay=args.delay):
            ok_ports.append(port)

    print("")
    if ok_ports:
        print("Robotiq-responsive ports:")
        for port in ok_ports:
            print(f"  {port}")
        print("Use one of these with ROBOTIQ_PORT=<port> scripts/franka_polymetis_launch_gripper.sh")
        return 0

    print("No probed port returned a valid Robotiq status.")
    print("Check power, RS485/USB adapter wiring, permissions, and whether another process owns the port.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
