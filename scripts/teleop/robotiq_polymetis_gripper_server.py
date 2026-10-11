#!/usr/bin/env python3
import argparse
import logging
import threading
import time

from google.protobuf import timestamp_pb2
import grpc

import polymetis
import polymetis_pb2
import polymetis_pb2_grpc
from polymetis.robot_client.robotiq_gripper.third_party.robotiq_2finger_grippers.robotiq_2f_gripper import (
    Robotiq2FingerGripper,
)
from polymetis.robot_servers import GripperServerLauncher
from polymetis.utils import Spinner
from polymetis.utils.grpc_utils import check_server_exists

log = logging.getLogger(__name__)


class RobustRobotiqGripperClient:
    def __init__(
            self,
            server_ip,
            server_port,
            comport="/dev/ttyUSB1",
            stroke=0.085,
            hz=60,
            activate=True,
            activation_timeout=10.0,
            reset_before_activate=False,
            verbose=False):
        self.hz = hz
        self.verbose = verbose
        self.gripper = Robotiq2FingerGripper(comport=comport, stroke=stroke)

        if not self.gripper.init_success:
            raise RuntimeError(f"Unable to open Robotiq serial port {comport}")
        if not self.gripper.getStatus():
            raise RuntimeError(
                f"Serial port {comport} opened, but Robotiq did not return a valid status. "
                "This usually means the wrong ttyUSB device, no gripper power/RS485 link, "
                "or another process owns the adapter. Probe ports with: "
                "conda run -n polymetis python scripts/debug_robotiq_ports.py"
            )

        print(f"Robotiq connected on {comport}: {self._status_string()}")
        if activate:
            self._activate(timeout=activation_timeout, reset_first=reset_before_activate)
        else:
            print("Skipping Robotiq activation.")

        self.channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        self.connection = polymetis_pb2_grpc.GripperServerStub(self.channel)

        metadata = polymetis_pb2.GripperMetadata()
        metadata.polymetis_version = polymetis.__version__
        metadata.hz = self.hz
        metadata.max_width = self.gripper.stroke
        self.connection.InitRobotClient(metadata)
        print(
            f"Robotiq client connected to Polymetis gripper server "
            f"{server_ip}:{server_port}."
        )

    def _status_string(self):
        return (
            f"gACT={self.gripper.gACT}, gSTA={self.gripper.gSTA}, "
            f"gGTO={self.gripper.gGTO}, gOBJ={self.gripper.gOBJ}, "
            f"gFLT={self.gripper.gFLT}, width={self.gripper.get_pos():.4f}m"
        )

    def _wait_until_ready(self, timeout):
        deadline = time.time() + timeout
        last_print = 0.0
        while time.time() < deadline:
            if not self.gripper.getStatus():
                print("Robotiq status read failed while waiting for activation.")
            elif self.gripper.is_ready():
                print(f"Robotiq activated: {self._status_string()}")
                return True

            now = time.time()
            if self.verbose or now - last_print >= 1.0:
                print(f"Waiting for Robotiq activation: {self._status_string()}")
                last_print = now
            time.sleep(0.1)
        return False

    def _activate(self, timeout, reset_first):
        if self.gripper.is_ready():
            print(f"Robotiq already ready: {self._status_string()}")
            return

        if reset_first:
            print("Resetting Robotiq activation bit...")
            self.gripper.deactivate_gripper()
            self.gripper.sendCommand()
            time.sleep(0.5)
            self.gripper.getStatus()

        print("Activating Robotiq gripper...")
        self.gripper.activate_gripper()
        self.gripper.sendCommand()
        if self._wait_until_ready(timeout):
            return

        print("Activation did not finish. Trying reset-then-activate once...")
        self.gripper.deactivate_gripper()
        self.gripper.sendCommand()
        time.sleep(0.5)
        self.gripper.activate_gripper()
        self.gripper.sendCommand()
        if self._wait_until_ready(timeout):
            return

        raise RuntimeError(f"Unable to activate Robotiq gripper. Last state: {self._status_string()}")

    def get_gripper_state(self):
        state = polymetis_pb2.GripperState()
        if not self.gripper.getStatus():
            state.error_code = 1
            log.warning("Failed to read gripper state; returning last observed values.")

        state.timestamp.GetCurrentTime()
        state.width = float(self.gripper.get_pos())
        state.is_grasped = bool(self.gripper.object_detected())
        state.is_moving = bool(self.gripper.is_moving())
        return state

    def apply_gripper_command(self, cmd):
        width = 0.0 if cmd.grasp else float(cmd.width)
        self.gripper.goto(pos=width, vel=float(cmd.speed), force=float(cmd.force))
        self.gripper.sendCommand()
        return polymetis_pb2.Empty()

    def run(self):
        prev_timestamp = timestamp_pb2.Timestamp()
        spinner = Spinner(self.hz)
        while True:
            state = self.get_gripper_state()
            cmd = self.connection.ControlUpdate(state)
            if cmd.timestamp != prev_timestamp:
                self.apply_gripper_command(cmd)
                prev_timestamp = cmd.timestamp
            spinner.spin()


def parse_args():
    parser = argparse.ArgumentParser(description="Robust Polymetis gripper server for Robotiq 2F.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--comport", default="/dev/ttyUSB1")
    parser.add_argument("--stroke", type=float, default=0.085)
    parser.add_argument("--hz", type=float, default=60)
    parser.add_argument("--launch-timeout", type=float, default=15.0)
    parser.add_argument("--activation-timeout", type=float, default=10.0)
    parser.add_argument("--no-activate", action="store_true")
    parser.add_argument("--reset-before-activate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    gripper_server = GripperServerLauncher(args.host, args.port)
    server_thread = threading.Thread(target=gripper_server.run, daemon=True)
    server_thread.start()

    t0 = time.time()
    while not check_server_exists(args.host, args.port):
        time.sleep(0.1)
        if time.time() - t0 > args.launch_timeout:
            raise TimeoutError(
                f"Unable to locate Polymetis gripper server at "
                f"{args.host}:{args.port}"
            )

    client = RobustRobotiqGripperClient(
        server_ip=args.host,
        server_port=args.port,
        comport=args.comport,
        stroke=args.stroke,
        hz=args.hz,
        activate=not args.no_activate,
        activation_timeout=args.activation_timeout,
        reset_before_activate=args.reset_before_activate,
        verbose=args.verbose,
    )
    try:
        client.run()
    finally:
        gripper_server.server.stop(0)


if __name__ == "__main__":
    main()
