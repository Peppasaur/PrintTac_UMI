import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import scipy.spatial.transform as st
import torch
from polymetis import GripperInterface, RobotInterface

from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_wait


class FrankaPolymetisRobot:
    def __init__(
            self,
            robot_ip="127.0.0.1",
            robot_port=50051,
            gripper_ip="127.0.0.1",
            gripper_port=50052,
            frequency=300,
            vr_frequency=60,
            kx_scale=1.0,
            kxd_scale=1.0,
            use_grav_comp=True,
            enable_gripper=True):
        self.robot = RobotInterface(
            ip_address=robot_ip,
            port=robot_port,
            use_grav_comp=use_grav_comp,
        )
        self.gripper = (
            GripperInterface(ip_address=gripper_ip, port=gripper_port)
            if enable_gripper else None
        )
        self.control_frequency = frequency
        self.control_cycle_time = 1.0 / frequency
        self.vr_frequency = vr_frequency
        self.kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * kx_scale
        self.kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * kxd_scale
        self.command_queue = deque(maxlen=256)
        self.lock = threading.Lock()
        self.pose_interp = None
        self.last_waypoint_time = None
        self._stop = threading.Event()

    def start(self):
        thread = threading.Thread(target=self._control_loop, daemon=True)
        thread.start()
        return thread

    def close(self):
        self._stop.set()
        try:
            self.robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    def get_current_tcp(self):
        pos, quat_xyzw = self.robot.get_ee_pose()
        quat_xyzw = quat_xyzw.numpy()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([pos.numpy(), quat_wxyz]).tolist()

    def get_ee_pose_rotvec(self):
        pos, quat_xyzw = self.robot.get_ee_pose()
        rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
        return np.concatenate([pos.numpy(), rotvec])

    def get_robot_state(self):
        robot_state = self.robot.get_robot_state()
        ee_pose = self.get_current_tcp()
        try:
            tcp_vel = self._get_tcp_velocity_flange(robot_state).tolist()
        except Exception:
            tcp_vel = [0.0] * 6
        try:
            tcp_wrench = self._get_tcp_wrench_flange(robot_state).tolist()
        except Exception:
            tcp_wrench = [0.0] * 6
        gripper_state = [0.0, 0.0]
        if self.gripper is not None:
            state = self.gripper.get_state()
            gripper_state = [
                float(getattr(state, "width", 0.0)),
                float(getattr(state, "force", 0.0)),
            ]
        return {
            "leftRobotTCP": ee_pose,
            "rightRobotTCP": [0.0] * 7,
            "leftRobotTCPVel": tcp_vel,
            "rightRobotTCPVel": [0.0] * 6,
            "leftRobotTCPWrench": tcp_wrench,
            "rightRobotTCPWrench": [0.0] * 6,
            "leftGripperState": gripper_state,
            "rightGripperState": [0.0, 0.0],
        }

    def move_tcp(self, target_tcp, target_duration=None):
        target_7d_pose = np.array(target_tcp, dtype=np.float64)
        pos = target_7d_pose[:3]
        quat_wxyz = target_7d_pose[3:]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        target_pose = np.concatenate([pos, rotvec])
        curr_time = time.monotonic()
        if target_duration is None:
            target_duration = 1.0 / self.vr_frequency
        target_duration = max(float(target_duration), 2.0 * self.control_cycle_time)
        target_time = curr_time + target_duration
        with self.lock:
            self.command_queue.append({
                "target_pose": target_pose,
                "target_time": target_time,
            })

    def move_gripper(self, width, velocity, force_limit):
        if self.gripper is None:
            raise RuntimeError(
                "Gripper is disabled on this robot server. Restart with "
                "ENABLE_GRIPPER=1 scripts/franka_polymetis_robot_server.sh "
                "after starting scripts/franka_polymetis_launch_gripper.sh."
            )
        self.gripper.goto(
            width=float(width),
            speed=float(velocity),
            force=float(force_limit),
        )

    def grasp_gripper(self, velocity, force_limit):
        if self.gripper is None:
            raise RuntimeError(
                "Gripper is disabled on this robot server. Restart with "
                "ENABLE_GRIPPER=1 scripts/franka_polymetis_robot_server.sh "
                "after starting scripts/franka_polymetis_launch_gripper.sh."
            )
        self.gripper.grasp(
            speed=float(velocity),
            force=float(force_limit),
        )

    def go_home(self):
        home = torch.Tensor([-0.07, -0.96, -0.01, -2.55, -0.09, 2.14, 0.59])
        self.robot.move_to_joint_positions(positions=home, time_to_go=8.0)

    def _control_loop(self):
        current_pose = self.get_ee_pose_rotvec()
        curr_time = time.monotonic()
        self.pose_interp = PoseTrajectoryInterpolator(
            times=[curr_time],
            poses=[current_pose],
        )
        self.last_waypoint_time = curr_time
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(self.kx),
            Kxd=torch.Tensor(self.kxd),
        )

        t_start = time.monotonic()
        iter_idx = 0
        while not self._stop.is_set():
            t_now = time.monotonic()
            flange_pos = self.pose_interp(t_now)
            self.robot.update_desired_ee_pose(
                position=torch.Tensor(flange_pos[:3]),
                orientation=torch.Tensor(st.Rotation.from_rotvec(flange_pos[3:]).as_quat()),
            )

            with self.lock:
                command = self.command_queue.popleft() if self.command_queue else None
            if command is not None:
                curr_time = t_now + self.control_cycle_time
                target_time = float(command["target_time"])
                if curr_time < target_time:
                    self.pose_interp = self.pose_interp.schedule_waypoint(
                        pose=command["target_pose"],
                        time=target_time,
                        curr_time=curr_time,
                        last_waypoint_time=self.last_waypoint_time,
                    )
                    self.last_waypoint_time = target_time

            precise_wait(
                t_start + (iter_idx + 1) * self.control_cycle_time,
                time_func=time.monotonic,
            )
            iter_idx += 1

    def _get_base_to_flange_rotation_matrix(self):
        joint_pos = self.robot.get_joint_positions()
        _pos, quat = self.robot.robot_model.forward_kinematics(joint_pos)
        return torch.from_numpy(st.Rotation.from_quat(quat.numpy()).as_matrix()).to(joint_pos.dtype)

    def _get_tcp_wrench(self, robot_state):
        joint_pos = torch.Tensor(robot_state.joint_positions)
        tau_external = torch.Tensor(robot_state.motor_torques_external)
        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)
        wrench, _, _, _ = torch.linalg.lstsq(jacobian.T, tau_external)
        return wrench

    def _get_tcp_velocity(self, robot_state):
        joint_pos = torch.Tensor(robot_state.joint_positions)
        joint_vel = torch.Tensor(robot_state.joint_velocities)
        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)
        return jacobian @ joint_vel

    def _get_tcp_velocity_flange(self, robot_state):
        rotation = self._get_base_to_flange_rotation_matrix().T
        tcp_velocity_base = self._get_tcp_velocity(robot_state)
        return torch.cat([
            rotation @ tcp_velocity_base[:3],
            rotation @ tcp_velocity_base[3:6],
        ])

    def _get_tcp_wrench_flange(self, robot_state):
        rotation = self._get_base_to_flange_rotation_matrix().T
        tcp_wrench_base = self._get_tcp_wrench(robot_state)
        return torch.cat([
            rotation @ tcp_wrench_base[:3],
            rotation @ tcp_wrench_base[3:6],
        ])


def make_handler(robot):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path == "/get_current_robot_states":
                    self._send_json(200, robot.get_robot_state())
                elif path == "/get_current_tcp/left":
                    self._send_json(200, robot.get_current_tcp())
                else:
                    self._send_json(404, {"detail": f"Unknown endpoint: {path}"})
            except Exception as exc:
                self._send_json(500, {"detail": str(exc)})

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                data = self._read_json()
                if path == "/clear_fault":
                    self._send_json(200, {"message": "Polymetis faults must be cleared at the robot/driver level."})
                elif path == "/move_tcp/left":
                    robot.move_tcp(
                        data["target_tcp"],
                        target_duration=data.get("target_duration"),
                    )
                    self._send_json(200, {"message": "Waypoint added for Franka robot"})
                elif path == "/move_gripper/left":
                    robot.move_gripper(
                        data.get("width", 0.05),
                        data.get("velocity", 0.1),
                        data.get("force_limit", 7.0),
                    )
                    self._send_json(200, {"message": "Gripper moving"})
                elif path == "/move_gripper_force/left":
                    robot.grasp_gripper(
                        data.get("velocity", 0.1),
                        data.get("force_limit", 7.0),
                    )
                    self._send_json(200, {"message": "Gripper grasping"})
                elif path == "/stop_gripper/left":
                    self._send_json(200, {"message": "Gripper stop acknowledged"})
                elif path == "/birobot_go_home":
                    robot.go_home()
                    self._send_json(200, {"message": "Robot moved to home position"})
                elif path.endswith("/right"):
                    self._send_json(400, {"detail": "Only the left arm is available in this single-FR3 deployment."})
                else:
                    self._send_json(404, {"detail": f"Unknown endpoint: {path}"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._send_json(500, {"detail": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="RDP-compatible Franka robot server using Polymetis.")
    parser.add_argument("--robot_ip", default="127.0.0.1")
    parser.add_argument("--robot_port", type=int, default=50051)
    parser.add_argument("--gripper_ip", default="127.0.0.1")
    parser.add_argument("--gripper_port", type=int, default=50052)
    parser.add_argument("--host_ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--frequency", type=int, default=300)
    parser.add_argument("--vr_frequency", type=int, default=60)
    parser.add_argument("--kx_scale", type=float, default=1.0)
    parser.add_argument("--kxd_scale", type=float, default=1.0)
    parser.add_argument("--polymetis_adds_gravity", action="store_true")
    parser.add_argument("--disable_gripper", action="store_true")
    args = parser.parse_args()

    robot = FrankaPolymetisRobot(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        gripper_ip=args.gripper_ip,
        gripper_port=args.gripper_port,
        frequency=args.frequency,
        vr_frequency=args.vr_frequency,
        kx_scale=args.kx_scale,
        kxd_scale=args.kxd_scale,
        use_grav_comp=not args.polymetis_adds_gravity,
        enable_gripper=not args.disable_gripper,
    )
    control_thread = robot.start()
    server = ThreadingHTTPServer((args.host_ip, args.port), make_handler(robot))
    print(f"RDP Franka Polymetis robot server listening on http://{args.host_ip}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        robot.close()
        control_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
