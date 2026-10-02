'''
Franka Interface Client and Interpolator:
Recieve commands from teleoperation server,
Interpolate the moving trajectory,
and Send commands to Franka through directly using Franka Control Interface (FCI).
'''

import threading
import time
import socket
import numpy as np
import torch
import enum
from fastapi import FastAPI, HTTPException
import scipy.spatial.transform as st
from loguru import logger
from typing import Dict
from collections import deque
import uvicorn
import argparse
import multiprocessing as mp
from scipy.spatial.transform import Rotation as ScipyRotation

from polymetis import RobotInterface, GripperInterface

from reactive_diffusion_policy.common.data_models import (TargetTCPRequest, MoveGripperRequest, BimanualRobotStates)
from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_wait


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_GRIPPER = 3

class FrankaServer:
    def __init__(self, 
                 robot_ip='172.16.1.1', 
                 robot_port=50051, 
                 gripper_ip='172.16.1.1',
                 gripper_port=50052,
                 host_ip='192.168.110.111', 
                 port=8092,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 vr_frequency=60,
                 frequency=300,
                 bimanual_teleop=False,
                 enable_gripper=True,
                 default_gripper_width=1.0,
                 **kwargs):
        """
        robot_ip: ip address of Desktop directly connected to Franka Emika
        gripper_ip: ip address of Desktop directly connected to Franka Emika
        host_ip: ip address of the Desktop running the FastAPI server
        frequency: frequency of control command sent to the robot
        vr_frequency: frequency of command from teleop server
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains.
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gripper_ip = gripper_ip
        self.gripper_port = gripper_port
        self.host_ip = host_ip    
        self.port = port
        self.vr_frequency = vr_frequency
        self.control_frequency = frequency
        self.control_cycle_time = 1.0 / self.control_frequency
        self.bimanual_teleop = bimanual_teleop
        self.enable_gripper = bool(enable_gripper)
        self.default_gripper_width = float(default_gripper_width)

        # Initialize the robot interface. Gripper is optional for no-gripper tasks.
        self.robot = RobotInterface(ip_address=self.robot_ip, port=self.robot_port)
        if self.enable_gripper:
            self.gripper = GripperInterface(ip_address=self.gripper_ip, port=self.gripper_port)
            logger.info(f"Gripper interface enabled at {self.gripper_ip}:{self.gripper_port}")
        else:
            self.gripper = None
            logger.warning(
                "Gripper interface disabled; state will use default width "
                f"{self.default_gripper_width:.4f} and gripper commands will be ignored."
            )
        
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale

        self.command_queue = deque(maxlen=256)
        self.pose_interp = None
        self.last_waypoint_time = None
        self.stop_event = threading.Event()
        self.command_thread = None
        self.command_thread_error = None
        self.command_loop_started_at = None
        self.last_command_loop_time = None
        self.last_received_waypoint = None
        self.last_scheduled_waypoint = None
        self.last_desired_pose = None
        self.last_aborted_waypoint = None
        self.last_controller_restart = None
        self.controller_restart_times = deque(maxlen=10)
        self.gripper_command_lock = threading.Lock()
        self.gripper_command_event = threading.Event()
        self.gripper_command_thread = None
        self.pending_gripper_command = None
        self.last_gripper_command = None
        self.last_gripper_command_log_time = 0.0
        
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post('/clear_fault')
        async def clear_fault():
            """
            Clear any fault in the robot.
            Polymetis RobotInterface has no method to clear fault, so we use a workaround.
            """
            logger.warning("Fault occurred on franka robot server")
            logger.info("Please clear the fault manually on the robot controller.")
            return {"message": "Fault cleared"}

        @self.app.get('/get_current_tcp/{robot_side}')
        async def get_current_tcp(robot_side: str):
            """
            Get the current TCP pose of the robot.
            Returns:
                (x, y, z, qw, qx, qy, qz), in flange coordinate
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                cur_tcp = self.get_current_tcp()
            except Exception as e:
                logger.error(f"Failed to get current TCP: {e}")
                raise HTTPException(status_code=500, detail="Failed to get current TCP")
            return cur_tcp

        @self.app.get('/get_current_robot_states')
        async def get_current_robot_state() -> BimanualRobotStates:
            """
            Get the current state of the robot.
            """
            try:
                state = self.get_robot_state()
                logger.info(f"Current Robot State: {state}")
                return BimanualRobotStates(**state)
            except Exception as e:
                logger.error(f"Failed to get robot state: {e}")
                raise HTTPException(status_code=500, detail="Failed to get robot state")

        @self.app.get('/get_control_debug_state')
        async def get_control_debug_state():
            """
            Report command-loop health. This intentionally does not touch robot hardware;
            it only exposes server-side bookkeeping for debugging stuck waypoint execution.
            """
            now = time.monotonic()
            command_thread_alive = (
                self.command_thread is not None and self.command_thread.is_alive()
            )
            loop_age = None
            if self.last_command_loop_time is not None:
                loop_age = now - self.last_command_loop_time
            return {
                "command_thread_alive": command_thread_alive,
                "stop_event_set": self.stop_event.is_set(),
                "command_thread_error": (
                    repr(self.command_thread_error)
                    if self.command_thread_error is not None
                    else None
                ),
                "queue_len": len(self.command_queue),
                "pose_interp_ready": self.pose_interp is not None,
                "command_loop_started_at": self.command_loop_started_at,
                "last_command_loop_time": self.last_command_loop_time,
                "last_command_loop_age": loop_age,
                "last_waypoint_time": self.last_waypoint_time,
                "last_received_waypoint": self.last_received_waypoint,
                "last_scheduled_waypoint": self.last_scheduled_waypoint,
                "last_desired_pose": self.last_desired_pose,
                "last_aborted_waypoint": self.last_aborted_waypoint,
                "last_controller_restart": self.last_controller_restart,
                "enable_gripper": self.enable_gripper,
                "default_gripper_width": self.default_gripper_width,
            }

        # Franka gripper command is non-realtime and low-frequency, thus there's no need to interpolate the gripper command 
        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Move the gripper to a target width with specified velocity and force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            if self.gripper is None:
                logger.info(
                    "Gripper disabled; ignoring move_gripper command "
                    f"width={request.width}, velocity={request.velocity}, "
                    f"force_limit={request.force_limit}"
                )
                return {"message": "Gripper disabled; command ignored"}
            self._queue_gripper_command(
                {
                    "type": "goto",
                    "width": float(request.width),
                    "velocity": float(request.velocity),
                    "force_limit": float(request.force_limit),
                }
            )
            return {"message": f"Gripper target queued: width {request.width}"}

        @self.app.post('/move_gripper_force/{robot_side}')
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Close the gripper with a specified force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            if self.gripper is None:
                logger.info(
                    "Gripper disabled; ignoring move_gripper_force command "
                    f"velocity={request.velocity}, force_limit={request.force_limit}"
                )
                return {"message": "Gripper disabled; command ignored"}
            self._queue_gripper_command(
                {
                    "type": "grasp",
                    "velocity": float(request.velocity),
                    "force_limit": float(request.force_limit),
                }
            )
            return {"message": f"Gripper grasp queued: force {request.force_limit}"}

        @self.app.post('/stop_gripper/{robot_side}')
        async def stop_gripper(robot_side: str)-> Dict[str, str]:
            """
            Stop the gripper's current motion.
            Polymetis GripperInterface has no stop method, and gripper motion is non-realtime and blocking.
            So we just log the information here.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            if self.gripper is None:
                logger.info("Gripper disabled; ignoring stop_gripper command")
                return {"message": "Gripper disabled; command ignored"}
            # self.gripper.stop()
            logger.info("Gripper stopped successfully")
            return {"message": "Gripper stopped"}

        @self.app.post('/move_tcp/{robot_side}')
        async def move_tcp(robot_side: str, request: TargetTCPRequest):
            '''
            Move the robot to a target TCP pose.
            Add a new low-frequency target pose to the command queue, waiting for the interpolator to process it.
            '''
            if robot_side != "left":
                logger.info("Only left arm is supported")

            command_thread_alive = (
                self.command_thread is not None and self.command_thread.is_alive()
            )
            loop_age = None
            if self.last_command_loop_time is not None:
                loop_age = time.monotonic() - self.last_command_loop_time
            if (
                not command_thread_alive
                or self.stop_event.is_set()
                or (loop_age is not None and loop_age > 1.0)
            ):
                detail = {
                    "message": "Franka command loop is not healthy; refusing waypoint.",
                    "command_thread_alive": command_thread_alive,
                    "stop_event_set": self.stop_event.is_set(),
                    "last_command_loop_age": loop_age,
                    "command_thread_error": (
                        repr(self.command_thread_error)
                        if self.command_thread_error is not None
                        else None
                    ),
                }
                logger.error(detail)
                raise HTTPException(status_code=503, detail=detail)
                
            target_7d_pose = np.array(request.target_tcp) # (x, y, z, qw, qx, qy, qz), in flange coordinate
            pos = target_7d_pose[:3]
            quat_wxyz = target_7d_pose[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]] # wxyz to xyzw
            rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
            target_pose = np.concatenate([pos, rotvec]) # (x, y, z, rx, ry, rz)， in flange coordinate
            
            curr_time = time.monotonic()
            command_duration = request.target_duration
            if command_duration is None:
                command_duration = 1 / self.vr_frequency
            command_duration = max(float(command_duration), 2.0 * self.control_cycle_time)
            target_time = curr_time + command_duration # target time set at half control cycle time in the future
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_pose': target_pose,
                'target_time': target_time
            })
            self.last_received_waypoint = {
                "target_pose": target_pose.tolist(),
                "target_time": target_time,
                "command_duration": command_duration,
                "queue_len_after_append": len(self.command_queue),
            }
            
            return {"message": "Waypoint added for franka robot"}

        @self.app.post('/birobot_go_home')
        async def go_home():
            """
            Move the robot to its home position.
            """
            logger.info("Moving Franka robot to its home position...")
            try:
                self.go_home()
            except Exception as e:
                logger.error(f"Failed to move robot to home position: {e}")
                raise HTTPException(status_code=500, detail="Failed to move robot to home position")
            return {"message": "Robot moved to home position"}
          

    def get_current_tcp(self):
        """
        Get the current TCP pose of the robot.
        Returns:
            (x, y, z, qw, qx, qy, qz), in flange coordinate
        """
        data = self.robot.get_ee_pose() # (position, quaternion(xyzw))
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([pos, quat_wxyz]).tolist()

    def _fallback_gripper_width(self):
        command = self.last_gripper_command
        if command is not None:
            if command.get("type") == "goto" and "width" in command:
                return float(command["width"])
            if command.get("type") == "grasp":
                return 0.0
        return self.default_gripper_width

    def get_robot_state(self):
        # libfranka Gripper State has no element 'gripper_force'
        # libfranka Robot State has no element 'tcp_velocities'
        if self.gripper is None:
            gripper_width = self.default_gripper_width
            gripper_force = 0.0
        else:
            try:
                gripper_state = self.gripper.get_state()
                gripper_width = float(getattr(gripper_state, "width", 0.0))
                gripper_force = float(getattr(gripper_state, "force", 0.0))
            except Exception as e:
                gripper_width = self._fallback_gripper_width()
                gripper_force = 0.0
                logger.warning(
                    "Failed to get gripper state; using fallback width "
                    f"{gripper_width:.4f}m instead of reporting zero: {e}"
                )

        try:
            tcp_wrench = self.get_tcp_wrench_flange().tolist() # (fx, fy, fz, mx, my, mz)
        except Exception as e:
            logger.warning(f"Failed to get TCP wrench, using zeros: {e}")
            tcp_wrench = [0.0] * 6

        try:
            tcp_vel = self.get_tcp_velocity_flange().tolist() # (vx, vy, vz, wx, wy, wz)
        except Exception as e:
            logger.warning(f"Failed to get TCP velocity, using zeros: {e}")
            tcp_vel = [0.0] * 6

        ee_pose = self.get_current_tcp()
        return {
            "leftRobotTCP": ee_pose, # (x, y, z, qw, qx, qy, qz), in flange coordinate
            "leftRobotTCPWrench": tcp_wrench, # (fx, fy, fz, mx, my, mz), in flange coordinate
            "leftRobotTCPVel": tcp_vel, # (vx, vy, vz, wx, wy, wz), in flange coordinate
            "leftGripperState": [gripper_width, gripper_force] # (width, force)
        }

    def _queue_gripper_command(self, command):
        command = dict(command)
        command["queued_time"] = time.monotonic()
        with self.gripper_command_lock:
            self.pending_gripper_command = command
        self.gripper_command_event.set()

    def process_gripper_commands(self):
        logger.info("Franka gripper command worker started.")
        try:
            while not self.stop_event.is_set():
                if not self.gripper_command_event.wait(timeout=0.1):
                    continue
                with self.gripper_command_lock:
                    command = self.pending_gripper_command
                    self.pending_gripper_command = None
                    self.gripper_command_event.clear()
                if command is None or self.gripper is None:
                    continue

                try:
                    if command["type"] == "goto":
                        self.gripper.goto(
                            width=command["width"],
                            speed=command["velocity"],
                            force=command["force_limit"],
                            blocking=False,
                        )
                        self._maybe_log_gripper_command(
                            "Gripper moving to width "
                            f"{command['width']} with velocity {command['velocity']} "
                            f"and force limit {command['force_limit']}"
                        )
                    elif command["type"] == "grasp":
                        self.gripper.grasp(
                            speed=command["velocity"],
                            force=command["force_limit"],
                            blocking=False,
                        )
                        self._maybe_log_gripper_command(
                            "Gripper grasping with force "
                            f"{command['force_limit']} and velocity {command['velocity']}"
                        )
                    self.last_gripper_command = command
                except Exception as e:
                    logger.error(f"Failed to execute gripper command: {e}")
        finally:
            logger.info("Franka gripper command worker stopped.")

    def _maybe_log_gripper_command(self, message):
        now = time.monotonic()
        if now - self.last_gripper_command_log_time > 1.0:
            logger.info(message)
            self.last_gripper_command_log_time = now

    def get_base_to_flange_rotation_matrix(self) -> torch.Tensor:
        """
        Return:
            torch.Tensor: 3x3 rotation matrix from robot base frame to flange frame
        """
        joint_pos = self.robot.get_joint_positions()
        _pos, quat = self.robot.robot_model.forward_kinematics(joint_pos) # quat: (x, y, z, w)

        # transform quaternion to rotation matrix
        quat_np = quat.numpy()
        r = ScipyRotation.from_quat(quat_np)
        rotation_matrix = torch.from_numpy(r.as_matrix()).to(joint_pos.dtype) # transformation matrix from base to flange(3x3)
        
        return rotation_matrix
    
    def get_tcp_wrench(self) -> torch.Tensor:
        """
        Return:
            torch.Tensor: TCP wrench under robot base frame (fx, fy, fz, mx, my, mz)
            Additional coordinate transformation is needed if want to convert to flange frame(refer to get_tcp_wrench_flange).
        """
        robot_state = self.robot.get_robot_state()
        joint_pos = torch.Tensor(robot_state.joint_positions)
        tau_external = torch.Tensor(robot_state.motor_torques_external)

        # compute Jacobian matrix
        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)

        # compute TCP wrench
        # F_tcp = (J^T)^+ * tau_external
        # J_transpose_pseudo_inv = torch.linalg.pinv(jacobian.T)
        # wrench = J_transpose_pseudo_inv @ tau_external
        wrench, _, _, _ = torch.linalg.lstsq(jacobian.T, tau_external)
        return wrench

    def get_tcp_velocity(self) -> torch.Tensor:
        """
        Return:
            torch.Tensor: (vx, vy, vz, wx, wy, wz),
            TCP velocity under robot base frame
            Additional coordinate transformation is needed if want to convert to flange frame.(refer to get_tcp_velocity_flange).
        """
        robot_state = self.robot.get_robot_state()
        joint_pos = torch.Tensor(robot_state.joint_positions)
        joint_vel = torch.Tensor(robot_state.joint_velocities)

        # compute Jacobian matrix J
        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)

        # compute TCP velocity V = J * q_dot
        tcp_velocity = jacobian @ joint_vel

        return tcp_velocity

    def get_tcp_velocity_flange(self) -> torch.Tensor:
        """
        Returns:
            TCP velocity under flange frame
        """
        R_flange_in_base = self.get_base_to_flange_rotation_matrix()
        R_base_to_flange = R_flange_in_base.T
        tcp_velocity_base = self.get_tcp_velocity()

        v_base = tcp_velocity_base[0:3]
        w_base = tcp_velocity_base[3:6]
        v_flange = R_base_to_flange @ v_base
        w_flange = R_base_to_flange @ w_base       
        tcp_velocity_flange = torch.cat([v_flange, w_flange])
        
        return tcp_velocity_flange

    def get_tcp_wrench_flange(self) -> torch.Tensor:
        """
        Returns:            
            TCP wrench under flange frame
        """
        R_flange_in_base = self.get_base_to_flange_rotation_matrix()
        R_base_to_flange = R_flange_in_base.T
        tcp_wrench_base = self.get_tcp_wrench()

        f_flange = R_base_to_flange @ tcp_wrench_base[0:3]
        m_flange = R_base_to_flange @ tcp_wrench_base[3:6]
        tcp_wrench_flange = torch.cat([f_flange, m_flange])
        
        return tcp_wrench_flange

    def go_home(self):
        home_joint_positions = [-0.07, -0.96, -0.01, -2.55, -0.09, 2.14, 0.59]
        homing_duration = 8.0
        logger.info(f"Moving Franka robot to home position: {home_joint_positions} with duration {homing_duration}s")
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(home_joint_positions),
            time_to_go=homing_duration
        )

    def _start_cartesian_impedance(self):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(self.Kx),
            Kxd=torch.Tensor(self.Kxd)
        )

    def _reset_interpolator_to_current_pose(self):
        curr_flange_pose = self.get_ee_pose()
        curr_time = time.monotonic()
        self.pose_interp = PoseTrajectoryInterpolator(
            times=[curr_time],
            poses=[curr_flange_pose]
        )
        self.last_waypoint_time = curr_time
        return curr_flange_pose, curr_time

    @staticmethod
    def _is_missing_controller_error(exc):
        return "no controller running" in str(exc).lower()

    def _recover_missing_controller(self, exc):
        now = time.monotonic()
        self.controller_restart_times.append(now)
        recent_restarts = [
            t for t in self.controller_restart_times
            if now - t < 5.0
        ]
        if len(recent_restarts) >= 5:
            raise RuntimeError(
                "Polymetis controller dropped repeatedly within 5s; "
                "refusing automatic restart."
            ) from exc

        logger.warning(
            "Polymetis controller is not running; restarting Cartesian impedance "
            "at the current robot pose and dropping stale waypoints."
        )
        self.command_queue.clear()
        curr_pose, curr_time = self._reset_interpolator_to_current_pose()
        self._start_cartesian_impedance()
        self.last_controller_restart = {
            "time": now,
            "pose": np.asarray(curr_pose).tolist(),
            "interpolator_time": curr_time,
            "reason": repr(exc),
        }

    def process_commands(self):
        """
        Main loop for processing commands and updating the interpolator.
        """
        try:
            if self.pose_interp is None:
                self._reset_interpolator_to_current_pose()

            # start franka cartesian impedance policy
            self._start_cartesian_impedance()

            t_start = time.monotonic()
            self.command_loop_started_at = t_start
            last_print = time.monotonic()
            count = 0
            iter_idx = 0

            while not self.stop_event.is_set():
                t_now = time.monotonic()
                self.last_command_loop_time = t_now
                flange_pos = self.pose_interp(t_now) # (x, y, z, rx, ry, rz), in flange coordinate
                self.last_desired_pose = np.asarray(flange_pos).tolist()

                try:
                    self.robot.update_desired_ee_pose(
                        position=torch.Tensor(flange_pos[:3]),
                        orientation=torch.Tensor(st.Rotation.from_rotvec(flange_pos[3:]).as_quat()) # (qx, qy, qz, qw)
                    )
                except Exception as e:
                    if self._is_missing_controller_error(e):
                        self._recover_missing_controller(e)
                        continue
                    raise

                count += 1
                if t_now - last_print > 1.0:
                    logger.info(f"update_desired_ee_pose called {count} times in last second")
                    count = 0
                    last_print = t_now

                '''
                Process high-level commands from VR
                command_queue: low-frequency moving command from VR
                target_time: timestamp where new target pose should be inserted into interpolator
                curr_time: the time base of interpolator
                last_waypoint_time: last target pose in the interpolator
                '''
                try:
                    command = self.command_queue.popleft()
                    if command['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        curr_time = t_now + self.control_cycle_time
                        target_time = float(command['target_time'])

                        if curr_time >= target_time:
                            self.last_aborted_waypoint = {
                                "reason": "curr_time >= target_time",
                                "curr_time": curr_time,
                                "target_time": target_time,
                                "target_pose": np.asarray(target_pose).tolist(),
                            }
                            logger.warning(f"curr_time ({curr_time:.6f}) >= target_time ({target_time:.6f}), this target point is aborted.")
                        if self.last_waypoint_time is not None and self.last_waypoint_time >= curr_time:
                            self.last_aborted_waypoint = {
                                "reason": "last_waypoint_time >= curr_time",
                                "last_waypoint_time": self.last_waypoint_time,
                                "curr_time": curr_time,
                                "target_time": target_time,
                                "target_pose": np.asarray(target_pose).tolist(),
                            }
                            logger.warning(f"last_waypoint_time ({self.last_waypoint_time:.6f}) >= curr_time ({curr_time:.6f}), the trajectory may be twisted.")

                        self.pose_interp = self.pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=self.last_waypoint_time
                        )
                        self.last_waypoint_time = target_time
                        self.last_scheduled_waypoint = {
                            "target_pose": np.asarray(target_pose).tolist(),
                            "target_time": target_time,
                            "curr_time": curr_time,
                            "queue_len_after_pop": len(self.command_queue),
                        }
                except IndexError:
                    pass

                t_wait_util = t_start + (iter_idx + 1) * self.control_cycle_time
                precise_wait(t_wait_util, time_func=time.monotonic)
                iter_idx += 1
        except BaseException as e:
            self.command_thread_error = e
            self.stop_event.set()
            logger.exception(f"Franka command loop failed: {e}")
            raise
        finally:
            logger.info("Franka command loop stopped.")

    def get_ee_pose(self):
        data = self.robot.get_ee_pose() # (position, quaternion(xyzw))
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def assert_http_port_available(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self.host_ip, self.port))
            except OSError as e:
                raise RuntimeError(
                    f"HTTP robot server port {self.host_ip}:{self.port} is not available. "
                    "Refusing to start the 300Hz Franka control loop."
                ) from e

    def run(self):
        self.assert_http_port_available()
        logger.info("Interpolation Controller started, waiting for commands...")
        command_thread = threading.Thread(
            target=self.process_commands,
            name="franka-command-loop",
            daemon=True
        )
        self.command_thread = command_thread
        gripper_thread = None
        if self.gripper is not None:
            gripper_thread = threading.Thread(
                target=self.process_gripper_commands,
                name="franka-gripper-command-loop",
                daemon=True,
            )
            self.gripper_command_thread = gripper_thread
        try:
            command_thread.start()
            if gripper_thread is not None:
                gripper_thread.start()
            logger.info("Start FastAPI Franka Server!")
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
        except KeyboardInterrupt:
            logger.info("Franka server interrupted by user.")
        except BaseException as e:
            logger.exception(e)
        finally:
            logger.info("Stopping Franka command loop...")
            self.stop_event.set()
            command_thread.join(timeout=2.0)
            if command_thread.is_alive():
                logger.warning(
                    "Franka command loop did not stop within 2 seconds; "
                    "terminating current Polymetis policy and exiting."
                )
            self.gripper_command_event.set()
            if gripper_thread is not None:
                gripper_thread.join(timeout=2.0)
                if gripper_thread.is_alive():
                    logger.warning("Franka gripper command loop did not stop within 2 seconds.")
            try:
                self.robot.terminate_current_policy()
            except Exception as e:
                logger.warning(f"Failed to terminate current Polymetis policy cleanly: {e}")
            logger.info("Franka Interpolation Controller terminated.")

def main():
    parser = argparse.ArgumentParser(description="Franka Server with Polymetis")
    parser.add_argument("--robot_ip", type=str, default='172.16.1.1', help="IP address of the robot")
    parser.add_argument("--robot_port", type=int, default=50051, help="Polymetis robot server port")
    parser.add_argument("--gripper_ip", type=str, default='172.16.1.1', help="IP address of the gripper")
    parser.add_argument("--gripper_port", type=int, default=50052, help="Polymetis gripper server port")
    parser.add_argument("--host_ip", type=str, default="localhost", help="Host IP for FastAPI server")
    parser.add_argument("--port", type=int, default=8092, help="Port for FastAPI server")
    parser.add_argument("--Kx_scale", type=float, default=1.0, help="Cartesian impedance stiffness scale")
    parser.add_argument("--Kxd_scale", type=float, default=1.0, help="Cartesian impedance damping scale")
    parser.add_argument("--vr_frequency", type=float, default=60, help="Low-frequency waypoint command rate")
    parser.add_argument("--frequency", type=float, default=300, help="High-frequency Polymetis update rate")
    parser.add_argument("--bimanual_teleop", action="store_true", help="Keep original bimanual teleop flag")
    parser.add_argument("--disable_gripper", action="store_true", help="Run without a Polymetis gripper server")
    parser.add_argument("--default_gripper_width", type=float, default=1.0, help="Reported gripper width when gripper is disabled")
    args = parser.parse_args()

    server = FrankaServer(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        gripper_ip=args.gripper_ip,
        gripper_port=args.gripper_port,
        host_ip=args.host_ip,
        port=args.port,
        Kx_scale=args.Kx_scale,
        Kxd_scale=args.Kxd_scale,
        vr_frequency=args.vr_frequency,
        frequency=args.frequency,
        bimanual_teleop=args.bimanual_teleop,
        enable_gripper=not args.disable_gripper,
        default_gripper_width=args.default_gripper_width,
    )
    server.run()

if __name__ == "__main__":
    main()
