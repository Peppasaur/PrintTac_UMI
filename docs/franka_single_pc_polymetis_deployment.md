# Single-PC Franka Research 3 Polymetis Deployment

This setup keeps the README's Franka/Polymetis control path, but runs the robot-side server and the client-side teleoperation services on one computer.

## Network Used On This Machine

- Workstation Ethernet to FR3: `172.16.0.1/24`
- FR3 robot IP used by the scripts: `172.16.0.2`
- Workstation Wi-Fi / TactAR-facing IP detected during setup: `10.16.1.240`
- Polymetis arm gRPC server: `127.0.0.1:50051`
- Polymetis gripper gRPC server: `127.0.0.1:50052`
- RDP-compatible Franka HTTP robot server: `127.0.0.1:8092`
- TactAR teleop HTTP server: `10.16.1.240:8082`

Override these with environment variables if the network changes:

```bash
FR3_ROBOT_IP=172.16.0.2
POLYMETIS_HOST=127.0.0.1
POLYMETIS_PORT=50051
POLYMETIS_GRIPPER_HOST=127.0.0.1
POLYMETIS_GRIPPER_PORT=50052
RDP_ROBOT_SERVER_HOST=127.0.0.1
RDP_ROBOT_SERVER_PORT=8092
```

## Startup Order

Open Franka Desk, unlock the joints, and activate FCI first. Then run each command in a separate terminal from the repo root:

```bash
./scripts/franka_polymetis_launch_arm.sh
```

```bash
./scripts/franka_polymetis_launch_gripper.sh
```

```bash
./scripts/franka_polymetis_robot_server.sh
```

```bash
./scripts/franka_tactar_teleop.sh
```

The first two commands are the Polymetis hardware servers. The third command exposes the RDP robot-server API used by this repo. The fourth command runs the TactAR teleoperation server without launching ROS2 camera or robot publishers.

## Full README Pipeline

For camera publishing, data recording, and policy rollout, the README still expects `camera_node_launcher.py`, `record_data.py`, and `eval.sh`. On this machine, the existing `umi` environment has the FastAPI/Hydra/Torch side but cannot import ROS Jazzy `rclpy`; the existing `franka` environment can import ROS Jazzy `rclpy` but is missing several RDP Python dependencies. Use the teleoperation-only path above first, then finish one Python 3.12 environment with the RDP dependencies before running the full ROS2 pipeline.
