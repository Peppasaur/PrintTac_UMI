## Customized Deployment Guide

### Add Customized Tasks
There are two kinds of task configs: **shared configs** and **policy-related configs**.
#### Shared Configs
Shared configs define the action DoFs and gripper parameters, as well as the sensor parameters of a specific task. This config will also be used for teleoperation in TactAR.
You can take [reactive_diffusion_policy/config/task/real_peel_two_realsense_one_gelsight_one_mctac_24fps.yaml](../reactive_diffusion_policy/config/task/real_peel_two_realsense_one_gelsight_one_mctac_24fps.yaml)
as an example.

Note that the `teleop_mode` represents the degrees of freedom of the robot action in teleoperation, so you can freeeze some action dimensions by changing the `teleop_mode`.
You can add a new teleoperation mode in [reactive_diffusion_policy/common/data_models.py](../reactive_diffusion_policy/common/data_models.py) and
modify [reactive_diffusion_policy/real_world/teleoperation/teleop_server.py](../reactive_diffusion_policy/real_world/teleoperation/teleop_server.py) accordingly.

#### Policy-related Configs
Policy-related configs define the observation shape, action shape, environment runner parameters, and dataset parameters.

Note that the `mode` in `data_processing_params` represents the `SENSOR_MODE`, which defines how much cameras or tactile sensors are used for observation.
Make sure the `mode` in the task config matches the `SENSOR_MODE` in [post_process_data.py](post_process_data.py).
You can add a new sensor mode in [reactive_diffusion_policy/common/data_models.py](../reactive_diffusion_policy/common/data_models.py) and
modify [post_process_data.py](post_process_data.py) and [reactive_diffusion_policy/real_world/post_process_utils.py](../reactive_diffusion_policy/real_world/post_process_utils.py) accordingly.

### Add Customized Tactile / Force Sensors
1. **Implement the Sensor Publisher.**
   Refer to [reactive_diffusion_policy/real_world/publisher/gelsight_camera_publisher.py](../reactive_diffusion_policy/real_world/publisher/gelsight_camera_publisher.py)
   and implement a similar publisher for your sensor.
   This publisher will publish the image, markers or wrench of the sensor to the ROS2 topic. Besides, it will send the 3D deformation field and optional image streams to the TactAR APP in Meta Quest3. You can customize your own format of 3D deformation field or camera streaming by changing the config of these sensors.
   > Because Flexiv Rizon 4 is equipped with built-in joint torque sensors, we have implemented the publisher in
     [reactive_diffusion_policy/real_world/publisher/bimanual_robot_publisher.py](../reactive_diffusion_policy/real_world/publisher/bimanual_robot_publisher.py).
     If you want to use a separate force sensor, you can add an individual force sensor publisher.
2. **Modify the Device Mapping Server.**
   Our experiments use many different settings of sensor hardwares, and these sensors may not work correctly, so we use `Device Mapping Server` as an online database for other processes to query the current hardware settings.
   It provides the mapping from the sensor to the ROS2 topic name,
   which is requested by the `Data Recorder` and `Real Runner`. Please add the new sensor mapping in [reactive_diffusion_policy/real_world/device_mapping/device_mapping_server.py](../reactive_diffusion_policy/real_world/device_mapping/device_mapping_server.py).
    
3. **Modify Services.**
   Modify the following services to be compatible with the new sensors:
   - Data Recorder: [reactive_diffusion_policy/real_world/teleoperation/data_recorder.py](../reactive_diffusion_policy/real_world/teleoperation/data_recorder.py)
   - Real Runner: [reactive_diffusion_policy/env_runner/real_runner.py](../reactive_diffusion_policy/env_runner/real_runner.py)

### Add Customized Robots
1. **Implement the Robot Server.**
   Refer to [reactive_diffusion_policy/real_world/robot/bimanual_flexiv_server.py](../reactive_diffusion_policy/real_world/robot/bimanual_flexiv_server.py)
   and implement a robot server for your robot with the same API.
   This server is requested by the Teleop Server and Real Env.
2. **Implement the Robot Publisher.**
   Refer to [reactive_diffusion_policy/real_world/publisher/bimanual_robot_publisher.py](../reactive_diffusion_policy/real_world/publisher/bimanual_robot_publisher.py)
   and implement a similar publisher for your robot.
3. **(Optional) Modify Services.**
   The following services may need to be modified to be compatible with the new robot:
   - Teleop Server: [reactive_diffusion_policy/real_world/teleoperation/teleop_server.py](../reactive_diffusion_policy/real_world/teleoperation/teleop_server.py)
   - Real Env: [reactive_diffusion_policy/env/real_bimanual/real_env.py](../reactive_diffusion_policy/env/real_bimanual/real_env.py)
