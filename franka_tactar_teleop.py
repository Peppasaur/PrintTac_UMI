import os

import cv2
import hydra
import psutil
from loguru import logger
from omegaconf import DictConfig

from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.real_world.teleoperation.teleop_server import TeleopServer

os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
cv2.setNumThreads(12)

total_cores = psutil.cpu_count()
num_cores_to_bind = min(8, total_cores)
os.sched_setaffinity(0, set(range(total_cores - num_cores_to_bind, total_cores)))


@hydra.main(
    config_path="reactive_diffusion_policy/config",
    config_name="real_world_env",
    version_base="1.3",
)
def main(cfg: DictConfig):
    transforms = RealWorldTransforms(option=cfg.task.transforms)
    teleop_server = TeleopServer(
        robot_server_ip=cfg.task.robot_server.host_ip,
        robot_server_port=cfg.task.robot_server.port,
        transforms=transforms,
        **cfg.task.teleop_server,
    )
    logger.info("Starting TactAR teleop server without the ROS robot publisher.")
    teleop_server.run()


if __name__ == "__main__":
    main()
