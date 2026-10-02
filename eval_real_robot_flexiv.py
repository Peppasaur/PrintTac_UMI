# %%
import pathlib
import torch
import dill
import hydra
from omegaconf import OmegaConf
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy

import os
import psutil

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 10
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _sync_relative_gripper_action_from_checkpoint(cfg, payload):
    checkpoint_cfg = payload.get("cfg")
    trained_relative_gripper_action = bool(OmegaConf.select(
        checkpoint_cfg,
        "task.dataset.relative_gripper_action",
        default=False,
    ))
    configured_relative_gripper_action = bool(OmegaConf.select(
        cfg,
        "task.env_runner.use_relative_gripper_action",
        default=False,
    ))
    if configured_relative_gripper_action != trained_relative_gripper_action:
        print(
            "Using relative gripper action mode from checkpoint: "
            f"{configured_relative_gripper_action} -> {trained_relative_gripper_action}"
        )
    OmegaConf.update(
        cfg,
        "task.dataset.relative_gripper_action",
        trained_relative_gripper_action,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "task.env_runner.use_relative_gripper_action",
        trained_relative_gripper_action,
        force_add=True,
    )

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'reactive_diffusion_policy', 'config')),
    config_name="train_diffusion_unet_real_image_workspace"
)
def main(cfg):
    # load checkpoint
    ckpt_path = cfg.ckpt_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    _sync_relative_gripper_action_from_checkpoint(cfg, payload)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        if 'latent' in cfg.name:
            policy.at.set_normalizer(policy.normalizer)

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 8  # DDIM inference iterations
    else:
        raise NotImplementedError

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner)
    env_runner.run(policy)


# %%
if __name__ == '__main__':
    main()
