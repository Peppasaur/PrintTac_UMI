#!/usr/bin/env python3
"""Offline check for LDP/RDP eval actions from a checkpoint and zarr episode."""

import argparse
import os
import pathlib
import sys

import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reactive_diffusion_policy.common.action_utils import (  # noqa: E402
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="train_latent_diffusion_unet_real_image_workspace")
    parser.add_argument("--task", default="franka_polymetis_image_gelsight_emb_ldp_absolute_12fps")
    parser.add_argument("--at", default="at_wipe_lift_12fps")
    parser.add_argument("--at-load-dir", default=None)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--dataset-path", default="dataset/traj_rdp10d_command_downsample2")
    parser.add_argument("--episode", type=int, default=15)
    parser.add_argument("--latency-step", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def image_to_policy_tensor(images):
    images = np.asarray(images)
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    else:
        images = images.astype(np.float32)
    return np.moveaxis(images, -1, 1)


def main():
    args = parse_args()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(pathlib.Path(REPO_ROOT).joinpath("reactive_diffusion_policy", "config"))
    overrides = [
        f"task={args.task}",
        f"task.dataset_path={args.dataset_path}",
        f"+ckpt_path={args.ckpt_path}",
        f"at={args.at}",
    ]
    if args.at_load_dir:
        overrides.append(f"at_load_dir={args.at_load_dir}")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

    payload = torch.load(open(args.ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if "latent" in cfg.name:
        policy.at.set_normalizer(policy.normalizer)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    policy.eval().to(device)
    policy.num_inference_steps = int(args.num_inference_steps)

    rb = zarr.open(os.path.join(args.dataset_path, "replay_buffer.zarr"), mode="r")
    episode_ends = np.asarray(rb["meta/episode_ends"], dtype=np.int64)
    start = 0 if args.episode == 0 else int(episode_ends[args.episode - 1])
    end = int(episode_ends[args.episode])

    n_obs_steps = int(cfg.n_obs_steps)
    dataset_ratio = int(cfg.dataset_obs_temporal_downsample_ratio)
    dataset_obs_steps = int(cfg.dataset_obs_steps)
    horizon = int(cfg.horizon)
    shape_meta = cfg.task.shape_meta

    obs = {}
    abs_obs = {}
    for key, attr in shape_meta.obs.items():
        typ = attr.get("type", "low_dim")
        dim = attr.shape
        if typ == "rgb":
            obs[key] = image_to_policy_tensor(rb[f"data/{key}"][start:start + n_obs_steps])
        else:
            arr = np.asarray(rb[f"data/{key}"][start:start + n_obs_steps], dtype=np.float32)
            arr = arr[:, : int(dim[0])]
            abs_obs[key] = arr.copy()
            obs[key] = arr.copy()

    if bool(cfg.task.dataset.relative_action):
        base = abs_obs["left_robot_tcp_pose"][-1].copy()
        obs["left_robot_tcp_pose"] = absolute_actions_to_relative_actions(
            obs["left_robot_tcp_pose"],
            base_absolute_action=base,
        )
    else:
        base = abs_obs["left_robot_tcp_pose"][-1].copy()

    obs_torch = {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in obs.items()
    }

    ext = {}
    for key, attr in shape_meta.get("extended_obs", {}).items():
        dim = int(attr.shape[0])
        ext[key] = np.asarray(rb[f"data/{key}"][start:start + horizon], dtype=np.float32)[:, :dim]
    ext_torch = {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in ext.items()
    }

    with torch.no_grad():
        latent_dict = policy.predict_action(
            obs_torch,
            dataset_obs_temporal_downsample_ratio=dataset_ratio,
            return_latent_action=True,
        )
        direct_dict = policy.predict_action(
            obs_torch,
            dataset_obs_temporal_downsample_ratio=dataset_ratio,
            extended_obs_dict=ext_torch,
            return_latent_action=False,
        )

    latent_all = latent_dict["action"][0].detach().cpu().numpy()
    direct_rel = direct_dict["action"][0].detach().cpu().numpy()
    direct_abs = relative_actions_to_absolute_actions(direct_rel.copy(), base)

    action_steps = np.arange(
        dataset_obs_steps,
        latent_all.shape[0] + dataset_obs_steps,
        dtype=np.float32,
    )[:, None]
    base_repeat = base[None, :].repeat(latent_all.shape[0], axis=0)
    runner_latent_rows = np.concatenate([latent_all, base_repeat, action_steps], axis=-1)
    row = runner_latent_rows[args.latency_step]
    extended_obs_last_step = int(row[-1])
    latent = row[: -1 - base.shape[0]]

    ext_step = {}
    for key, value in ext.items():
        n = min(extended_obs_last_step, value.shape[0])
        chunk = value[:n]
        if n < extended_obs_last_step:
            pad = np.repeat(chunk[-1:], extended_obs_last_step - n, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        ext_step[key] = torch.from_numpy(chunk).unsqueeze(0).to(device)

    with torch.no_grad():
        decoded = policy.predict_from_latent_action(
            torch.from_numpy(latent.astype(np.float32)).unsqueeze(0).to(device),
            ext_step,
            extended_obs_last_step=extended_obs_last_step,
            dataset_obs_temporal_downsample_ratio=dataset_ratio,
        )["action"][0].detach().cpu().numpy()

    decoded_abs = relative_actions_to_absolute_actions(decoded.copy(), base)
    gt_abs = np.asarray(rb["data/action"][start:start + horizon], dtype=np.float32)
    gt_rel = absolute_actions_to_relative_actions(gt_abs.copy(), base_absolute_action=base)

    with torch.no_grad():
        gt_rel_t = torch.from_numpy(gt_rel[None]).to(device)
        ngt_rel_t = policy.normalizer["action"].normalize(gt_rel_t)
        gt_latent = policy.at.encoder(policy.at.preprocess(ngt_rel_t / policy.at.act_scale))
        if policy.at.use_vq:
            gt_latent, _, _ = policy.at.quant_state_with_vq(gt_latent)
        else:
            gt_latent, _ = policy.at.quant_state_without_vq(gt_latent)
            gt_latent = policy.at.postprocess_quant_state_without_vq(gt_latent)
        if policy.at.use_rnn_decoder:
            gt_temporal_cond = policy.at.get_temporal_cond(ext_torch).to(device)
            recon_norm = policy.at.get_action_from_latent_with_temporal_cond(gt_latent, gt_temporal_cond)
        else:
            recon_norm = policy.at.get_action_from_latent(gt_latent)
        recon_rel = policy.normalizer["action"].unnormalize(recon_norm)[0].detach().cpu().numpy()
    recon_abs = relative_actions_to_absolute_actions(recon_rel.copy(), base)

    np.set_printoptions(precision=4, suppress=True)
    print(f"dataset episode={args.episode} rows=[{start}, {end})")
    print(f"base xyz={np.round(base[:3], 4).tolist()}")
    print(f"extended_obs_last_step={extended_obs_last_step} latency_step={args.latency_step}")
    print("gt rel xyz first12:")
    print(gt_rel[:12, :3])
    print("direct predict rel xyz first12:")
    print(direct_rel[:12, :3])
    print("AT reconstruction rel xyz first12:")
    print(recon_rel[:12, :3])
    print("AT reconstruction abs xyz first/last:", np.round(recon_abs[0, :3], 4).tolist(), np.round(recon_abs[min(len(recon_abs), 12)-1, :3], 4).tolist())
    print("AT reconstruction gripper first12:")
    print(np.round(recon_rel[:12, 9], 4))
    print("decoded-from-runner-token rel xyz:")
    print(decoded[:, :3])
    print("decoded-from-runner-token abs xyz:")
    print(decoded_abs[:, :3])
    print("runner would send decoded last abs xyz:", np.round(decoded_abs[-1, :3], 4).tolist())
    print("runner would send decoded last gripper:", float(decoded_abs[-1, 9]))
    print("direct abs xyz first/last:", np.round(direct_abs[0, :3], 4).tolist(), np.round(direct_abs[-1, :3], 4).tolist())
    print("gt abs xyz first/last:", np.round(gt_abs[0, :3], 4).tolist(), np.round(gt_abs[min(len(gt_abs), len(direct_abs))-1, :3], 4).tolist())


if __name__ == "__main__":
    main()
