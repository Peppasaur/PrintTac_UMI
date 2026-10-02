#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np


def latest_recording(path):
    if path:
        return path
    candidates = sorted(
        glob.glob(
            "data/eval_outputs/franka_polymetis/policy_recordings/*_magnet_trace.npz"
        )
    )
    if not candidates:
        raise FileNotFoundError("No policy recording npz found")
    return candidates[-1]


def first_crossing(values, threshold):
    idxs = np.flatnonzero(values >= threshold)
    if idxs.size == 0:
        return None
    return int(idxs[0])


def fmt(values, precision=4):
    return np.round(np.asarray(values, dtype=np.float32), precision).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="policy recording *_magnet_trace.npz")
    parser.add_argument("--threshold", type=float, default=300.0)
    args = parser.parse_args()

    path = latest_recording(args.path)
    data = np.load(path)
    elapsed = data["elapsed"]
    tactile = data["tactile_emb"]
    normalized = data["normalized_tactile_emb"]
    raw_abs = np.nanmax(np.abs(tactile), axis=1)
    norm_abs = np.nanmax(np.abs(normalized), axis=1)

    print(f"path: {path}")
    print(f"frames: {len(elapsed)}")
    if len(elapsed) > 0:
        print(f"duration: {float(elapsed[-1]):.3f}s")
    print(f"raw_abs_max: {float(np.nanmax(raw_abs)):.3f}")
    print(f"norm_abs_max: {float(np.nanmax(norm_abs)):.3f}")

    idx = first_crossing(raw_abs, args.threshold)
    if idx is None:
        print(f"first raw_abs >= {args.threshold:g}: none")
        idx = int(np.nanargmax(raw_abs)) if len(raw_abs) else None
    else:
        print(
            f"first raw_abs >= {args.threshold:g}: "
            f"frame={idx}, t={float(elapsed[idx]):.3f}s, raw_abs={float(raw_abs[idx]):.3f}, "
            f"norm_abs={float(norm_abs[idx]):.3f}"
        )

    if idx is None:
        return

    max_idx = int(np.nanargmax(raw_abs))
    print(
        f"peak: frame={max_idx}, t={float(elapsed[max_idx]):.3f}s, "
        f"raw_abs={float(raw_abs[max_idx]):.3f}, norm_abs={float(norm_abs[max_idx]):.3f}"
    )
    print(f"raw row @ crossing: {fmt(tactile[idx], 2)}")
    print(f"norm row @ crossing: {fmt(normalized[idx], 2)}")

    if "tcp_pose_obs" in data:
        tcp = data["tcp_pose_obs"]
        base_tcp = tcp[0]
        print(f"tcp xyz @ start: {fmt(base_tcp[:3], 4)}")
        print(f"tcp xyz @ crossing: {fmt(tcp[idx, :3], 4)}")
        print(f"tcp xyz delta @ crossing: {fmt(tcp[idx, :3] - base_tcp[:3], 4)}")
        print(f"tcp 6d rot @ crossing: {fmt(tcp[idx, 3:9], 4)}")
    else:
        print("tcp_pose_obs: not recorded in this npz")

    if "gripper_obs" in data:
        gripper = data["gripper_obs"].reshape(len(elapsed), -1)
        print(f"gripper obs @ start: {fmt(gripper[0], 4)}")
        print(f"gripper obs @ crossing: {fmt(gripper[idx], 4)}")
    else:
        print("gripper_obs: not recorded in this npz")

    if "last_action_command" in data:
        action = data["last_action_command"]
        print(f"last action command @ crossing: {fmt(action[idx], 4)}")
        finite = np.isfinite(action[:, :3]).all(axis=1)
        if np.any(finite):
            first_action_idx = int(np.flatnonzero(finite)[0])
            print(f"first finite action frame: {first_action_idx}")
            print(
                "action xyz delta crossing-first: "
                f"{fmt(action[idx, :3] - action[first_action_idx, :3], 4)}"
            )
            print(
                "action gripper crossing: "
                f"{float(action[idx, 6]):.4f}"
                if action.shape[1] > 6 and np.isfinite(action[idx, 6])
                else "action gripper crossing: nan"
            )
    else:
        print("last_action_command: not recorded in this npz")


if __name__ == "__main__":
    main()
