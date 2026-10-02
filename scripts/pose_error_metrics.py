#!/usr/bin/env python3
"""Pose trajectory error metrics shared by comparison and fusion tools."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def pose6_to_matrices(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 6:
        raise ValueError(f"Expected pose [T,6], got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("Pose sequence contains non-finite values")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(pose), axis=0)
    matrices[:, :3, :3] = Rotation.from_rotvec(pose[:, 3:6]).as_matrix()
    matrices[:, :3, 3] = pose[:, :3]
    return matrices


def first_frame_align(
    iphone_matrices: np.ndarray, robot_matrices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if iphone_matrices.shape != robot_matrices.shape:
        raise ValueError(
            f"Pose matrix shapes differ: {iphone_matrices.shape} vs {robot_matrices.shape}"
        )
    if len(iphone_matrices) == 0:
        raise ValueError("Cannot align empty pose sequences")
    alignment = robot_matrices[0] @ np.linalg.inv(iphone_matrices[0])
    return alignment[None] @ iphone_matrices, alignment


def rotation_geodesic_deg(
    first_rotation: np.ndarray, second_rotation: np.ndarray
) -> np.ndarray:
    relative = np.transpose(first_rotation, (0, 2, 1)) @ second_rotation
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


def compare_pose_sequences(
    iphone_pose: np.ndarray, robot_pose: np.ndarray
) -> Dict[str, np.ndarray]:
    if len(iphone_pose) != len(robot_pose):
        raise ValueError(
            f"Pose sequence lengths differ: {len(iphone_pose)} vs {len(robot_pose)}"
        )
    if len(iphone_pose) < 2:
        raise ValueError("At least two aligned poses are required")

    iphone = pose6_to_matrices(iphone_pose)
    robot = pose6_to_matrices(robot_pose)
    iphone_aligned, alignment = first_frame_align(iphone, robot)

    position_delta_mm = (iphone_aligned[:, :3, 3] - robot[:, :3, 3]) * 1000.0
    position_error_mm = np.linalg.norm(position_delta_mm, axis=1)
    rotation_error_deg = rotation_geodesic_deg(
        iphone_aligned[:, :3, :3], robot[:, :3, :3]
    )

    iphone_relative = np.linalg.inv(iphone[:-1]) @ iphone[1:]
    robot_relative = np.linalg.inv(robot[:-1]) @ robot[1:]
    relative_translation_delta_mm = (
        iphone_relative[:, :3, 3] - robot_relative[:, :3, 3]
    ) * 1000.0
    relative_translation_error_mm = np.linalg.norm(
        relative_translation_delta_mm, axis=1
    )
    relative_rotation_error_deg = rotation_geodesic_deg(
        iphone_relative[:, :3, :3], robot_relative[:, :3, :3]
    )

    return {
        "alignment_matrix": alignment,
        "position_delta_mm": position_delta_mm,
        "position_error_mm": position_error_mm,
        "rotation_error_deg": rotation_error_deg,
        "relative_translation_delta_mm": relative_translation_delta_mm,
        "relative_translation_error_mm": relative_translation_error_mm,
        "relative_rotation_error_deg": relative_rotation_error_deg,
    }


def error_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty error array")
    percentiles = np.percentile(values, [50, 90, 95, 99, 100])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "max": float(percentiles[4]),
    }


def vector_component_summary(values: np.ndarray) -> Dict[str, Dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected vector errors [T,3], got {values.shape}")
    result = {}
    for axis, axis_values in zip("xyz", values.T):
        result[axis] = {
            "signed_mean": float(np.mean(axis_values)),
            "signed_median": float(np.median(axis_values)),
            "absolute_median": float(np.median(np.abs(axis_values))),
            "absolute_p95": float(np.percentile(np.abs(axis_values), 95)),
            "absolute_max": float(np.max(np.abs(axis_values))),
        }
    return result
