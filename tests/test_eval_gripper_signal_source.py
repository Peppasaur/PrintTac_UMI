import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


def _eval_overrides(tmp_path, source=None, extra_env=None):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    for key in (
        "GRIPPER_SIGNAL_SOURCE",
        "GRIPPER_RAW_MIN",
        "GRIPPER_RAW_MAX",
        "GRIPPER_RAW_RANGE_MODE",
        "GRIPPER_RAW_CALIBRATION_SCALE",
        "GRIPPER_RAW_CALIBRATION_OFFSET",
        "GRIPPER_COMMAND_WIDTH_OFFSET",
        "GRIPPER_OBS_MODE",
    ):
        env.pop(key, None)
    env.update({
        "PATH": f"{tmp_path}:{env['PATH']}",
        "CKPT_PATH": "fixture.ckpt",
        "TASK": "franka_polymetis_image_dp_absolute_12fps",
        "DATASET_PATH": "fixture_dataset",
    })
    if source is not None:
        env["GRIPPER_SIGNAL_SOURCE"] = source
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", "eval.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_teleop_raw_source_uses_direct_open_ratio_and_commanded_observation(tmp_path):
    overrides = _eval_overrides(tmp_path, "teleop_raw")

    assert "++task.env_runner.env_params.gripper_raw_range_mode=trajectory_zero_to_max" in overrides
    assert "task.env_runner.env_params.gripper_raw_min=0.0" in overrides
    assert "task.env_runner.env_params.gripper_raw_max=1.0" in overrides
    assert "task.env_runner.env_params.gripper_obs_mode=commanded" in overrides
    assert "++task.env_runner.env_params.gripper_command_width_offset=0.000" in overrides


def test_teleop_width_source_uses_direct_meter_width_and_commanded_observation(tmp_path):
    overrides = _eval_overrides(tmp_path, "teleop_width")

    assert "++task.env_runner.env_params.gripper_raw_range_mode=calibrated_marker_width" in overrides
    assert "++task.env_runner.env_params.gripper_raw_calibration_scale=1.0" in overrides
    assert "++task.env_runner.env_params.gripper_raw_calibration_offset=0.0" in overrides
    assert "++task.env_runner.env_params.gripper_command_width_offset=0.000" in overrides
    assert "task.env_runner.env_params.gripper_obs_mode=commanded" in overrides


def test_umi_marker_source_uses_direct_calibrated_width_mapping(tmp_path):
    overrides = _eval_overrides(tmp_path, "umi_marker")

    assert "++task.env_runner.env_params.gripper_raw_range_mode=calibrated_marker_width" in overrides
    assert "++task.env_runner.env_params.gripper_raw_calibration_scale=1.0" in overrides
    assert "++task.env_runner.env_params.gripper_raw_calibration_offset=0.0" in overrides
    assert "++task.env_runner.env_params.gripper_command_width_offset=-0.010" in overrides
    assert "task.env_runner.env_params.gripper_obs_mode=auto" in overrides


def test_umi_marker_mapping_is_the_default(tmp_path):
    overrides = _eval_overrides(tmp_path)

    assert "++task.env_runner.env_params.gripper_raw_range_mode=calibrated_marker_width" in overrides
    assert "++task.env_runner.env_params.gripper_raw_calibration_scale=1.0" in overrides
    assert "++task.env_runner.env_params.gripper_raw_calibration_offset=0.0" in overrides
    assert "++task.env_runner.env_params.gripper_command_width_offset=-0.010" in overrides


def test_eval_uses_sibling_data_directory_for_calibration_and_outputs(tmp_path):
    overrides = _eval_overrides(tmp_path)

    assert "task.transforms.calibration_path=../data/calibration/v6" in overrides
    assert "task.env_runner.output_dir=../data/eval_outputs/franka_polymetis" in overrides


def test_eval_data_directory_can_be_overridden(tmp_path):
    overrides = _eval_overrides(
        tmp_path,
        extra_env={"DATA_DIR": "/mnt/rdp-data"},
    )

    assert "task.transforms.calibration_path=/mnt/rdp-data/calibration/v6" in overrides
    assert "task.env_runner.output_dir=/mnt/rdp-data/eval_outputs/franka_polymetis" in overrides


def test_fixed_gripper_width_is_forwarded_as_physical_millimeters(tmp_path):
    overrides = _eval_overrides(
        tmp_path,
        "umi_marker",
        {"FIXED_GRIPPER_WIDTH_MM": "55.0"},
    )

    assert "++task.env_runner.env_params.fixed_gripper_width_mm=55.0" in overrides


def test_start_gripper_width_is_forwarded_to_runner_in_millimeters(tmp_path):
    overrides = _eval_overrides(
        tmp_path,
        "umi_marker",
        {"START_GRIPPER_WIDTH_MM": "52.0"},
    )

    assert "++task.env_runner.start_gripper_width_mm=52.0" in overrides


def test_magnet_rezero_delay_is_forwarded_to_environment(tmp_path):
    overrides = _eval_overrides(
        tmp_path,
        "umi_marker",
        {"MAGNET_REZERO_AFTER_POLICY_START_SEC": "1.0"},
    )

    assert "++task.env_runner.env_params.magnet_rezero_after_policy_start_sec=1.0" in overrides
