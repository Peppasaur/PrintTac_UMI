from pathlib import Path


def test_aligned_eval_does_not_double_transform_relative_tcp_frame():
    script = (Path(__file__).parents[1] / "eval_iphone_aligned.sh").read_text()

    assert 'export POLICY_TCP_POSE_OBS_MODE="none"' in script
    assert 'export POLICY_RELATIVE_ACTION_FRAME_MODE="none"' in script


def test_aligned_eval_keeps_training_observation_stride():
    script = (Path(__file__).parents[1] / "eval_iphone_aligned.sh").read_text()

    assert 'OBS_TEMPORAL_DOWNSAMPLE_RATIO:-2' in script


def test_aligned_eval_uses_dataset_derived_gripper_mapping():
    script = (Path(__file__).parents[1] / "eval_iphone_aligned.sh").read_text()

    assert 'export GRIPPER_RAW_MIN=' not in script
    assert 'export GRIPPER_RAW_MAX=' not in script
    assert 'export GRIPPER_OBS_RAW_OFFSET=' not in script


def test_eval_script_preserves_task_ensemble_defaults():
    script = (Path(__file__).parents[1] / "eval.sh").read_text()

    assert 'TCP_ENSEMBLE_MODE="${TCP_ENSEMBLE_MODE:-}"' in script
    assert 'GRIPPER_ENSEMBLE_MODE="${GRIPPER_ENSEMBLE_MODE:-}"' in script
    assert 'task.env_runner.tcp_ensemble_buffer_params.ensemble_mode=${TCP_ENSEMBLE_MODE}' in script
