from pathlib import Path

import numpy as np

from reactive_diffusion_policy.env.franka_polymetis.franka_polymetis_env import (
    FrankaPolymetisEnv,
)


def _make_env(bottom_rows):
    env = FrankaPolymetisEnv.__new__(FrankaPolymetisEnv)
    env.camera_backend = "iphone"
    env.camera_preprocess_mode = "square_crop"
    env.camera_square_crop_bottom_rows = bottom_rows
    env.camera_flip = False
    env.image_shape = (3, 224, 224)
    env.camera_color_match_dataset_start = False
    return env


def test_square_crop_can_remove_bottom_rows_before_resizing():
    row_values = np.arange(224, dtype=np.uint8)[:, None, None]
    frame = np.broadcast_to(row_values, (224, 224, 3)).copy()

    unchanged = _make_env(0)._preprocess_bgr(frame)
    cropped = _make_env(24)._preprocess_bgr(frame)

    assert unchanged.shape == (224, 224, 3)
    assert cropped.shape == (224, 224, 3)
    assert float(unchanged[-1].mean()) > 220
    assert float(cropped[-1].mean()) < 202


def test_eval_scripts_forward_square_crop_bottom_rows():
    repo = Path(__file__).parents[1]
    for script_name in ("eval.sh", "eval_test.sh"):
        script = (repo / script_name).read_text()
        assert 'CAMERA_SQUARE_CROP_BOTTOM_ROWS="${CAMERA_SQUARE_CROP_BOTTOM_ROWS:-0}"' in script
        assert "camera_square_crop_bottom_rows=${CAMERA_SQUARE_CROP_BOTTOM_ROWS}" in script
