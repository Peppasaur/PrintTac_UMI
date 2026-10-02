import numpy as np

from reactive_diffusion_policy.common.sampler import SequenceSampler
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import (
    RealImageTactileDataset,
)


def test_sampler_returns_absolute_episode_start_for_each_window():
    sampler = SequenceSampler.__new__(SequenceSampler)
    sampler.episode_ends = np.array([3, 7], dtype=np.int64)

    sampler.indices = np.array([[0, 2, 1, 3]], dtype=np.int64)
    assert sampler.get_episode_start(0) == 0

    sampler.indices = np.array([[5, 7, 0, 2]], dtype=np.int64)
    assert sampler.get_episode_start(0) == 3

    # The first valid row of episode 1 equals episode 0's exclusive end.
    sampler.indices = np.array([[3, 5, 0, 2]], dtype=np.int64)
    assert sampler.get_episode_start(0) == 3


class _WindowSampler:
    def __init__(self, window, episode_start):
        self.window = window
        self.episode_start = episode_start

    def sample_sequence(self, _idx):
        return {key: value.copy() for key, value in self.window.items()}

    def get_episode_start(self, _idx):
        return self.episode_start


def test_dataset_injects_episode_first_rgb_and_zeroes_later_window_images():
    episode_images = np.stack(
        [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 11, 12, 20, 21, 22)]
    )
    window = {
        "left_wrist_img": episode_images[4:6],
        "left_gripper1_marker_offset_emb": np.array([[101.0], [102.0]], dtype=np.float32),
        "action": np.array([[1.0], [2.0]], dtype=np.float32),
    }
    dataset = RealImageTactileDataset.__new__(RealImageTactileDataset)
    dataset.sampler = _WindowSampler(window, episode_start=3)
    dataset.replay_buffer = {"left_wrist_img": episode_images}
    dataset.shape_meta = {
        "obs": {
            "left_wrist_img": {"shape": [3, 2, 2], "type": "rgb"},
            "left_gripper1_marker_offset_emb": {"shape": [1], "type": "low_dim"},
        },
        "action": {"shape": [1]},
    }
    dataset.rgb_keys = ["left_wrist_img"]
    dataset.lowdim_keys = ["left_gripper1_marker_offset_emb"]
    dataset.episode_first_frame_rgb_keys = ("left_wrist_img",)
    dataset.extended_rgb_keys = []
    dataset.extended_lowdim_keys = []
    dataset.n_obs_steps = 2
    dataset.obs_downsample_ratio = 1
    dataset.n_latency_steps = 0
    dataset.relative_action = False
    dataset.relative_gripper_action = False
    dataset.zero_gripper_action = False
    dataset.transforms = None

    sample = dataset[0]

    image = sample["obs"]["left_wrist_img"].numpy()
    np.testing.assert_allclose(image[0], 20.0 / 255.0)
    np.testing.assert_array_equal(image[1], np.zeros_like(image[1]))
    np.testing.assert_array_equal(
        sample["obs"]["left_gripper1_marker_offset_emb"].numpy(),
        window["left_gripper1_marker_offset_emb"],
    )
