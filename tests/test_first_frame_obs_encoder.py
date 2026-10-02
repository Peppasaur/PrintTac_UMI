import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from reactive_diffusion_policy.model.vision.first_frame_obs_encoder import (  # noqa: E402
    FirstFrameObsEncoder,
)


class MeanBackbone(torch.nn.Module):
    def forward(self, image):
        return image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)


def test_first_frame_rgb_is_zeroed_after_t0_but_tactile_stays_temporal():
    shape_meta = {
        "obs": {
            "left_wrist_img": {"shape": [3, 2, 2], "type": "rgb"},
            "left_gripper1_img": {"shape": [3, 2, 2], "type": "rgb"},
            "left_gripper1_marker_offset_emb": {"shape": [2], "type": "low_dim"},
            "left_robot_tcp_pose": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [1]},
    }
    encoder = FirstFrameObsEncoder(
        shape_meta=shape_meta,
        rgb_model=MeanBackbone(),
        resize_shape=None,
        random_transforms=None,
        share_rgb_model=False,
        temporal_low_dim_keys="tactile",
    )

    wrist = torch.zeros((1, 2, 3, 2, 2), dtype=torch.float32)
    wrist[:, 0] = 1.0
    wrist[:, 1] = 99.0  # Must not affect the sequence feature.
    tactile = torch.zeros_like(wrist)
    tactile[:, 0] = 2.0
    tactile[:, 1] = 3.0
    low_dim = torch.tensor([[[4.0, 5.0], [6.0, 7.0]]])
    tcp_pose = torch.tensor([[[8.0, 9.0], [10.0, 11.0]]])

    features = encoder.forward_sequence(
        {
            "left_wrist_img": wrist,
            "left_gripper1_img": tactile,
            "left_gripper1_marker_offset_emb": low_dim,
            "left_robot_tcp_pose": tcp_pose,
        }
    )

    # Feature order follows the encoder's sorted keys: gripper RGB, wrist RGB,
    # then low-dim.
    assert features.shape == (1, 2, 6)
    assert torch.allclose(features[0, :, 0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(features[0, :, 1], torch.tensor([1.0, 0.0]))
    assert torch.equal(features[0, 0, 2:4], torch.tensor([4.0, 5.0]))
    assert torch.equal(features[0, 1, 2:4], torch.tensor([6.0, 7.0]))
    assert torch.equal(features[0, 0, 4:], torch.tensor([8.0, 9.0]))
    assert torch.equal(features[0, 1, 4:], torch.zeros(2))


def test_cached_context_ignores_new_visual_and_robot_state_after_first_call():
    shape_meta = {
        "obs": {
            "left_wrist_img": {"shape": [3, 2, 2], "type": "rgb"},
            "left_gripper1_marker_offset_emb": {"shape": [1], "type": "low_dim"},
            "left_robot_tcp_pose": {"shape": [1], "type": "low_dim"},
        },
        "action": {"shape": [1]},
    }
    encoder = FirstFrameObsEncoder(
        shape_meta=shape_meta,
        rgb_model=MeanBackbone(),
        temporal_low_dim_keys="tactile",
    )
    first = encoder.forward_sequence(
        {
            "left_wrist_img": torch.ones((1, 2, 3, 2, 2)),
            "left_gripper1_marker_offset_emb": torch.tensor([[[2.0], [3.0]]]),
            "left_robot_tcp_pose": torch.tensor([[[4.0], [5.0]]]),
        },
        use_cached_first_frame=True,
    )
    second = encoder.forward_sequence(
        {
            "left_wrist_img": torch.full((1, 2, 3, 2, 2), 99.0),
            "left_gripper1_marker_offset_emb": torch.tensor([[[6.0], [7.0]]]),
            "left_robot_tcp_pose": torch.tensor([[[8.0], [9.0]]]),
        },
        use_cached_first_frame=True,
    )
    # RGB remains the cached initial context at t=0, while robot state is
    # removed entirely after the first call. Tactile values keep changing.
    assert torch.equal(first[0, :, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(second[0, :, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(second[0, :, 1], torch.tensor([6.0, 7.0]))
    assert torch.equal(second[0, :, 2], torch.zeros(2))

    encoder.reset()
    after_reset = encoder.forward_sequence(
        {
            "left_wrist_img": torch.full((1, 2, 3, 2, 2), 7.0),
            "left_gripper1_marker_offset_emb": torch.zeros((1, 2, 1)),
            "left_robot_tcp_pose": torch.zeros((1, 2, 1)),
        },
        use_cached_first_frame=True,
    )
    assert torch.equal(after_reset[0, 0, 0], torch.tensor(7.0))
