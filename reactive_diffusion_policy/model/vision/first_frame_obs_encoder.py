from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)


class FirstFrameObsEncoder(MultiImageObsEncoder):
    """Encode one visual frame and keep tactile/state observations temporal.

    The regular :class:`MultiImageObsEncoder` treats every RGB observation in
    every time step identically.  This encoder adds a sequence entry point for
    policies that receive a short observation window ``(B, T, ...)``:

    * RGB keys listed in ``first_frame_rgb_keys`` are encoded only at ``t=0``;
      their feature is zero for later time steps.
    * Other RGB keys (normally tactile images) are encoded at every time step.
    * Low-dimensional keys listed in ``temporal_low_dim_keys`` are passed
      through at every time step; other low-dimensional keys are first-step
      context only.  ``None`` keeps all low-dimensional keys temporal.

    ``forward`` remains the original single-frame API so that existing policy
    code and ``output_shape`` continue to work unchanged.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_transforms: Optional[list] = None,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        first_frame_rgb_keys: Optional[Sequence[str]] = None,
        temporal_low_dim_keys: Optional[Union[str, Sequence[str]]] = None,
    ):
        super().__init__(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            resize_shape=resize_shape,
            random_transforms=random_transforms,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
        )

        if first_frame_rgb_keys is None:
            # Wrist/camera streams are the visual context.  Gripper image
            # streams are tactile and remain available at every time step.
            first_frame_rgb_keys = [
                key
                for key in self.rgb_keys
                if "gripper" not in key.lower()
                and "tactile" not in key.lower()
                and "gelsight" not in key.lower()
                and "mctac" not in key.lower()
            ]
        first_frame_rgb_keys = tuple(first_frame_rgb_keys)
        unknown = sorted(set(first_frame_rgb_keys) - set(self.rgb_keys))
        if unknown:
            raise ValueError(
                "first_frame_rgb_keys must refer to RGB observations; "
                f"unknown keys: {unknown}"
            )
        self.first_frame_rgb_keys = tuple(
            key for key in self.rgb_keys if key in first_frame_rgb_keys
        )
        if temporal_low_dim_keys is None:
            temporal_low_dim_keys = self.low_dim_keys
        elif isinstance(temporal_low_dim_keys, str):
            if temporal_low_dim_keys.strip().lower() != "tactile":
                raise ValueError(
                    "temporal_low_dim_keys as a string must be 'tactile'; "
                    "otherwise pass an explicit key list"
                )
            tactile_tokens = (
                "tactile",
                "marker",
                "gelsight",
                "mctac",
                "wrench",
                "force",
            )
            temporal_low_dim_keys = [
                key
                for key in self.low_dim_keys
                if any(token in key.lower() for token in tactile_tokens)
            ]
        unknown_low_dim = sorted(
            set(temporal_low_dim_keys) - set(self.low_dim_keys)
        )
        if unknown_low_dim:
            raise ValueError(
                "temporal_low_dim_keys must refer to low_dim observations; "
                f"unknown keys: {unknown_low_dim}"
            )
        self.temporal_low_dim_keys = tuple(
            key for key in self.low_dim_keys if key in set(temporal_low_dim_keys)
        )
        self.sequence_encoder = True
        self._cached_first_frame_features = {}
        self._cached_batch_size = None
        self._initial_context_captured = False

    def reset(self):
        """Clear the cached scene context at the beginning of an episode."""
        self._cached_first_frame_features = {}
        self._cached_batch_size = None
        self._initial_context_captured = False

    def _encode_rgb(self, key: str, image: torch.Tensor) -> torch.Tensor:
        """Encode a flattened ``(N, C, H, W)`` image batch."""
        expected_shape = self.key_shape_map[key]
        if tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected {key} shape {expected_shape}, got {tuple(image.shape[1:])}"
            )
        # Match MultiImageObsEncoder: tactile images are already preprocessed
        # by the dataset and do not receive camera image transforms.
        if "gripper" not in key:
            image = self.key_transform_map[key](image)
        model = self.key_model_map["rgb"] if self.share_rgb_model else self.key_model_map[key]
        return model(image)

    def forward_sequence(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_cached_first_frame: bool = False,
    ) -> torch.Tensor:
        """Return per-step features with shape ``(B, T, feature_dim)``.

        When ``use_cached_first_frame`` is true, the first call captures the
        visual/context features and subsequent calls ignore new visual and
        non-tactile state values.  Tactile modalities continue to be encoded
        from the current observation window.
        """
        if not obs_dict:
            raise ValueError("obs_dict cannot be empty")

        first_value = next(iter(obs_dict.values()))
        if first_value.ndim < 2:
            raise ValueError(
                "FirstFrameObsEncoder.forward_sequence expects batched temporal "
                f"observations, got shape {tuple(first_value.shape)}"
            )
        batch_size, time_steps = first_value.shape[:2]
        if time_steps < 1:
            raise ValueError("Observation sequence must contain at least one frame")
        if use_cached_first_frame:
            if self._cached_batch_size is not None and self._cached_batch_size != batch_size:
                raise ValueError(
                    "Cached first-frame context was created for batch size "
                    f"{self._cached_batch_size}, got {batch_size}; call reset()"
                )
            self._cached_batch_size = batch_size
        capture_context = use_cached_first_frame and not self._initial_context_captured

        features = []
        for key in self.rgb_keys:
            image = obs_dict[key]
            expected_shape = self.key_shape_map[key]
            if tuple(image.shape[2:]) != expected_shape:
                raise ValueError(
                    f"Expected {key} temporal shape {(batch_size, time_steps) + expected_shape}, "
                    f"got {tuple(image.shape)}"
                )

            if key in self.first_frame_rgb_keys:
                if use_cached_first_frame and not capture_context:
                    first_feature = self._cached_first_frame_features[key]
                else:
                    first_feature = self._encode_rgb(key, image[:, 0])
                    if use_cached_first_frame:
                        self._cached_first_frame_features[key] = first_feature.detach().clone()
                feature_shape = first_feature.shape[1:]
                temporal_feature = torch.zeros(
                    (batch_size, time_steps) + feature_shape,
                    dtype=first_feature.dtype,
                    device=first_feature.device,
                )
                temporal_feature[:, 0] = first_feature
            else:
                flat_image = image.reshape(batch_size * time_steps, *expected_shape)
                encoded = self._encode_rgb(key, flat_image)
                temporal_feature = encoded.reshape(
                    batch_size, time_steps, *encoded.shape[1:]
                )
            features.append(temporal_feature)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            expected_shape = self.key_shape_map[key]
            if tuple(data.shape[2:]) != expected_shape:
                raise ValueError(
                    f"Expected {key} temporal shape {(batch_size, time_steps) + expected_shape}, "
                    f"got {tuple(data.shape)}"
                )
            temporal_data = data.reshape(batch_size, time_steps, -1)
            if key not in self.temporal_low_dim_keys:
                # Robot state is useful to initialize the condition, but the
                # recurrent part of the window must be tactile-only.
                first_data = temporal_data[:, :1]
                temporal_data = torch.zeros_like(temporal_data)
                if not (use_cached_first_frame and not capture_context):
                    temporal_data[:, :1] = first_data
            features.append(temporal_data)

        if not features:
            raise ValueError("shape_meta must define at least one observation key")
        result = torch.cat(features, dim=-1)
        if capture_context:
            self._initial_context_captured = True
        return result
