# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP
from rsl_rl.utils import resolve_nn_activation


class AttentionMapEncoder(nn.Module):
    """AME-style privileged terrain encoder with proprioception-conditioned attention."""

    def __init__(
        self,
        map_scan_shape: tuple[int, int] | tuple[int, int, int],
        proprio_dim: int,
        output_dim: int,
        *,
        d_model: int = 64,
        num_heads: int = 16,
        activation: str = "elu",
        use_layer_norm: bool = False,
        map_resolution: float = 0.1,
        map_x_range: tuple[float, float] | None = None,
        map_y_range: tuple[float, float] | None = None,
        use_xyz_cnn_input: bool = True,
        cnn_downsample: bool = True,
        attach_global_context: bool = False,
    ) -> None:
        super().__init__()
        if len(map_scan_shape) == 2:
            length, width = map_scan_shape
            coord_dim = 1
        elif len(map_scan_shape) == 3:
            length, width, coord_dim = map_scan_shape
        else:
            raise ValueError(f"map_scan_shape must be (L, W) or (L, W, C), got {map_scan_shape}")
        if length <= 0 or width <= 0:
            raise ValueError(f"map scan dimensions must be positive, got {map_scan_shape}")
        if coord_dim not in (1, 3):
            raise ValueError(f"map scan channel dimension must be 1 or 3, got {coord_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not use_xyz_cnn_input and d_model <= 3:
            raise ValueError(f"d_model must be greater than 3, got {d_model}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if proprio_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {proprio_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.map_scan_shape = (length, width, coord_dim)
        self.map_scan_dim = length * width * coord_dim
        self.proprio_dim = proprio_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_xyz_cnn_input = use_xyz_cnn_input
        self.cnn_downsample = cnn_downsample
        self.attach_global_context = attach_global_context

        cnn_input_channels = 3 if use_xyz_cnn_input else 1
        cnn_output_channels = d_model if use_xyz_cnn_input else d_model - 3
        first_stride = 2 if cnn_downsample else 1
        self.conv1 = nn.Conv2d(cnn_input_channels, 16, kernel_size=5, padding=2, stride=first_stride)
        self.conv2 = nn.Conv2d(
            16,
            cnn_output_channels,
            kernel_size=3 if cnn_downsample else 5,
            padding=1 if cnn_downsample else 2,
        )
        self.activation = resolve_nn_activation(activation)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        if attach_global_context:
            self.global_encoder = MLP(d_model, d_model, [256, 128], activation)
            self.query_projector = nn.Linear(d_model * 2, d_model)
            encoder_feature_dim = d_model * 2
        else:
            self.global_encoder = None
            self.query_projector = None
            encoder_feature_dim = d_model
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.latent_proj = nn.Identity() if encoder_feature_dim == output_dim else nn.Linear(encoder_feature_dim, output_dim)
        self.last_attention_weights: torch.Tensor | None = None
        # The point sequence presented to the attention layer.  Keep this alongside
        # the weights so play-time viewers can render each weight at its true scan
        # location, including the CNN downsampling layout.
        self.last_attention_points: torch.Tensor | None = None

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.point_ln = nn.LayerNorm(d_model)
            self.query_ln = nn.LayerNorm(d_model)
            self.out_ln = nn.LayerNorm(encoder_feature_dim)
        else:
            self.point_ln = nn.Identity()
            self.query_ln = nn.Identity()
            self.out_ln = nn.Identity()

        grid_xy = self._make_grid_xy(length, width, map_resolution, map_x_range, map_y_range)
        self.register_buffer("_grid_xy", grid_xy, persistent=False)

    def forward(
        self,
        map_scan: torch.Tensor,
        proprio: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if map_scan.shape[-1] != self.map_scan_dim:
            raise ValueError(f"expected map scan last dim {self.map_scan_dim}, got {map_scan.shape[-1]}")
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"expected proprio last dim {self.proprio_dim}, got {proprio.shape[-1]}")
        if map_scan.shape[:-1] != proprio.shape[:-1]:
            raise ValueError(
                "map_scan and proprio must share leading dimensions, "
                f"got {tuple(map_scan.shape[:-1])} and {tuple(proprio.shape[:-1])}"
            )

        leading_shape = map_scan.shape[:-1]
        flat_map = map_scan.reshape(-1, self.map_scan_dim)
        flat_proprio = proprio.reshape(-1, self.proprio_dim)
        encoded = self._encode_flat(flat_map, flat_proprio, return_attention=return_attention)
        return encoded.reshape(*leading_shape, self.output_dim)

    def _encode_flat(
        self,
        map_scan: torch.Tensor,
        proprio: torch.Tensor,
        *,
        return_attention: bool,
    ) -> torch.Tensor:
        batch_size = map_scan.shape[0]
        length, width, coord_dim = self.map_scan_shape
        scan = map_scan.reshape(batch_size, length, width, coord_dim)
        if coord_dim == 1:
            z = scan[..., 0]
            xy = self._grid_xy.to(device=scan.device, dtype=scan.dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
            xyz = torch.cat((xy, z.unsqueeze(-1)), dim=-1)
        else:
            xyz = scan
            z = scan[..., 2]

        cnn_input = xyz.permute(0, 3, 1, 2) if self.use_xyz_cnn_input else z.unsqueeze(1)
        features = self.activation(self.conv1(cnn_input))
        features = self.activation(self.conv2(features))
        features = features.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        attention_xyz = xyz[:, ::2, ::2, :] if self.cnn_downsample else xyz
        if self.use_xyz_cnn_input:
            point_features = features
        else:
            point_features = torch.cat(
                (features, attention_xyz.reshape(batch_size, features.shape[1], 3)), dim=-1
            )
        point_features = self.point_ln(point_features)

        query = self.proprio_proj(proprio)
        global_features_max = None
        if self.attach_global_context:
            assert self.global_encoder is not None
            assert self.query_projector is not None
            global_features = self.global_encoder(point_features)
            global_features_max = torch.max(global_features, dim=1).values
            query = self.query_projector(torch.cat((global_features_max, query), dim=-1))
        query = self.query_ln(query.unsqueeze(1))
        z_map, attention_weights = self.mha(
            query=query,
            key=point_features,
            value=point_features,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        if return_attention:
            assert attention_weights is not None
            self.last_attention_weights = attention_weights.detach()
            self.last_attention_points = attention_xyz.reshape(batch_size, -1, 3).detach()
        else:
            self.last_attention_weights = None
            self.last_attention_points = None
        z_map = z_map.squeeze(1)
        if global_features_max is not None:
            z_map = torch.cat((global_features_max, z_map), dim=-1)
        z_map = self.out_ln(z_map)
        return self.latent_proj(z_map)

    @staticmethod
    def _make_grid_xy(
        length: int,
        width: int,
        resolution: float,
        x_range: tuple[float, float] | None,
        y_range: tuple[float, float] | None,
    ) -> torch.Tensor:
        if x_range is None:
            half_x = 0.5 * (length - 1) * resolution
            x_range = (-half_x, half_x)
        if y_range is None:
            half_y = 0.5 * (width - 1) * resolution
            y_range = (-half_y, half_y)
        xs = torch.linspace(float(x_range[0]), float(x_range[1]), length)
        ys = torch.linspace(float(y_range[0]), float(y_range[1]), width)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)
