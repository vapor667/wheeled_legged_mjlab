# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import MLP


class DepthHeightEstimator(nn.Module):
    """Estimate a terrain-height latent and reconstruction from proprio history and depth."""

    def __init__(
        self,
        proprio_history_dim: int,
        depth_shape: tuple[int, int, int],
        height_dim: int,
        height_latent_dim: int = 32,
        proprio_feature_dim: int = 64,
        depth_feature_dim: int = 64,
        gru_hidden_dim: int = 128,
        proprio_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 512),
        activation: str = "elu",
        normalize_latent: bool = True,
    ) -> None:
        super().__init__()
        if proprio_history_dim <= 0:
            raise ValueError(f"proprio_history_dim must be positive, got {proprio_history_dim}")
        if len(depth_shape) != 3:
            raise ValueError(f"depth_shape must be (channels, height, width), got {depth_shape}")
        if any(dim <= 0 for dim in depth_shape):
            raise ValueError(f"depth_shape dimensions must be positive, got {depth_shape}")
        if height_dim <= 0:
            raise ValueError(f"height_dim must be positive, got {height_dim}")

        self.proprio_history_dim = proprio_history_dim
        self.depth_shape = tuple(depth_shape)
        self.height_dim = height_dim
        self.height_latent_dim = height_latent_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.normalize_latent = normalize_latent

        self.proprio_encoder = MLP(
            proprio_history_dim,
            proprio_feature_dim,
            proprio_hidden_dims,
            activation,
        )
        self.depth_encoder = _DepthCNN(
            input_shape=self.depth_shape,
            channels=depth_channels,
            output_dim=depth_feature_dim,
            activation=activation,
        )
        self.gru = nn.GRUCell(proprio_feature_dim + depth_feature_dim, gru_hidden_dim)
        self.latent_head = nn.Linear(gru_hidden_dim, height_latent_dim)
        self.height_decoder = MLP(
            height_latent_dim,
            height_dim,
            decoder_hidden_dims,
            activation,
        )

    def forward(
        self,
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(height_latent, height_hat, next_hidden_state)``."""
        self._check_inputs(proprio_history, depth, hidden_state)
        if hidden_state is None:
            hidden_state = torch.zeros(
                proprio_history.shape[0],
                self.gru_hidden_dim,
                device=proprio_history.device,
                dtype=proprio_history.dtype,
            )

        proprio_feature = self.proprio_encoder(proprio_history)
        depth_feature = self.depth_encoder(depth)
        next_hidden_state = self.gru(torch.cat((proprio_feature, depth_feature), dim=-1), hidden_state)
        height_latent = self.latent_head(next_hidden_state)
        if self.normalize_latent:
            height_latent = F.normalize(height_latent, p=2.0, dim=-1)
        height_hat = self.height_decoder(height_latent)
        return height_latent, height_hat, next_hidden_state

    def get_initial_state(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru_hidden_dim, device=device, dtype=dtype)

    def forward_sequence(
        self,
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        dones: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll a continuous chunk and reset recurrent state after terminal steps."""
        if proprio_history.ndim != 3:
            raise ValueError(
                "proprio_history sequence must have shape [time, batch, proprio_history_dim], "
                f"got {tuple(proprio_history.shape)}"
            )
        if depth.ndim != 5:
            raise ValueError(
                "depth sequence must have shape [time, batch, channels, height, width], "
                f"got {tuple(depth.shape)}"
            )
        if tuple(dones.shape[:2]) != tuple(proprio_history.shape[:2]):
            raise ValueError(
                "dones sequence must share [time, batch] with proprio_history, "
                f"got {tuple(dones.shape)}"
            )

        batch_size = proprio_history.shape[1]
        if hidden_state is None:
            hidden_state = self.get_initial_state(
                batch_size,
                device=proprio_history.device,
                dtype=proprio_history.dtype,
            )

        latent_sequence = []
        height_sequence = []
        for step in range(proprio_history.shape[0]):
            latent, height_hat, hidden_state = self(
                proprio_history[step],
                depth[step],
                hidden_state,
            )
            latent_sequence.append(latent)
            height_sequence.append(height_hat)
            done_mask = dones[step].to(device=hidden_state.device, dtype=torch.bool).view(-1, 1)
            hidden_state = torch.where(done_mask, torch.zeros_like(hidden_state), hidden_state)

        return (
            torch.stack(latent_sequence),
            torch.stack(height_sequence),
            hidden_state,
        )

    def _check_inputs(
        self,
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> None:
        if proprio_history.ndim != 2:
            raise ValueError(
                "proprio_history must have shape [batch, proprio_history_dim], "
                f"got {tuple(proprio_history.shape)}"
            )
        if proprio_history.shape[1] != self.proprio_history_dim:
            raise ValueError(
                f"expected proprio_history dim {self.proprio_history_dim}, got {proprio_history.shape[1]}"
            )
        expected_depth_shape = (proprio_history.shape[0], *self.depth_shape)
        if tuple(depth.shape) != expected_depth_shape:
            raise ValueError(f"expected depth shape {expected_depth_shape}, got {tuple(depth.shape)}")
        if hidden_state is not None:
            expected_hidden_shape = (proprio_history.shape[0], self.gru_hidden_dim)
            if tuple(hidden_state.shape) != expected_hidden_shape:
                raise ValueError(
                    f"expected hidden_state shape {expected_hidden_shape}, got {tuple(hidden_state.shape)}"
                )


class _DepthCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("depth_channels must contain at least one output channel")

        activation_cls = _resolve_activation(activation)
        in_channels = input_shape[0]
        layers: list[nn.Module] = []
        for idx, out_channels in enumerate(channels):
            kernel_size = 5 if idx == 0 else 3
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2))
            layers.append(activation_cls())
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = int(self.cnn(dummy).flatten(start_dim=1).shape[1])

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, output_dim),
            activation_cls(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.projection(self.cnn(depth))


def _resolve_activation(name: str) -> type[nn.Module]:
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation for DepthHeightEstimator: {name}")
