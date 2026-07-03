# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.models.depth_height_estimator import DepthHeightEstimator
from rsl_rl.modules import MLP


@dataclass
class HeightRepresentationOutput:
    teacher_height_latent: torch.Tensor
    student_height_latent: torch.Tensor
    height_hat: torch.Tensor
    next_hidden_state: torch.Tensor


class HeightRepresentationPair(nn.Module):
    """Teacher height encoder paired with a student depth-height estimator."""

    def __init__(
        self,
        height_dim: int,
        proprio_history_dim: int,
        depth_shape: tuple[int, int, int],
        height_latent_dim: int = 32,
        teacher_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
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
        if height_dim <= 0:
            raise ValueError(f"height_dim must be positive, got {height_dim}")
        self.height_dim = height_dim
        self.height_latent_dim = height_latent_dim
        self.normalize_latent = normalize_latent

        self.teacher_height_encoder = MLP(
            height_dim,
            height_latent_dim,
            teacher_hidden_dims,
            activation,
        )
        self.student_height_estimator = DepthHeightEstimator(
            proprio_history_dim=proprio_history_dim,
            depth_shape=depth_shape,
            height_dim=height_dim,
            height_latent_dim=height_latent_dim,
            proprio_feature_dim=proprio_feature_dim,
            depth_feature_dim=depth_feature_dim,
            gru_hidden_dim=gru_hidden_dim,
            proprio_hidden_dims=proprio_hidden_dims,
            depth_channels=depth_channels,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
            normalize_latent=normalize_latent,
        )

    def forward(
        self,
        height_scan: torch.Tensor,
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> HeightRepresentationOutput:
        self._check_height_scan(height_scan)
        teacher_height_latent = self.encode_teacher(height_scan)
        student_height_latent, height_hat, next_hidden_state = self.student_height_estimator(
            proprio_history,
            depth,
            hidden_state,
        )
        return HeightRepresentationOutput(
            teacher_height_latent=teacher_height_latent,
            student_height_latent=student_height_latent,
            height_hat=height_hat,
            next_hidden_state=next_hidden_state,
        )

    def encode_teacher(self, height_scan: torch.Tensor) -> torch.Tensor:
        self._check_height_scan(height_scan)
        latent = self.teacher_height_encoder(height_scan)
        return F.normalize(latent, p=2.0, dim=-1) if self.normalize_latent else latent

    def compute_height_loss(
        self,
        output: HeightRepresentationOutput,
        height_scan: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._check_height_scan(height_scan)
        height_latent_loss = F.mse_loss(
            output.student_height_latent,
            output.teacher_height_latent.detach(),
        )
        height_reconstruction_loss = F.mse_loss(output.height_hat, height_scan)
        return {
            "height_latent": height_latent_loss,
            "height_reconstruction": height_reconstruction_loss,
            "height_total": height_latent_loss + height_reconstruction_loss,
        }

    def _check_height_scan(self, height_scan: torch.Tensor) -> None:
        if height_scan.ndim != 2:
            raise ValueError(f"height_scan must have shape [batch, height_dim], got {tuple(height_scan.shape)}")
        if height_scan.shape[1] != self.height_dim:
            raise ValueError(f"expected height_scan dim {self.height_dim}, got {height_scan.shape[1]}")
