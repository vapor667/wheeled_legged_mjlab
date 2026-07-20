# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.depth_representation_velocity_actor_critic import (
    DepthRepresentationVelocityActorCritic,
)
from rsl_rl.modules import MLP


class DepthRepresentationVelocityPredictorActorCritic(DepthRepresentationVelocityActorCritic):
    """Depth velocity model with training-only latent-and-velocity dynamics predictors."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 32,
        activation: str = "elu",
        obs_normalization: bool = False,
        normalize_latent: bool = True,
        distribution_cfg: dict | None = None,
        depth_feature_dim: int = 64,
        depth_gru_hidden_dim: int = 64,
        depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
        latent_dynamics_hidden_dims: tuple[int, ...] | list[int] = (128, 256, 256, 128),
        latent_dynamics_horizons: tuple[int, ...] | list[int] = (1,),
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            output_dim,
            hidden_dims=hidden_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            obs_normalization=obs_normalization,
            normalize_latent=normalize_latent,
            distribution_cfg=distribution_cfg,
            depth_feature_dim=depth_feature_dim,
            depth_gru_hidden_dim=depth_gru_hidden_dim,
            depth_channels=depth_channels,
        )
        self.latent_dynamics_horizons = tuple(int(horizon) for horizon in latent_dynamics_horizons)
        if not self.latent_dynamics_horizons:
            raise ValueError("latent_dynamics_horizons must not be empty.")
        if any(horizon <= 0 for horizon in self.latent_dynamics_horizons):
            raise ValueError("latent_dynamics_horizons must contain only positive integers.")
        if len(set(self.latent_dynamics_horizons)) != len(self.latent_dynamics_horizons):
            raise ValueError("latent_dynamics_horizons must not contain duplicates.")
        self.latent_dynamics_action_dim = output_dim
        self.latent_dynamics_state_dim = latent_dim + self.lin_vel_dim
        self.target_privileged_encoder = copy.deepcopy(self.privileged_encoder)
        self.target_privileged_encoder.requires_grad_(False)
        self.latent_dynamics_predictors = torch.nn.ModuleDict(
            {
                str(horizon): MLP(
                    self.latent_dynamics_state_dim + horizon * output_dim,
                    self.latent_dynamics_state_dim,
                    latent_dynamics_hidden_dims,
                    activation,
                )
                for horizon in self.latent_dynamics_horizons
            }
        )

    def ppo_parameters(self):
        """Yield teacher optimizer parameters, including the training-only predictors."""
        yield from super().ppo_parameters()
        yield from self.latent_dynamics_predictors.parameters()

    def latent_dynamics_parameters(self):
        """Yield parameters optimized by the direct multi-horizon dynamics loss."""
        yield from self.privileged_encoder.parameters()
        yield from self.latent_dynamics_predictors.parameters()

    def get_normalized_lin_vel_target(self, obs: TensorDict) -> torch.Tensor:
        """Return ground-truth linear velocity in the actor's normalized coordinates."""
        return self.lin_vel_normalizer(self.get_lin_vel_target(obs))

    def get_target_privileged_latent(self, obs: TensorDict) -> torch.Tensor:
        """Encode privileged observations with the EMA target encoder."""
        latent = self.target_privileged_encoder(self.get_privileged_obs(obs))
        return self._normalize_latent(latent)

    @torch.no_grad()
    def get_latent_dynamics_target(
        self,
        obs: TensorDict,
        use_ema_target: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached latent and normalized ground-truth velocity targets."""
        if use_ema_target:
            latent = self.get_target_privileged_latent(obs)
        else:
            latent = self.get_privileged_latent(obs)
        return latent, self.get_normalized_lin_vel_target(obs)

    @torch.no_grad()
    def update_target_privileged_encoder(self, decay: float) -> None:
        """Update the target privileged encoder with an exponential moving average."""
        for target_parameter, online_parameter in zip(
            self.target_privileged_encoder.parameters(),
            self.privileged_encoder.parameters(),
            strict=True,
        ):
            target_parameter.lerp_(online_parameter, 1.0 - decay)

    @torch.no_grad()
    def compute_privileged_latent_diagnostics(self, obs: TensorDict) -> dict[str, float]:
        """Measure online-target alignment and teacher latent collapse indicators."""
        online_latent = self.get_privileged_latent(obs)
        target_latent = self.get_target_privileged_latent(obs)
        centered_latent = online_latent - online_latent.mean(dim=0, keepdim=True)
        feature_variance = centered_latent.square().mean(dim=0).mean()
        covariance = centered_latent.transpose(0, 1) @ centered_latent
        covariance = covariance / max(online_latent.shape[0] - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0.0)
        eigenvalue_sum = eigenvalues.sum()
        if eigenvalue_sum.item() > 0.0:
            spectrum = eigenvalues / eigenvalue_sum
            effective_rank = torch.exp(-(spectrum * torch.log(spectrum.clamp_min(1.0e-12))).sum())
        else:
            effective_rank = torch.zeros((), device=online_latent.device)
        return {
            "teacher_online_ema_cosine_similarity": F.cosine_similarity(
                online_latent,
                target_latent,
                dim=-1,
            ).mean().item(),
            "teacher_latent_feature_variance": feature_variance.item(),
            "teacher_latent_covariance_effective_rank": effective_rank.item(),
            "teacher_latent_batch_mean_norm": online_latent.mean(dim=0).norm().item(),
        }

    def predict_privileged_state(
        self,
        latent: torch.Tensor,
        normalized_lin_vel: torch.Tensor,
        applied_action_block: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictor_key = str(horizon)
        if predictor_key not in self.latent_dynamics_predictors:
            raise ValueError(
                f"No latent dynamics predictor configured for horizon {horizon}; "
                f"available horizons are {self.latent_dynamics_horizons}."
            )
        expected_action_dim = horizon * self.latent_dynamics_action_dim
        if applied_action_block.shape[-1] != expected_action_dim:
            raise ValueError(
                f"Horizon {horizon} expects an action block with {expected_action_dim} features, "
                f"got shape {tuple(applied_action_block.shape)}."
            )
        predictor_input = torch.cat((latent, normalized_lin_vel, applied_action_block), dim=-1)
        prediction = self.latent_dynamics_predictors[predictor_key](predictor_input)
        predicted_latent, predicted_normalized_lin_vel = prediction.split(
            (self.latent_dim, self.lin_vel_dim),
            dim=-1,
        )
        return F.normalize(predicted_latent, p=2.0, dim=-1), predicted_normalized_lin_vel

    def compute_latent_dynamics_losses(
        self,
        obs_t: TensorDict,
        applied_action_block: torch.Tensor,
        obs_future: TensorDict,
        horizon: int = 1,
        detach_source: bool = False,
        use_ema_target: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_t = self.get_privileged_latent(obs_t)
        normalized_lin_vel_t = self.get_normalized_lin_vel_target(obs_t)
        if detach_source:
            latent_t = latent_t.detach()
        with torch.no_grad():
            latent_future, normalized_lin_vel_future = self.get_latent_dynamics_target(
                obs_future,
                use_ema_target=use_ema_target,
            )
        predicted_latent_future, predicted_normalized_lin_vel_future = self.predict_privileged_state(
            latent_t,
            normalized_lin_vel_t,
            applied_action_block,
            horizon,
        )
        representation_loss = (
            1.0
            - F.cosine_similarity(
                predicted_latent_future,
                latent_future,
                dim=-1,
            )
        ).mean()
        velocity_loss = F.smooth_l1_loss(
            predicted_normalized_lin_vel_future,
            normalized_lin_vel_future,
        )
        return representation_loss, velocity_loss

    def rollout_privileged_state(
        self,
        latent: torch.Tensor,
        normalized_lin_vel: torch.Tensor,
        applied_action_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively compose latent and velocity state predictions."""
        if "1" not in self.latent_dynamics_predictors:
            raise ValueError("Autoregressive latent rollout requires a horizon-1 predictor.")
        if applied_action_sequence.shape[-1] != self.latent_dynamics_action_dim:
            raise ValueError(
                "Each latent rollout action must have "
                f"{self.latent_dynamics_action_dim} features, got shape "
                f"{tuple(applied_action_sequence.shape)}."
            )
        latent_predictions = []
        velocity_predictions = []
        current_latent = latent
        current_normalized_lin_vel = normalized_lin_vel
        for applied_action in applied_action_sequence.unbind(dim=0):
            current_latent, current_normalized_lin_vel = self.predict_privileged_state(
                current_latent,
                current_normalized_lin_vel,
                applied_action,
                horizon=1,
            )
            latent_predictions.append(current_latent)
            velocity_predictions.append(current_normalized_lin_vel)
        if not latent_predictions:
            raise ValueError("Latent rollout action sequence must contain at least one step.")
        return torch.stack(latent_predictions, dim=0), torch.stack(velocity_predictions, dim=0)
