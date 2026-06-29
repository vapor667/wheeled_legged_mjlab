# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.height_representation_pair import HeightRepresentationPair
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class VisualRepresentationActorCritic(nn.Module):
    """Visual-CTS actor-critic with mixed teacher/student latent paths."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 32,
        height_latent_dim: int = 32,
        height_scan_start: int | None = 49,
        height_dim: int = 121,
        activation: str = "elu",
        obs_normalization: bool = False,
        normalize_latent: bool = True,
        distribution_cfg: dict | None = None,
        height_teacher_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        height_proprio_feature_dim: int = 64,
        height_depth_feature_dim: int = 64,
        height_gru_hidden_dim: int = 128,
        height_proprio_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        height_depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
        height_decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 256),
        privileged_decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 256),
    ) -> None:
        super().__init__()
        self.actor_obs_groups, self.actor_obs_dim = self._get_obs_dim(obs, obs_groups, "actor")
        self.critic_obs_groups, self.critic_obs_dim = self._get_obs_dim(obs, obs_groups, "critic")
        self.proprio_encoder_obs_groups, self.proprio_encoder_obs_dim = self._get_obs_dim(
            obs, obs_groups, "proprio_encoder"
        )
        self.privileged_encoder_obs_groups, self.privileged_encoder_obs_dim = self._get_obs_dim(
            obs, obs_groups, "privileged_encoder"
        )
        self.depth_obs_group, self.depth_shape = self._get_depth_group_and_shape(obs, obs_groups, "depth_encoder")
        if height_dim <= 0:
            raise ValueError(f"height_dim must be positive, got {height_dim}")
        self.height_obs_groups: list[str] | None = None
        if "height_encoder" in obs_groups:
            self.height_obs_groups, height_obs_dim = self._get_obs_dim(obs, obs_groups, "height_encoder")
            if height_obs_dim != height_dim:
                raise ValueError(f"height observation dim {height_obs_dim} does not match height_dim {height_dim}")
        else:
            if height_scan_start is None or height_scan_start < 0:
                raise ValueError("height_scan_start must be non-negative when no height_encoder group is configured")
            if height_scan_start + height_dim > self.critic_obs_dim:
                raise ValueError(
                    f"height slice [{height_scan_start}:{height_scan_start + height_dim}] exceeds critic dim "
                    f"{self.critic_obs_dim}"
                )

        self.obs_groups = self.actor_obs_groups
        self.obs_dim = self.actor_obs_dim
        self.latent_dim = latent_dim
        self.height_latent_dim = height_latent_dim
        self.height_scan_start = height_scan_start
        self.height_dim = height_dim
        self.normalize_latent = normalize_latent

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.actor_obs_dim)
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim)
            self.proprio_obs_normalizer = EmpiricalNormalization(self.proprio_encoder_obs_dim)
            self.privileged_obs_normalizer = EmpiricalNormalization(self.privileged_encoder_obs_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
            self.proprio_obs_normalizer = nn.Identity()
            self.privileged_obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        encoder_hidden_dims = hidden_dims if encoder_hidden_dims is None else encoder_hidden_dims
        self.privileged_encoder = MLP(self.privileged_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.proprio_encoder = MLP(self.proprio_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.privileged_decoder = MLP(
            latent_dim,
            self.privileged_encoder_obs_dim,
            privileged_decoder_hidden_dims,
            activation,
        )
        combined_latent_dim = latent_dim + height_latent_dim
        self.height_pair = HeightRepresentationPair(
            height_dim=height_dim,
            proprio_history_dim=self.proprio_encoder_obs_dim,
            depth_shape=self.depth_shape,
            height_latent_dim=height_latent_dim,
            teacher_hidden_dims=height_teacher_hidden_dims,
            proprio_feature_dim=height_proprio_feature_dim,
            depth_feature_dim=height_depth_feature_dim,
            gru_hidden_dim=height_gru_hidden_dim,
            proprio_hidden_dims=height_proprio_hidden_dims,
            depth_channels=height_depth_channels,
            decoder_hidden_dims=height_decoder_hidden_dims,
            activation=activation,
            normalize_latent=normalize_latent,
        )
        self.actor_head = MLP(self.actor_obs_dim + combined_latent_dim, actor_output_dim, hidden_dims, activation)
        self.critic_head = MLP(self.critic_obs_dim + combined_latent_dim, 1, hidden_dims, activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_head)
        self._student_hidden_state: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run the deployable student policy path."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        actor_obs = self.get_actor_obs(obs)
        active_hidden_state = self._student_hidden_state if hidden_state is None else hidden_state
        latent, next_hidden_state = self._get_student_latent_and_hidden(obs, active_hidden_state)
        self._student_hidden_state = next_hidden_state.detach()
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def act_teacher(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run the privileged teacher path with teacher height latent."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        actor_obs = self.get_actor_obs(obs)
        latent = self.get_teacher_latent(obs, hidden_state=hidden_state)
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def act_mixed(
        self,
        obs: TensorDict,
        teacher_mask: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        update_hidden_state: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        actor_obs = self.get_actor_obs(obs)
        student_latent, next_hidden_state = self._get_student_latent_and_hidden(obs, hidden_state)
        teacher_latent = self.get_teacher_latent(obs)
        teacher_mask = self._validate_teacher_mask(teacher_mask, student_latent.shape[0])
        latent = torch.where(teacher_mask, teacher_latent, student_latent.detach())
        if update_hidden_state:
            self._student_hidden_state = next_hidden_state.detach()
        return self._actor(actor_obs, latent, stochastic_output=stochastic_output)

    def evaluate_teacher(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        critic_obs = self.get_critic_obs(obs)
        latent = self.get_teacher_latent(obs, hidden_state=hidden_state)
        return self.critic_head(torch.cat((critic_obs, latent), dim=-1))

    def evaluate_mixed(
        self,
        obs: TensorDict,
        teacher_mask: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        critic_obs = self.get_critic_obs(obs)
        student_latent, _ = self._get_student_latent_and_hidden(obs, hidden_state)
        teacher_latent = self.get_teacher_latent(obs)
        teacher_mask = self._validate_teacher_mask(teacher_mask, student_latent.shape[0])
        latent = torch.where(teacher_mask, teacher_latent, student_latent.detach())
        return self.critic_head(torch.cat((critic_obs, latent), dim=-1))

    def compute_visual_height_loss(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> dict[str, torch.Tensor]:
        height_scan = self.get_height_scan(obs)
        output = self.height_pair(
            height_scan,
            self.get_proprio_obs(obs),
            self.get_depth_obs(obs),
            hidden_state,
        )
        return self.height_pair.compute_height_loss(output, height_scan)

    def compute_visual_representation_loss(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> dict[str, torch.Tensor]:
        proprio_latent, privileged_hat = self.get_student_privileged_output(obs)
        privileged_latent = self.get_privileged_latent(obs).detach()
        privileged_latent_loss = F.mse_loss(proprio_latent, privileged_latent)
        privileged_reconstruction_loss = F.mse_loss(
            privileged_hat,
            self.get_privileged_obs(obs).detach(),
        )
        privileged_total = privileged_latent_loss + privileged_reconstruction_loss
        height_losses = self.compute_visual_height_loss(obs, hidden_state=hidden_state)
        representation_total = privileged_total + height_losses["height_total"]
        return {
            "privileged_latent": privileged_latent_loss,
            "privileged_reconstruction": privileged_reconstruction_loss,
            "privileged_total": privileged_total,
            **height_losses,
            "representation_total": representation_total,
        }

    def compute_representation_losses(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> dict[str, torch.Tensor]:
        return self.compute_visual_representation_loss(obs, hidden_state=hidden_state)

    def compute_representation_loss(self, obs: TensorDict, hidden_state: HiddenState = None) -> torch.Tensor:
        return self.compute_representation_losses(obs, hidden_state=hidden_state)["representation_total"]

    def ppo_parameters(self):
        yield from self.privileged_encoder.parameters()
        yield from self.height_pair.teacher_height_encoder.parameters()
        yield from self.actor_head.parameters()
        yield from self.critic_head.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def proprio_parameters(self):
        yield from self.proprio_encoder.parameters()
        yield from self.privileged_decoder.parameters()

    def height_parameters(self):
        yield from self.height_pair.student_height_estimator.parameters()

    def representation_parameters(self):
        yield from self.proprio_parameters()
        yield from self.height_parameters()

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_obs_normalizer(self._cat_obs(obs, self.actor_obs_groups))

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_obs_normalizer(self._cat_obs(obs, self.critic_obs_groups))

    def get_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.proprio_obs_normalizer(self._cat_obs(obs, self.proprio_encoder_obs_groups))

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.privileged_obs_normalizer(self._cat_obs(obs, self.privileged_encoder_obs_groups))

    def get_depth_obs(self, obs: TensorDict) -> torch.Tensor:
        if self.depth_obs_group not in obs:
            raise ValueError(f"missing depth observation group '{self.depth_obs_group}'")
        depth = obs[self.depth_obs_group]
        expected_shape = (obs.batch_size[0], *self.depth_shape)
        if tuple(depth.shape) != expected_shape:
            raise ValueError(f"expected depth shape {expected_shape}, got {tuple(depth.shape)}")
        return depth

    def get_height_scan(self, obs: TensorDict) -> torch.Tensor:
        if self.height_obs_groups is not None:
            return self._cat_obs(obs, self.height_obs_groups)
        assert self.height_scan_start is not None
        critic_raw = self._cat_obs(obs, self.critic_obs_groups)
        return critic_raw[:, self.height_scan_start : self.height_scan_start + self.height_dim]

    def get_proprio_latent(self, obs: TensorDict) -> torch.Tensor:
        return self._normalize_latent(self.proprio_encoder(self.get_proprio_obs(obs)))

    def get_student_privileged_output(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.get_proprio_latent(obs)
        return latent, self.privileged_decoder(latent)

    def get_privileged_latent(self, obs: TensorDict) -> torch.Tensor:
        return self._normalize_latent(self.privileged_encoder(self.get_privileged_obs(obs)))

    def get_student_height_latent(self, obs: TensorDict, hidden_state: HiddenState = None) -> torch.Tensor:
        height_latent, _, _ = self.height_pair.student_height_estimator(
            self.get_proprio_obs(obs),
            self.get_depth_obs(obs),
            hidden_state,
        )
        return height_latent

    def get_teacher_height_latent(self, obs: TensorDict, hidden_state: HiddenState = None) -> torch.Tensor:
        del hidden_state
        return self.height_pair.encode_teacher(self.get_height_scan(obs))

    def get_student_latent(self, obs: TensorDict, hidden_state: HiddenState = None) -> torch.Tensor:
        latent, _ = self._get_student_latent_and_hidden(obs, hidden_state)
        return latent

    def get_teacher_latent(self, obs: TensorDict, hidden_state: HiddenState = None) -> torch.Tensor:
        return torch.cat(
            (
                self.get_privileged_latent(obs),
                self.get_teacher_height_latent(obs, hidden_state=hidden_state),
            ),
            dim=-1,
        )

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self._student_hidden_state = None if hidden_state is None else hidden_state.detach()
        elif hidden_state is not None:
            raise NotImplementedError("Resetting done environments with a custom hidden state is not supported")
        elif self._student_hidden_state is not None:
            done_mask = dones.to(device=self._student_hidden_state.device, dtype=torch.bool).view(-1)
            self._student_hidden_state[done_mask] = 0.0

    def get_hidden_state(self) -> HiddenState:
        return self._student_hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._student_hidden_state is not None:
            if dones is None:
                self._student_hidden_state = self._student_hidden_state.detach()
            else:
                done_mask = dones.to(device=self._student_hidden_state.device, dtype=torch.bool).view(-1)
                self._student_hidden_state[done_mask] = self._student_hidden_state[done_mask].detach()

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a minimal ONNX wrapper for state-explicit student inference."""
        return _OnnxVisualStudentPolicy(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.actor_obs_normalizer.update(self._cat_obs(obs, self.actor_obs_groups))  # type: ignore
            self.critic_obs_normalizer.update(self._cat_obs(obs, self.critic_obs_groups))  # type: ignore
            self.proprio_obs_normalizer.update(self._cat_obs(obs, self.proprio_encoder_obs_groups))  # type: ignore
            self.privileged_obs_normalizer.update(self._cat_obs(obs, self.privileged_encoder_obs_groups))  # type: ignore

    def _actor(self, actor_obs: torch.Tensor, latent: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        mlp_output = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent, p=2.0, dim=-1) if self.normalize_latent else latent

    def _get_student_latent_and_hidden(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("Visual height estimator expects a tensor GRU hidden state")
        height_latent, _, next_hidden_state = self.height_pair.student_height_estimator(
            self.get_proprio_obs(obs),
            self.get_depth_obs(obs),
            hidden_state,
        )
        latent = torch.cat(
            (self.get_proprio_latent(obs), height_latent),
            dim=-1,
        )
        return latent, next_hidden_state

    @staticmethod
    def _validate_teacher_mask(teacher_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if teacher_mask.ndim == 1:
            teacher_mask = teacher_mask.unsqueeze(-1)
        if tuple(teacher_mask.shape) != (batch_size, 1):
            raise ValueError(f"teacher_mask must have shape [{batch_size}] or [{batch_size}, 1]")
        return teacher_mask.to(dtype=torch.bool)

    def _cat_obs(self, obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
        missing = [obs_group for obs_group in obs_groups if obs_group not in obs]
        if missing:
            raise ValueError(f"missing observation group(s): {missing}")
        return torch.cat([obs[obs_group] for obs_group in obs_groups], dim=-1)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The visual representation model expects 1D observations for '{obs_set}', "
                    f"got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_depth_group_and_shape(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[str, tuple[int, int, int]]:
        active_obs_groups = obs_groups[obs_set]
        if len(active_obs_groups) != 1:
            raise ValueError(f"'{obs_set}' must contain exactly one depth observation group, got {active_obs_groups}")
        obs_group = active_obs_groups[0]
        if len(obs[obs_group].shape) != 4:
            raise ValueError(f"Depth observation '{obs_group}' must have shape [batch, C, H, W], got {obs[obs_group].shape}")
        return obs_group, tuple(obs[obs_group].shape[1:])


class _OnnxVisualStudentPolicy(nn.Module):
    """Deployment-only Visual student graph with explicit GRU state I/O."""

    is_recurrent: bool = True

    def __init__(self, model: VisualRepresentationActorCritic, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.use_external_data = False
        self.actor_obs_normalizer = copy.deepcopy(model.actor_obs_normalizer)
        self.proprio_obs_normalizer = copy.deepcopy(model.proprio_obs_normalizer)
        self.privileged_estimator_encoder = copy.deepcopy(model.proprio_encoder)

        estimator = model.height_pair.student_height_estimator
        self.height_proprio_encoder = copy.deepcopy(estimator.proprio_encoder)
        self.depth_encoder = copy.deepcopy(estimator.depth_encoder)
        self.gru = copy.deepcopy(estimator.gru)
        self.height_latent_head = copy.deepcopy(estimator.latent_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.actor_input_size = model.actor_obs_dim
        self.proprio_input_size = model.proprio_encoder_obs_dim
        self.depth_input_shape = model.depth_shape
        self.hidden_size = estimator.gru_hidden_dim

    def forward(
        self,
        actor_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        hidden_state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_obs = self.actor_obs_normalizer(actor_obs)
        proprio_history = self.proprio_obs_normalizer(proprio_history)

        privileged_latent = self.privileged_estimator_encoder(proprio_history)
        height_proprio_feature = self.height_proprio_encoder(proprio_history)
        depth_feature = self.depth_encoder(depth)
        hidden_state_out = self.gru(
            torch.cat((height_proprio_feature, depth_feature), dim=-1),
            hidden_state_in,
        )
        height_latent = self.height_latent_head(hidden_state_out)
        if self.normalize_latent:
            privileged_latent = F.normalize(privileged_latent, p=2.0, dim=-1)
            height_latent = F.normalize(height_latent, p=2.0, dim=-1)

        actor_input = torch.cat((actor_obs, privileged_latent, height_latent), dim=-1)
        actions = self.deterministic_output(self.actor_head(actor_input))
        return actions, hidden_state_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.actor_input_size),
            torch.zeros(1, self.proprio_input_size),
            torch.zeros(1, *self.depth_input_shape),
            torch.zeros(1, self.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["actor_obs", "proprio_history", "depth", "hidden_state_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "hidden_state_out"]
