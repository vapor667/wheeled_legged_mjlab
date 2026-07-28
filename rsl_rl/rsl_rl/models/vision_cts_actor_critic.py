"""Vision-CTS actor-critic on the WF-TRON1B observation interface."""

# ruff: file-ignore[missing-return-type-undocumented-public-function, undocumented-public-method, undocumented-public-init, lowercase-imported-as-non-lowercase]

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _activation(name: str) -> type[nn.Module]:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation '{name}'") from exc


class _DepthEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(dim <= 0 for dim in input_shape):
            raise ValueError(f"depth input must be [channels, height, width], got {input_shape}")
        if not channels:
            raise ValueError("height_depth_channels must not be empty")
        activation_cls = _activation(activation)
        layers: list[nn.Module] = []
        in_channels = input_shape[0]
        for index, out_channels in enumerate(channels):
            layers.extend((
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=5 if index == 0 else 3,
                    stride=2 if index < len(channels) - 1 else 1,
                ),
                activation_cls(),
            ))
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            flattened_dim = int(self.cnn(torch.zeros(1, *input_shape)).flatten(start_dim=1).shape[-1])
        self.projection = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(flattened_dim, output_dim), activation_cls())

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.projection(self.cnn(depth))


class VisionCTSActorCritic(nn.Module):
    """Concurrent-teacher-student VisionCTS with the current actor/critic observations.

    The deployable actor intentionally receives only current proprioception and the
    command.  Unlike the in-house depth policy, it has no linear-velocity head.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        latent_dim: int = 32,
        height_latent_dim: int = 32,
        activation: str = "elu",
        obs_normalization: bool = False,
        normalize_latent: bool = True,
        distribution_cfg: dict | None = None,
        height_teacher_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        height_proprio_feature_dim: int = 64,
        height_depth_feature_dim: int = 64,
        height_gru_hidden_dim: int = 128,
        height_proprio_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        height_depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
        height_decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 512),
        privileged_decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 512),
    ) -> None:
        super().__init__()
        self.teacher_actor_obs_groups, self.teacher_actor_obs_dim = self._get_obs_dim(obs, obs_groups, "teacher_actor")
        self.critic_obs_groups, self.critic_obs_dim = self._get_obs_dim(obs, obs_groups, "critic")
        self.student_history_obs_groups, self.student_history_length, self.student_actor_obs_dim = (
            self._get_history_shape(obs, obs_groups, "student_history")
        )
        if self.student_actor_obs_dim != self.teacher_actor_obs_dim:
            raise ValueError(
                "VisionCTS requires actor_history frames to match teacher_actor, got "
                f"{self.student_actor_obs_dim} and {self.teacher_actor_obs_dim}"
            )
        self.privileged_encoder_obs_groups, self.privileged_encoder_obs_dim = self._get_obs_dim(
            obs, obs_groups, "privileged_encoder"
        )
        self.height_obs_groups, self.height_dim = self._get_obs_dim(obs, obs_groups, "height_encoder")
        self.depth_obs_group, self.depth_shape = self._get_depth_group_and_shape(obs, obs_groups, "depth_encoder")

        self.proprio_encoder_obs_dim = self.student_history_length * self.student_actor_obs_dim
        self.latent_dim = latent_dim
        self.height_latent_dim = height_latent_dim
        self.height_gru_hidden_dim = height_gru_hidden_dim
        self.normalize_latent = normalize_latent
        self.obs_normalization = obs_normalization

        if obs_normalization:
            self.teacher_actor_obs_normalizer = EmpiricalNormalization(self.teacher_actor_obs_dim)
            self.student_actor_obs_normalizer = EmpiricalNormalization(self.student_actor_obs_dim)
            self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim)
            self.proprio_obs_normalizer = EmpiricalNormalization(self.proprio_encoder_obs_dim)
            self.privileged_obs_normalizer = EmpiricalNormalization(self.privileged_encoder_obs_dim)
        else:
            self.teacher_actor_obs_normalizer = nn.Identity()
            self.student_actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()
            self.proprio_obs_normalizer = nn.Identity()
            self.privileged_obs_normalizer = nn.Identity()

        if distribution_cfg is None:
            self.distribution: Distribution | None = None
            actor_output_dim = output_dim
        else:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            distribution_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution = distribution_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim

        self.privileged_encoder = MLP(self.privileged_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.proprio_encoder = MLP(self.proprio_encoder_obs_dim, latent_dim, encoder_hidden_dims, activation)
        self.privileged_decoder = MLP(
            latent_dim, self.privileged_encoder_obs_dim, privileged_decoder_hidden_dims, activation
        )
        self.teacher_height_encoder = MLP(self.height_dim, height_latent_dim, height_teacher_hidden_dims, activation)
        self.height_proprio_encoder = MLP(
            self.proprio_encoder_obs_dim, height_proprio_feature_dim, height_proprio_hidden_dims, activation
        )
        self.depth_encoder = _DepthEncoder(
            self.depth_shape, height_depth_channels, height_depth_feature_dim, activation
        )
        self.height_gru = nn.GRUCell(height_proprio_feature_dim + height_depth_feature_dim, height_gru_hidden_dim)
        self.height_latent_head = nn.Linear(height_gru_hidden_dim, height_latent_dim)
        self.height_decoder = MLP(height_latent_dim, self.height_dim, height_decoder_hidden_dims, activation)

        combined_latent_dim = latent_dim + height_latent_dim
        self.actor_head = MLP(
            self.teacher_actor_obs_dim + combined_latent_dim, actor_output_dim, hidden_dims, activation
        )
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
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent, next_hidden = self._get_student_latent_and_hidden(obs, hidden_state)
        if hidden_state is None:
            self._student_hidden_state = next_hidden.detach()
        return self._actor(self.get_student_actor_obs(obs), latent, stochastic_output)

    def act_teacher(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self._actor(self.get_teacher_actor_obs(obs), self.get_teacher_latent(obs), stochastic_output)

    def act_mixed(
        self,
        obs: TensorDict,
        teacher_mask: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        update_hidden_state: bool = False,
        student_latent: torch.Tensor | None = None,
        return_student_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        next_hidden: torch.Tensor | None = None
        if student_latent is None:
            student_latent, next_hidden = self._get_student_latent_and_hidden(obs, hidden_state)
        elif update_hidden_state:
            raise ValueError("A cached student latent cannot update the GRU state")
        if update_hidden_state:
            assert next_hidden is not None
            self._student_hidden_state = next_hidden.detach()
        mask = self._validate_teacher_mask(teacher_mask, student_latent.shape[0])
        actor_obs = torch.where(mask, self.get_teacher_actor_obs(obs), self.get_student_actor_obs(obs))
        latent = torch.where(mask, self.get_teacher_latent(obs), student_latent.detach())
        actions = self._actor(actor_obs, latent, stochastic_output)
        return (actions, student_latent.detach()) if return_student_latent else actions

    def evaluate_teacher(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        return self.critic_head(torch.cat((self.get_critic_obs(obs), self.get_teacher_latent(obs)), dim=-1))

    def evaluate_mixed(
        self,
        obs: TensorDict,
        teacher_mask: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        student_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        if student_latent is None:
            student_latent, _ = self._get_student_latent_and_hidden(obs, hidden_state)
        mask = self._validate_teacher_mask(teacher_mask, student_latent.shape[0])
        latent = torch.where(mask, self.get_teacher_latent(obs), student_latent.detach())
        return self.critic_head(torch.cat((self.get_critic_obs(obs), latent), dim=-1))

    def compute_representation_losses(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> dict[str, torch.Tensor]:
        proprio_latent = self.get_proprio_latent(obs)
        privileged_obs = self.get_privileged_obs(obs)
        with torch.no_grad():
            privileged_latent = self._normalize_latent(self.privileged_encoder(privileged_obs))
            teacher_height_latent = self._normalize_latent(self.teacher_height_encoder(self.get_height_scan(obs)))
        _, student_height_latent, height_hat, _ = self._student_height(obs, hidden_state)
        privileged_hat = self.privileged_decoder(proprio_latent)
        privileged_latent_loss = self._latent_mse(proprio_latent, privileged_latent)
        privileged_reconstruction_loss = F.mse_loss(privileged_hat, privileged_obs.detach())
        height_latent_loss = self._latent_mse(student_height_latent, teacher_height_latent)
        height_reconstruction_loss = F.mse_loss(height_hat, self.get_height_scan(obs))
        privileged_total = privileged_latent_loss + privileged_reconstruction_loss
        height_total = height_latent_loss + height_reconstruction_loss
        return {
            "privileged_latent": privileged_latent_loss,
            "privileged_reconstruction": privileged_reconstruction_loss,
            "privileged_total": privileged_total,
            "height_latent": height_latent_loss,
            "height_reconstruction": height_reconstruction_loss,
            "height_total": height_total,
            "representation_total": privileged_total + height_total,
        }

    def compute_representation_losses_sequence(
        self, obs: TensorDict, dones: torch.Tensor, hidden_state: HiddenState = None
    ) -> dict[str, torch.Tensor]:
        if len(obs.batch_size) != 2:
            raise ValueError(f"Expected [time, batch] observations, got {obs.batch_size}")
        proprio_history = self._cat_obs(obs, self.student_history_obs_groups)
        time_steps, batch_size = proprio_history.shape[:2]
        proprio_obs = self.proprio_obs_normalizer(proprio_history.flatten(start_dim=2))
        flat_proprio = proprio_obs.flatten(0, 1)
        privileged_obs = self.privileged_obs_normalizer(
            self._cat_obs(obs, self.privileged_encoder_obs_groups).flatten(0, 1)
        )
        with torch.no_grad():
            privileged_latent = self._normalize_latent(self.privileged_encoder(privileged_obs)).view(
                time_steps, batch_size, self.latent_dim
            )
            teacher_height_latent = self._normalize_latent(
                self.teacher_height_encoder(self.get_height_scan(obs).flatten(0, 1))
            ).view(time_steps, batch_size, self.height_latent_dim)

        proprio_latent = self._normalize_latent(self.proprio_encoder(flat_proprio)).view(
            time_steps, batch_size, self.latent_dim
        )
        privileged_hat = self.privileged_decoder(proprio_latent.flatten(0, 1)).view(
            time_steps, batch_size, self.privileged_encoder_obs_dim
        )
        student_height_latent, height_hat = self._student_height_sequence(
            proprio_obs, self.get_depth_obs(obs), dones, hidden_state
        )
        privileged_latent_loss = self._latent_mse(proprio_latent, privileged_latent)
        privileged_reconstruction_loss = F.mse_loss(privileged_hat, privileged_obs.view_as(privileged_hat).detach())
        height_latent_loss = self._latent_mse(student_height_latent, teacher_height_latent)
        height_reconstruction_loss = F.mse_loss(height_hat, self.get_height_scan(obs))
        privileged_total = privileged_latent_loss + privileged_reconstruction_loss
        height_total = height_latent_loss + height_reconstruction_loss
        return {
            "privileged_latent": privileged_latent_loss,
            "privileged_reconstruction": privileged_reconstruction_loss,
            "privileged_total": privileged_total,
            "height_latent": height_latent_loss,
            "height_reconstruction": height_reconstruction_loss,
            "height_total": height_total,
            "representation_total": privileged_total + height_total,
        }

    def ppo_parameters(self):
        yield from self.privileged_encoder.parameters()
        yield from self.teacher_height_encoder.parameters()
        yield from self.actor_head.parameters()
        yield from self.critic_head.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def representation_parameters(self):
        yield from self.proprio_encoder.parameters()
        yield from self.privileged_decoder.parameters()
        yield from self.height_proprio_encoder.parameters()
        yield from self.depth_encoder.parameters()
        yield from self.height_gru.parameters()
        yield from self.height_latent_head.parameters()
        yield from self.height_decoder.parameters()

    def proprio_parameters(self):
        """Compatibility alias used by the base representation PPO lifecycle."""
        yield from self.representation_parameters()

    def get_teacher_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.teacher_actor_obs_normalizer(self._cat_obs(obs, self.teacher_actor_obs_groups))

    def get_student_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.student_actor_obs_normalizer(self._cat_obs(obs, self.student_history_obs_groups)[..., -1, :])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_obs_normalizer(self._cat_obs(obs, self.critic_obs_groups))

    def get_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.proprio_obs_normalizer(self._cat_obs(obs, self.student_history_obs_groups).flatten(start_dim=1))

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.privileged_obs_normalizer(self._cat_obs(obs, self.privileged_encoder_obs_groups))

    def get_depth_obs(self, obs: TensorDict) -> torch.Tensor:
        depth = obs[self.depth_obs_group]
        expected = (*obs.batch_size, *self.depth_shape)
        if tuple(depth.shape) != expected:
            raise ValueError(f"Expected depth shape {expected}, got {tuple(depth.shape)}")
        return depth

    def get_height_scan(self, obs: TensorDict) -> torch.Tensor:
        return self._cat_obs(obs, self.height_obs_groups)

    def get_proprio_latent(self, obs: TensorDict) -> torch.Tensor:
        return self._normalize_latent(self.proprio_encoder(self.get_proprio_obs(obs)))

    def get_teacher_latent(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            (
                self._normalize_latent(self.privileged_encoder(self.get_privileged_obs(obs))),
                self._normalize_latent(self.teacher_height_encoder(self.get_height_scan(obs))),
            ),
            dim=-1,
        )

    def get_hidden_state(self) -> HiddenState:
        return self._student_hidden_state

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self._student_hidden_state = None if hidden_state is None else hidden_state.detach()  # type: ignore[union-attr]
        elif hidden_state is not None:
            raise NotImplementedError("VisionCTS cannot reset a supplied hidden state")
        elif self._student_hidden_state is not None:
            self._student_hidden_state[
                dones.to(dtype=torch.bool, device=self._student_hidden_state.device).view(-1)
            ] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._student_hidden_state is None:
            return
        if dones is None:
            self._student_hidden_state = self._student_hidden_state.detach()
        else:
            mask = dones.to(dtype=torch.bool, device=self._student_hidden_state.device).view(-1)
            self._student_hidden_state[mask] = self._student_hidden_state[mask].detach()

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        student_history = self._cat_obs(obs, self.student_history_obs_groups)
        self.teacher_actor_obs_normalizer.update(self._cat_obs(obs, self.teacher_actor_obs_groups))  # type: ignore[union-attr]
        self.student_actor_obs_normalizer.update(student_history[:, -1, :])  # type: ignore[union-attr]
        self.critic_obs_normalizer.update(self._cat_obs(obs, self.critic_obs_groups))  # type: ignore[union-attr]
        self.proprio_obs_normalizer.update(student_history.flatten(start_dim=1))  # type: ignore[union-attr]
        self.privileged_obs_normalizer.update(self._cat_obs(obs, self.privileged_encoder_obs_groups))  # type: ignore[union-attr]

    @property
    def output_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        assert self.distribution is not None
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.kl_divergence(old_params, new_params)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxVisionCTSPolicy(self, verbose)

    def _student_height(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("VisionCTS expects a tensor GRU hidden state")
        proprio_obs = self.get_proprio_obs(obs)
        batch_size = proprio_obs.shape[0]
        state = hidden_state
        if state is None:
            state = self._student_hidden_state
        if state is None:
            state = proprio_obs.new_zeros(batch_size, self.height_gru_hidden_dim)
        height_state = self.height_gru(
            torch.cat((self.height_proprio_encoder(proprio_obs), self.depth_encoder(self.get_depth_obs(obs))), dim=-1),
            state,
        )
        height_latent = self._normalize_latent(self.height_latent_head(height_state))
        return proprio_obs, height_latent, self.height_decoder(height_latent), height_state

    def _student_height_sequence(
        self,
        proprio_obs: torch.Tensor,
        depth: torch.Tensor,
        dones: torch.Tensor,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("VisionCTS expects a tensor GRU hidden state")
        time_steps, batch_size = proprio_obs.shape[:2]
        depth_features = self.depth_encoder(depth.flatten(0, 1)).view(time_steps, batch_size, -1)
        proprio_features = self.height_proprio_encoder(proprio_obs.flatten(0, 1)).view(time_steps, batch_size, -1)
        state = hidden_state
        if state is None:
            state = proprio_obs.new_zeros(batch_size, self.height_gru_hidden_dim)
        latents: list[torch.Tensor] = []
        reconstructions: list[torch.Tensor] = []
        for step in range(time_steps):
            if step > 0:
                state = state * (~dones[step - 1].to(dtype=torch.bool).view(-1, 1)).to(state.dtype)
            state = self.height_gru(torch.cat((proprio_features[step], depth_features[step]), dim=-1), state)
            latent = self._normalize_latent(self.height_latent_head(state))
            latents.append(latent)
            reconstructions.append(self.height_decoder(latent))
        return torch.stack(latents), torch.stack(reconstructions)

    def _get_student_latent_and_hidden(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, height_latent, _, next_hidden = self._student_height(obs, hidden_state)
        return torch.cat((self.get_proprio_latent(obs), height_latent), dim=-1), next_hidden

    def _actor(self, actor_obs: torch.Tensor, latent: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        output = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        if self.distribution is None:
            return output
        if stochastic_output:
            self.distribution.update(output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(output)

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(latent, p=2.0, dim=-1) if self.normalize_latent else latent

    @staticmethod
    def _latent_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student, teacher, reduction="none").sum(dim=-1).mean()

    @staticmethod
    def _validate_teacher_mask(teacher_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if teacher_mask.ndim == 1:
            teacher_mask = teacher_mask.unsqueeze(-1)
        if tuple(teacher_mask.shape) != (batch_size, 1):
            raise ValueError(f"teacher_mask must have shape [{batch_size}] or [{batch_size}, 1]")
        return teacher_mask.to(dtype=torch.bool)

    @staticmethod
    def _cat_obs(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        missing = [group for group in groups if group not in obs]
        if missing:
            raise ValueError(f"missing observation groups: {missing}")
        return torch.cat([obs[group] for group in groups], dim=-1)

    @staticmethod
    def _get_obs_dim(obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups = obs_groups[obs_set]
        dim = 0
        for group in groups:
            if len(obs[group].shape) != 2:
                raise ValueError(f"'{obs_set}' requires 1D observations, got {obs[group].shape} for '{group}'")
            dim += obs[group].shape[-1]
        return groups, dim

    @staticmethod
    def _get_history_shape(
        obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int, int]:
        groups = obs_groups[obs_set]
        history_length: int | None = None
        frame_dim = 0
        for group in groups:
            value = obs[group]
            if len(value.shape) != 3:
                raise ValueError(f"'{obs_set}' requires [batch, history, features], got {value.shape}")
            history_length = value.shape[-2] if history_length is None else history_length
            if value.shape[-2] != history_length:
                raise ValueError("All VisionCTS history groups must use the same history length")
            frame_dim += value.shape[-1]
        if history_length is None:
            raise ValueError("VisionCTS requires a student history group")
        return groups, history_length, frame_dim

    @staticmethod
    def _get_depth_group_and_shape(
        obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[str, tuple[int, int, int]]:
        groups = obs_groups[obs_set]
        if len(groups) != 1 or len(obs[groups[0]].shape) != 4:
            raise ValueError(f"'{obs_set}' must contain one [batch, channels, height, width] group")
        return groups[0], tuple(obs[groups[0]].shape[1:])


class _OnnxVisionCTSPolicy(nn.Module):
    """Deployment graph with explicit recurrent state input and output."""

    is_recurrent: bool = True

    def __init__(self, model: VisionCTSActorCritic, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.use_external_data = False
        self.student_actor_obs_normalizer = copy.deepcopy(model.student_actor_obs_normalizer)
        self.proprio_obs_normalizer = copy.deepcopy(model.proprio_obs_normalizer)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.height_proprio_encoder = copy.deepcopy(model.height_proprio_encoder)
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.height_gru = copy.deepcopy(model.height_gru)
        self.height_latent_head = copy.deepcopy(model.height_latent_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.history_length = model.student_history_length
        self.actor_input_size = model.student_actor_obs_dim
        self.depth_input_shape = model.depth_shape
        self.hidden_size = model.height_gru_hidden_dim

    def forward(
        self, student_history: torch.Tensor, depth: torch.Tensor, hidden_state_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_obs = self.student_actor_obs_normalizer(student_history[:, -1, :])
        proprio_obs = self.proprio_obs_normalizer(student_history.flatten(start_dim=1))
        privileged_latent = self.proprio_encoder(proprio_obs)
        height_state = self.height_gru(
            torch.cat((self.height_proprio_encoder(proprio_obs), self.depth_encoder(depth)), dim=-1),
            hidden_state_in,
        )
        height_latent = self.height_latent_head(height_state)
        if self.normalize_latent:
            privileged_latent = F.normalize(privileged_latent, p=2.0, dim=-1)
            height_latent = F.normalize(height_latent, p=2.0, dim=-1)
        actions = self.deterministic_output(
            self.actor_head(torch.cat((actor_obs, privileged_latent, height_latent), dim=-1))
        )
        return actions, height_state

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.history_length, self.actor_input_size),
            torch.zeros(1, *self.depth_input_shape),
            torch.zeros(1, self.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["student_history", "depth", "hidden_state_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "hidden_state_out"]
