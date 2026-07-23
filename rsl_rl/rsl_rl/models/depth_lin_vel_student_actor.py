"""Deployable depth student actor for staged teacher-student distillation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.depth_representation_velocity_actor_critic import _DepthCNN
from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class DepthLinVelStudentActor(nn.Module):
    """Student-only depth policy with a latent and linear-velocity auxiliary head."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 64,
        activation: str = "elu",
        obs_normalization: bool = False,
        normalize_latent: bool = True,
        distribution_cfg: dict | None = None,
        depth_feature_dim: int = 64,
        depth_gru_hidden_dim: int = 64,
        depth_channels: tuple[int, ...] | list[int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        self.history_group = self._one_group(obs, obs_groups, "student_history", 3)
        self.command_group = self._one_group(obs, obs_groups, "student_command", 2)
        self.depth_group = self._one_group(obs, obs_groups, "student_depth", 4)
        history = obs[self.history_group]
        self.history_length = history.shape[-2]
        self.current_proprio_dim = history.shape[-1]
        self.proprio_history_obs_dim = self.history_length * self.current_proprio_dim
        self.command_dim = obs[self.command_group].shape[-1]
        self.depth_shape = tuple(obs[self.depth_group].shape[1:])
        self.latent_dim = latent_dim
        self.depth_gru_hidden_dim = depth_gru_hidden_dim
        self.normalize_latent = normalize_latent
        self.obs_normalization = obs_normalization

        if obs_normalization:
            self.proprio_history_obs_normalizer = EmpiricalNormalization(self.proprio_history_obs_dim)
            self.current_proprio_obs_normalizer = EmpiricalNormalization(self.current_proprio_dim)
            self.command_obs_normalizer = EmpiricalNormalization(self.command_dim)
            self.lin_vel_normalizer = EmpiricalNormalization(3)
        else:
            self.proprio_history_obs_normalizer = nn.Identity()
            self.current_proprio_obs_normalizer = nn.Identity()
            self.command_obs_normalizer = nn.Identity()
            self.lin_vel_normalizer = nn.Identity()

        if distribution_cfg is None:
            self.distribution = None
            actor_output_dim = output_dim
        else:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            distribution_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = distribution_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim

        encoder_hidden_dims = hidden_dims if encoder_hidden_dims is None else encoder_hidden_dims
        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must not be empty.")
        self.depth_encoder = _DepthCNN(self.depth_shape, depth_channels, depth_feature_dim, activation)
        self.depth_gru = nn.GRUCell(depth_feature_dim, depth_gru_hidden_dim)
        self.proprio_encoder = MLP(
            self.proprio_history_obs_dim + depth_gru_hidden_dim,
            encoder_hidden_dims[-1],
            encoder_hidden_dims,
            activation,
        )
        self.student_latent_head = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.lin_vel_head = nn.Linear(encoder_hidden_dims[-1], 3)
        actor_obs_dim = 3 + self.current_proprio_dim + self.command_dim
        self.actor_head = MLP(actor_obs_dim + latent_dim, actor_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_head)
        self._hidden_state: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks
        actions, _, _ = self.forward_with_outputs(obs, hidden_state, stochastic_output=stochastic_output)
        return actions

    def forward_with_outputs(
        self,
        obs: TensorDict,
        hidden_state: HiddenState = None,
        *,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, latent, predicted_lin_vel, next_hidden = self.forward_with_state(
            obs, hidden_state, stochastic_output=stochastic_output
        )
        if hidden_state is None:
            # Rollout/inference owns the module's persistent state.  Training uses
            # ``forward_with_state`` directly so it can retain gradients through a
            # configurable truncated-BPTT window without mutating this state.
            self._hidden_state = next_hidden.detach()
        return actions, latent, predicted_lin_vel

    def forward_with_state(
        self,
        obs: TensorDict,
        hidden_state: torch.Tensor | None = None,
        *,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one step without changing the persistent GRU state.

        The staged distillation algorithm uses the returned state for
        truncated backpropagation through time.  Keeping this method functional
        prevents rollout hidden state from being overwritten during updates.
        """
        latent, predicted_lin_vel, next_hidden = self._get_student_outputs(obs, hidden_state)
        actor_obs = torch.cat(
            (
                self.lin_vel_normalizer(predicted_lin_vel),
                self.current_proprio_obs_normalizer(self.get_history(obs)[:, -1, :]),
                self.command_obs_normalizer(obs[self.command_group]),
            ),
            dim=-1,
        )
        output = self.actor_head(torch.cat((actor_obs, latent), dim=-1))
        if self.distribution is not None:
            self.distribution.update(output)
            actions = self.distribution.sample() if stochastic_output else self.distribution.deterministic_output(output)
        else:
            actions = output
        return actions, latent, predicted_lin_vel, next_hidden

    def get_student_outputs(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, predicted_lin_vel, next_hidden = self._get_student_outputs(obs, hidden_state)
        if hidden_state is None:
            self._hidden_state = next_hidden.detach()
        return latent, predicted_lin_vel

    def _get_student_outputs(
        self, obs: TensorDict, hidden_state: HiddenState = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_state is not None and not isinstance(hidden_state, torch.Tensor):
            raise ValueError("DepthLinVelStudentActor expects a tensor GRU hidden state.")
        history = self.get_history(obs)
        depth = self.get_depth(obs)
        active_hidden = self._hidden_state if hidden_state is None else hidden_state
        if active_hidden is None:
            active_hidden = torch.zeros(
                history.shape[0], self.depth_gru_hidden_dim, device=history.device, dtype=history.dtype
            )
        expected_hidden_shape = (history.shape[0], self.depth_gru_hidden_dim)
        if tuple(active_hidden.shape) != expected_hidden_shape:
            raise ValueError(f"Expected hidden state {expected_hidden_shape}, got {tuple(active_hidden.shape)}.")
        next_hidden = self.depth_gru(self.depth_encoder(depth), active_hidden)
        encoded = self.proprio_encoder(
            torch.cat((self.proprio_history_obs_normalizer(history.flatten(start_dim=1)), next_hidden), dim=-1)
        )
        latent = self.student_latent_head(encoded)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        return latent, self.lin_vel_head(encoded), next_hidden

    def get_history(self, obs: TensorDict) -> torch.Tensor:
        return obs[self.history_group]

    def get_depth(self, obs: TensorDict) -> torch.Tensor:
        depth = obs[self.depth_group]
        expected_shape = (obs.batch_size[0], *self.depth_shape)
        if tuple(depth.shape) != expected_shape:
            raise ValueError(f"Expected depth shape {expected_shape}, got {tuple(depth.shape)}.")
        return depth

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            history = self.get_history(obs)
            self.proprio_history_obs_normalizer.update(history.flatten(start_dim=1))  # type: ignore
            self.current_proprio_obs_normalizer.update(history[:, -1, :])  # type: ignore
            self.command_obs_normalizer.update(obs[self.command_group])  # type: ignore

    def initialize_from_teacher(self, teacher: nn.Module) -> None:
        """Copy the compatible policy head and input normalization from a frozen teacher."""
        self.actor_head.load_state_dict(teacher.actor_head.state_dict())  # type: ignore[attr-defined]
        if self.distribution is not None:
            self.distribution.load_state_dict(teacher.distribution.state_dict())  # type: ignore[attr-defined]
        for name in ("current_proprio_obs_normalizer", "command_obs_normalizer", "lin_vel_normalizer"):
            getattr(self, name).load_state_dict(getattr(teacher, name).state_dict())

    def encoder_parameters(self):
        yield from self.depth_encoder.parameters()
        yield from self.depth_gru.parameters()
        yield from self.proprio_encoder.parameters()
        yield from self.student_latent_head.parameters()
        yield from self.lin_vel_head.parameters()

    def actor_parameters(self):
        yield from self.actor_head.parameters()
        if self.distribution is not None:
            yield from self.distribution.parameters()

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self._hidden_state = None if hidden_state is None else hidden_state.detach()  # type: ignore[union-attr]
        elif self._hidden_state is not None:
            self._hidden_state[dones.to(dtype=torch.bool).view(-1)] = 0.0

    def get_hidden_state(self) -> HiddenState:
        return self._hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def as_jit(self) -> nn.Module:
        return _TorchDepthLinVelStudentActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxDepthLinVelStudentActor(self, verbose)

    @staticmethod
    def _one_group(obs: TensorDict, obs_groups: dict[str, list[str]], name: str, ndim: int) -> str:
        groups = obs_groups[name]
        if len(groups) != 1:
            raise ValueError(f"{name} must contain exactly one observation group.")
        group = groups[0]
        if obs[group].ndim != ndim:
            raise ValueError(f"{group} must have {ndim} dimensions, got {obs[group].shape}.")
        return group


class _TorchDepthLinVelStudentActor(nn.Module):
    def __init__(self, model: DepthLinVelStudentActor) -> None:
        super().__init__()
        self.proprio_history_obs_normalizer = copy.deepcopy(model.proprio_history_obs_normalizer)
        self.current_proprio_obs_normalizer = copy.deepcopy(model.current_proprio_obs_normalizer)
        self.command_obs_normalizer = copy.deepcopy(model.command_obs_normalizer)
        self.lin_vel_normalizer = copy.deepcopy(model.lin_vel_normalizer)
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.depth_gru = copy.deepcopy(model.depth_gru)
        self.proprio_encoder = copy.deepcopy(model.proprio_encoder)
        self.student_latent_head = copy.deepcopy(model.student_latent_head)
        self.lin_vel_head = copy.deepcopy(model.lin_vel_head)
        self.actor_head = copy.deepcopy(model.actor_head)
        self.normalize_latent = model.normalize_latent
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.register_buffer("hidden_state", torch.zeros(1, model.depth_gru_hidden_dim))

    def forward(
        self, proprio_history: torch.Tensor, actor_command: torch.Tensor, depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.depth_gru(self.depth_encoder(depth), self.hidden_state)
        self.hidden_state[:] = hidden.detach()
        return self._forward(proprio_history, actor_command, hidden)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0

    def _forward(
        self, proprio_history: torch.Tensor, actor_command: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.proprio_encoder(
            torch.cat((self.proprio_history_obs_normalizer(proprio_history.flatten(start_dim=1)), hidden), dim=-1)
        )
        latent = self.student_latent_head(encoded)
        if self.normalize_latent:
            latent = F.normalize(latent, p=2.0, dim=-1)
        predicted_lin_vel = self.lin_vel_head(encoded)
        actor_obs = torch.cat(
            (
                self.lin_vel_normalizer(predicted_lin_vel),
                self.current_proprio_obs_normalizer(proprio_history[:, -1, :]),
                self.command_obs_normalizer(actor_command),
            ),
            dim=-1,
        )
        return self.deterministic_output(self.actor_head(torch.cat((actor_obs, latent), dim=-1))), predicted_lin_vel


class _OnnxDepthLinVelStudentActor(_TorchDepthLinVelStudentActor):
    is_recurrent: bool = True

    def __init__(self, model: DepthLinVelStudentActor, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.history_length = model.history_length
        self.proprio_input_size = model.current_proprio_dim
        self.command_input_size = model.command_dim
        self.depth_input_shape = model.depth_shape
        self.hidden_size = model.depth_gru_hidden_dim

    def forward(
        self,
        proprio_history: torch.Tensor,
        actor_command: torch.Tensor,
        depth: torch.Tensor,
        hidden_state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_state_out = self.depth_gru(self.depth_encoder(depth), hidden_state_in)
        actions, predicted_lin_vel = self._forward(proprio_history, actor_command, hidden_state_out)
        return actions, predicted_lin_vel, hidden_state_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.history_length, self.proprio_input_size),
            torch.zeros(1, self.command_input_size),
            torch.zeros(1, *self.depth_input_shape),
            torch.zeros(1, self.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprio_history", "actor_command", "depth", "hidden_state_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "predicted_lin_vel", "hidden_state_out"]
