# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`RolloutStorage.add_transition` to record the data.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            """Observations at the current step."""

            self.actions: torch.Tensor | None = None
            """Actions taken at the current step."""

            self.applied_actions: torch.Tensor | None = None
            """Policy-space actions actually applied during the environment step."""

            self.rewards: torch.Tensor | None = None
            """Rewards received after the action."""

            self.dones: torch.Tensor | None = None
            """Done flags indicating episode termination."""

            # For reinforcement learning
            self.values: torch.Tensor | None = None
            """Value estimates at the current step (RL only)."""

            self.actions_log_prob: torch.Tensor | None = None
            """Log probability of the taken actions (RL only)."""

            self.distribution_params: tuple[torch.Tensor, ...] | None = None
            """Parameters of the action distribution (RL only)."""

            self.teacher_mask: torch.Tensor | None = None
            """Teacher/student rollout partition for concurrent training."""

            self.student_latent: torch.Tensor | None = None
            """Student latent cached while collecting a concurrent rollout."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = None
            """Privileged (teacher) actions (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)
            """Hidden states for recurrent networks, e.g., (actor, critic)."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    class Batch:
        """A batch of data yielded by the rollout storage generators.

        This class provides named access to mini-batch fields. Fields are optional to support different training modes
        (RL vs distillation) and architectures (feedforward vs recurrent).
        """

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            returns: torch.Tensor | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState, HiddenState] = (None, None),
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
            next_observations: TensorDict | None = None,
            future_observations: TensorDict | None = None,
            applied_actions: torch.Tensor | None = None,
            teacher_mask: torch.Tensor | None = None,
            student_latents: torch.Tensor | None = None,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict | None = observations
            """Batch of observations."""

            # For reinforcement learning
            self.actions: torch.Tensor | None = actions
            """Batch of actions."""

            self.applied_actions: torch.Tensor | None = applied_actions
            """Batch of policy-space actions actually applied by the environment."""

            self.next_observations: TensorDict | None = next_observations
            """Observations at the requested future horizon after ``observations``."""

            self.future_observations: TensorDict | None = future_observations
            """Observation sequence after ``observations``, with leading time dimension."""

            self.values: torch.Tensor | None = values
            """Batch of value estimates (RL only)."""

            self.advantages: torch.Tensor | None = advantages
            """Batch of advantage estimates (RL only)."""

            self.returns: torch.Tensor | None = returns
            """Batch of return targets (RL only)."""

            self.old_actions_log_prob: torch.Tensor | None = old_actions_log_prob
            """Batch of log probabilities of the old actions (RL only)."""

            self.old_distribution_params: tuple[torch.Tensor, ...] | None = old_distribution_params
            """Batch of parameters of the old action distribution (RL only)."""

            self.teacher_mask: torch.Tensor | None = teacher_mask
            """Teacher/student path used to collect each concurrent sample."""

            self.student_latents: torch.Tensor | None = student_latents
            """Student latents fixed during the PPO update."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = privileged_actions
            """Batch of privileged (teacher) actions (distillation only)."""

            self.dones: torch.Tensor | None = dones
            """Batch of done flags (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState, HiddenState] = hidden_states
            """Batch of hidden states for recurrent networks (RL recurrent only)."""

            self.masks: torch.Tensor | None = masks
            """Batch of trajectory masks for recurrent networks (RL recurrent only)."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate rollout buffers for a specific training mode and batch shape."""
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.applied_actions: torch.Tensor | None = None
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.latent_dynamics_valid_fraction = 0.0
        self.latent_dynamics_valid_fractions: dict[int, float] = {}
        self.latent_dynamics_sequence_valid_fraction = 0.0

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.distribution_params: tuple[torch.Tensor, ...] | None = None  # Lazily initialized on first transition
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.teacher_masks: torch.Tensor | None = None
            self.student_latents: torch.Tensor | None = None

        # For recurrent networks
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Add one transition to the storage at the current step index."""
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        if transition.applied_actions is not None:
            if self.applied_actions is None:
                self.applied_actions = torch.zeros_like(self.actions)
            self.applied_actions[self.step].copy_(transition.applied_actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)  # type: ignore

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)  # type: ignore
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if self.distribution_params is None:  # Initialize the distribution parameters
                self.distribution_params = tuple(
                    torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                    for p in transition.distribution_params  # type: ignore
                )
            for i, p in enumerate(transition.distribution_params):  # type: ignore
                self.distribution_params[i][self.step].copy_(p)
            if transition.teacher_mask is not None:
                if self.teacher_masks is None:
                    self.teacher_masks = torch.zeros(
                        self.num_transitions_per_env,
                        self.num_envs,
                        1,
                        dtype=torch.bool,
                        device=self.device,
                    )
                self.teacher_masks[self.step].copy_(transition.teacher_mask.view(-1, 1))
            elif self.teacher_masks is not None:
                raise ValueError("teacher_mask must be provided for every concurrent transition")
            if transition.student_latent is not None:
                if self.student_latents is None:
                    self.student_latents = torch.zeros(
                        self.num_transitions_per_env,
                        *transition.student_latent.shape,
                        dtype=transition.student_latent.dtype,
                        device=self.device,
                    )
                self.student_latents[self.step].copy_(transition.student_latent)
            elif self.student_latents is not None:
                raise ValueError("student_latent must be provided for every concurrent transition")

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0
        self.latent_dynamics_valid_fraction = 0.0
        self.latent_dynamics_valid_fractions.clear()
        self.latent_dynamics_sequence_valid_fraction = 0.0

    # For distillation
    def generator(self) -> Generator[Batch, None, None]:
        """Yield per-timestep batches for distillation training."""
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield RolloutStorage.Batch(
                observations=self.observations[i],  # type: ignore
                privileged_actions=self.privileged_actions[i],
                dones=self.dones[i],
            )

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches for feedforward RL updates."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        # Flatten the data
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        teacher_masks = self.teacher_masks.flatten(0, 1) if self.teacher_masks is not None else None
        student_latents = self.student_latents.flatten(0, 1) if self.student_latents is not None else None
        saved_hidden_state_a = self._flatten_saved_hidden_states(self.saved_hidden_state_a)
        saved_hidden_state_c = self._flatten_saved_hidden_states(self.saved_hidden_state_c)

        for _ in range(num_epochs):
            for batch_idx in self._flat_mini_batch_indices(num_mini_batches, teacher_masks):
                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    hidden_states=(
                        self._select_hidden_states(saved_hidden_state_a, batch_idx),
                        self._select_hidden_states(saved_hidden_state_c, batch_idx),
                    ),
                    teacher_mask=teacher_masks[batch_idx] if teacher_masks is not None else None,
                    student_latents=student_latents[batch_idx] if student_latents is not None else None,
                )

    def representation_chunk_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
        chunk_length: int,
        student_only: bool = False,
    ) -> Generator[Batch, None, None]:
        """Yield continuous trajectory chunks for representation learning."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {chunk_length}")

        chunk_length = min(chunk_length, self.num_transitions_per_env)
        chunk_starts = torch.arange(
            0,
            self.num_transitions_per_env - chunk_length + 1,
            chunk_length,
            device=self.device,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        if student_only and self.teacher_masks is not None:
            fixed_mask = self.teacher_masks[0, :, 0]
            if not torch.equal(
                self.teacher_masks[:, :, 0],
                fixed_mask.unsqueeze(0).expand_as(self.teacher_masks[:, :, 0]),
            ):
                raise ValueError("Concurrent representation chunks require a fixed teacher mask")
            env_ids = env_ids[~fixed_mask]
        if env_ids.numel() == 0:
            raise ValueError("Representation learning requires at least one student environment")
        chunk_envs = env_ids.repeat_interleave(chunk_starts.numel())
        chunk_times = chunk_starts.repeat(env_ids.numel())
        num_chunks = chunk_envs.numel()
        effective_num_mini_batches = min(num_mini_batches, num_chunks)

        for _ in range(num_epochs):
            permutation = torch.randperm(num_chunks, device=self.device)
            for chunk_indices in torch.tensor_split(permutation, effective_num_mini_batches):
                starts = chunk_times[chunk_indices]
                envs = chunk_envs[chunk_indices]
                time_indices = starts.unsqueeze(0) + torch.arange(chunk_length, device=self.device).unsqueeze(1)
                env_indices = envs.unsqueeze(0).expand(chunk_length, -1)
                observations = TensorDict(
                    {
                        key: value[time_indices, env_indices]
                        for key, value in self.observations.items()
                    },
                    batch_size=[chunk_length, chunk_indices.numel()],
                    device=self.device,
                )
                yield RolloutStorage.Batch(
                    observations=observations,
                    dones=self.dones[time_indices, env_indices],
                    hidden_states=(
                        self._select_chunk_initial_states(
                            self.saved_hidden_state_a,
                            starts,
                            envs,
                        ),
                        self._select_chunk_initial_states(
                            self.saved_hidden_state_c,
                            starts,
                            envs,
                        ),
                    ),
                    teacher_mask=(
                        self.teacher_masks[time_indices, env_indices]
                        if self.teacher_masks is not None
                        else None
                    ),
                )

    def latent_dynamics_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
        horizon: int = 1,
    ) -> Generator[Batch, None, None]:
        """Yield valid direct-transition pairs and aligned action blocks for one horizon."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.applied_actions is None:
            raise ValueError("Applied actions were not recorded in rollout storage.")
        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("Latent dynamics epochs and mini-batches must be positive.")
        if horizon <= 0:
            raise ValueError(f"Latent dynamics horizon must be positive, got {horizon}.")

        num_pair_steps = self.num_transitions_per_env - horizon
        if num_pair_steps <= 0:
            self.latent_dynamics_valid_fraction = 0.0
            self.latent_dynamics_valid_fractions[horizon] = 0.0
            return

        transition_is_valid = ~self.dones.squeeze(-1).bool()
        not_done = torch.stack(
            [
                transition_is_valid[offset : offset + num_pair_steps]
                for offset in range(horizon)
            ],
            dim=0,
        ).all(dim=0)
        valid = not_done
        self.latent_dynamics_valid_fraction = valid.float().mean().item()
        self.latent_dynamics_valid_fractions[horizon] = self.latent_dynamics_valid_fraction
        valid_indices = valid.flatten().nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return

        observations_t = self.observations[:num_pair_steps].flatten(0, 1)
        observations_future = self.observations[horizon:].flatten(0, 1)
        applied_action_blocks = torch.cat(
            [
                self.applied_actions[offset : offset + num_pair_steps]
                for offset in range(horizon)
            ],
            dim=-1,
        ).flatten(0, 1)
        effective_num_mini_batches = min(num_mini_batches, valid_indices.numel())

        for _ in range(num_epochs):
            permutation = valid_indices[
                torch.randperm(valid_indices.numel(), device=self.device)
            ]
            for batch_indices in torch.tensor_split(permutation, effective_num_mini_batches):
                yield RolloutStorage.Batch(
                    observations=observations_t[batch_indices],
                    next_observations=observations_future[batch_indices],
                    applied_actions=applied_action_blocks[batch_indices],
                )

    def latent_dynamics_sequence_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
        rollout_horizon: int = 5,
    ) -> Generator[Batch, None, None]:
        """Yield valid starts, all future observations, and ordered action sequences."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.applied_actions is None:
            raise ValueError("Applied actions were not recorded in rollout storage.")
        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("Latent dynamics epochs and mini-batches must be positive.")
        if rollout_horizon <= 0:
            raise ValueError(f"Latent rollout horizon must be positive, got {rollout_horizon}.")

        num_start_steps = self.num_transitions_per_env - rollout_horizon
        if num_start_steps <= 0:
            self.latent_dynamics_sequence_valid_fraction = 0.0
            return

        transition_is_valid = ~self.dones.squeeze(-1).bool()
        valid = torch.stack(
            [
                transition_is_valid[offset : offset + num_start_steps]
                for offset in range(rollout_horizon)
            ],
            dim=0,
        ).all(dim=0)
        self.latent_dynamics_sequence_valid_fraction = valid.float().mean().item()
        valid_indices = valid.flatten().nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return

        observations_t = self.observations[:num_start_steps].flatten(0, 1)
        future_observations = TensorDict(
            {
                key: torch.stack(
                    [
                        value[offset : offset + num_start_steps]
                        for offset in range(1, rollout_horizon + 1)
                    ],
                    dim=0,
                ).flatten(1, 2)
                for key, value in self.observations.items()
            },
            batch_size=[rollout_horizon, num_start_steps * self.num_envs],
            device=self.device,
        )
        applied_action_sequence = torch.stack(
            [
                self.applied_actions[offset : offset + num_start_steps]
                for offset in range(rollout_horizon)
            ],
            dim=0,
        ).flatten(1, 2)
        effective_num_mini_batches = min(num_mini_batches, valid_indices.numel())

        for _ in range(num_epochs):
            permutation = valid_indices[
                torch.randperm(valid_indices.numel(), device=self.device)
            ]
            for batch_indices in torch.tensor_split(permutation, effective_num_mini_batches):
                yield RolloutStorage.Batch(
                    observations=observations_t[batch_indices],
                    future_observations=future_observations[:, batch_indices],
                    applied_actions=applied_action_sequence[:, batch_indices],
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield trajectory mini-batches with masks and recurrent hidden states."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches

        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # Handle the hidden states
                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    # Remove the tuple for GRU
                    hidden_state_a_batch = (
                        hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                    )
                else:
                    hidden_state_a_batch = None
                if self.saved_hidden_state_c is not None:
                    hidden_state_c_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_c
                    ]
                    hidden_state_c_batch = (
                        hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                    )
                else:
                    hidden_state_c_batch = None

                # Yield the mini-batch
                yield RolloutStorage.Batch(
                    observations=padded_obs_trajectories[:, first_traj:last_traj],  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    hidden_states=(hidden_state_a_batch, hidden_state_c_batch),  # type: ignore
                    masks=trajectory_masks[:, first_traj:last_traj],
                )

                first_traj = last_traj

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        """Save recurrent hidden states to the rollout storage."""
        if hidden_states == (None, None):
            return
        # Make a tuple out of GRU hidden states to match the LSTM format
        if hidden_states[0] is not None:
            hidden_state_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        if hidden_states[1] is not None:
            hidden_state_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # Initialize hidden states if needed
        if self.saved_hidden_state_a is None and hidden_states[0] is not None:
            self.saved_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
        if self.saved_hidden_state_c is None and hidden_states[1] is not None:
            self.saved_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]
        # Copy the states
        if hidden_states[0] is not None:
            for i in range(len(hidden_state_a)):
                self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])  # type: ignore
        if hidden_states[1] is not None:
            for i in range(len(hidden_state_c)):
                self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])  # type: ignore

    def _flat_mini_batch_indices(
        self,
        num_mini_batches: int,
        teacher_masks: torch.Tensor | None,
    ) -> Generator[torch.Tensor, None, None]:
        """Yield shuffled batches, balanced across concurrent CTS paths when present."""
        batch_size = self.num_envs * self.num_transitions_per_env
        if teacher_masks is None:
            permutation = torch.randperm(batch_size, device=self.device)
            yield from torch.tensor_split(permutation, min(num_mini_batches, batch_size))
            return

        flat_mask = teacher_masks.view(-1)
        teacher_indices = flat_mask.nonzero(as_tuple=False).flatten()
        student_indices = (~flat_mask).nonzero(as_tuple=False).flatten()
        if teacher_indices.numel() == 0 or student_indices.numel() == 0:
            raise ValueError("Concurrent PPO mini-batches require both teacher and student samples")
        effective_num_mini_batches = min(
            num_mini_batches,
            teacher_indices.numel(),
            student_indices.numel(),
        )
        teacher_splits = torch.tensor_split(
            teacher_indices[torch.randperm(teacher_indices.numel(), device=self.device)],
            effective_num_mini_batches,
        )
        student_splits = torch.tensor_split(
            student_indices[torch.randperm(student_indices.numel(), device=self.device)],
            effective_num_mini_batches,
        )
        for teacher_batch, student_batch in zip(teacher_splits, student_splits, strict=True):
            indices = torch.cat((teacher_batch, student_batch))
            yield indices[torch.randperm(indices.numel(), device=self.device)]

    @staticmethod
    def _flatten_saved_hidden_states(saved_hidden_states):
        if saved_hidden_states is None:
            return None
        flattened = tuple(state.flatten(0, 1) for state in saved_hidden_states)
        return flattened[0] if len(flattened) == 1 else flattened

    @staticmethod
    def _select_hidden_states(hidden_states: HiddenState, batch_idx: torch.Tensor) -> HiddenState:
        if hidden_states is None:
            return None
        if isinstance(hidden_states, tuple):
            return tuple(state[batch_idx] for state in hidden_states)
        return hidden_states[batch_idx]

    @staticmethod
    def _select_chunk_initial_states(
        saved_hidden_states,
        starts: torch.Tensor,
        envs: torch.Tensor,
    ) -> HiddenState:
        if saved_hidden_states is None:
            return None
        selected = tuple(state[starts, envs] for state in saved_hidden_states)
        return selected[0] if len(selected) == 1 else selected
