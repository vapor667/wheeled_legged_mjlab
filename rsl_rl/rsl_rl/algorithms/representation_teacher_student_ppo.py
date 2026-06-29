# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config
from rsl_rl.models import RepresentationActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class RepresentationTeacherStudentPPO:
    """PPO with optional concurrent teacher/student rollout and representation alignment."""

    def __init__(
        self,
        model: RepresentationActorCritic,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        proprio_encoder_learning_rate: float = 0.001,
        num_proprio_encoder_substeps: int = 1,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        share_cnn_encoders: bool = False,
        teacher_student_ratio: float | None = None,
    ) -> None:
        if rnd_cfg is not None:
            raise ValueError("RND is not supported by RepresentationTeacherStudentPPO.")
        if symmetry_cfg is not None:
            raise ValueError("Symmetry augmentation is not supported by RepresentationTeacherStudentPPO.")
        if share_cnn_encoders:
            raise ValueError("CNN encoder sharing is not supported by RepresentationTeacherStudentPPO.")
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.actor = model.to(self.device)
        self.critic = self.actor
        self._raw_actor = self.actor
        self._raw_critic = self.actor

        optimizer_cls = resolve_optimizer(optimizer)
        self.optimizer = optimizer_cls(self.actor.ppo_parameters(), lr=learning_rate)  # type: ignore
        self.proprio_optimizer = optimizer_cls(
            self.actor.representation_parameters(), lr=proprio_encoder_learning_rate
        )  # type: ignore

        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.proprio_encoder_learning_rate = proprio_encoder_learning_rate
        self.num_proprio_encoder_substeps = num_proprio_encoder_substeps
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.rnd = None
        self.teacher_mask = self._make_teacher_mask(storage.num_envs, teacher_student_ratio)
        self._teacher_reward_sum = 0.0
        self._student_reward_sum = 0.0
        self._reward_sample_count = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        actor_hidden_state = self.actor.get_hidden_state()
        self.transition.hidden_states = (actor_hidden_state, None)
        if self.teacher_mask is None:
            self.transition.actions = self.actor.act_teacher(obs, stochastic_output=True).detach()
            self.transition.values = self.actor.evaluate_teacher(obs).detach()
        else:
            self.transition.teacher_mask = self.teacher_mask
            self.transition.actions = self.actor.act_mixed(
                obs,
                self.teacher_mask,
                hidden_state=actor_hidden_state,
                stochastic_output=True,
                update_hidden_state=True,
            ).detach()
            self.transition.values = self.actor.evaluate_mixed(
                obs,
                self.teacher_mask,
                hidden_state=actor_hidden_state,
            ).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.teacher_mask is not None:
            flat_rewards = rewards.view(-1)
            self._teacher_reward_sum += flat_rewards[self.teacher_mask].mean().item()
            self._student_reward_sum += flat_rewards[~self.teacher_mask].mean().item()
            self._reward_sample_count += 1
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        if self.critic is not self.actor:
            self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        if self.teacher_mask is None:
            last_values = self.actor.evaluate_teacher(obs).detach()
        else:
            last_values = self.actor.evaluate_mixed(
                obs,
                self.teacher_mask,
                hidden_state=self.actor.get_hidden_state(),
            ).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_representation_losses: dict[str, float] = {}

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            actor_hidden_state = batch.hidden_states[0]
            if batch.teacher_mask is None:
                self.actor.act_teacher(batch.observations, stochastic_output=True)
                values = self.actor.evaluate_teacher(batch.observations)
            else:
                self.actor.act_mixed(
                    batch.observations,
                    batch.teacher_mask,
                    hidden_state=actor_hidden_state,
                    stochastic_output=True,
                )
                values = self.actor.evaluate_mixed(
                    batch.observations,
                    batch.teacher_mask,
                    hidden_state=actor_hidden_state,
                )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor.ppo_parameters())
            nn.utils.clip_grad_norm_(self.actor.ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()

            representation_loss_values: dict[str, float] = {}
            for _ in range(self.num_proprio_encoder_substeps):
                representation_losses = self.actor.compute_representation_losses(
                    batch.observations,
                    hidden_state=actor_hidden_state,
                )
                representation_loss = representation_losses["representation_total"]
                self.proprio_optimizer.zero_grad()
                representation_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(self.actor.representation_parameters())
                nn.utils.clip_grad_norm_(self.actor.representation_parameters(), self.max_grad_norm)
                self.proprio_optimizer.step()
                for key, value in representation_losses.items():
                    representation_loss_values[key] = representation_loss_values.get(key, 0.0) + value.item()
            for key in representation_loss_values:
                representation_loss_values[key] /= self.num_proprio_encoder_substeps

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            for key, value in representation_loss_values.items():
                mean_representation_losses[key] = mean_representation_losses.get(key, 0.0) + value

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "representation": mean_representation_losses["representation_total"] / num_updates,
        }
        for key, value in mean_representation_losses.items():
            if key not in {"representation_total", "height_total"}:
                loss_dict[key] = value / num_updates
        if self.teacher_mask is not None and self._reward_sample_count > 0:
            loss_dict["CTS/teacher_mean_step_reward"] = self._teacher_reward_sum / self._reward_sample_count
            loss_dict["CTS/student_mean_step_reward"] = self._student_reward_sum / self._reward_sample_count
            self._teacher_reward_sum = 0.0
            self._student_reward_sum = 0.0
            self._reward_sample_count = 0
        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        self.actor.train()

    def eval_mode(self) -> None:
        self.actor.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "proprio_optimizer_state_dict": self.proprio_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor") or load_cfg.get("critic"):
            key = "actor_state_dict" if "actor_state_dict" in loaded_dict else "critic_state_dict"
            self._raw_actor.load_state_dict(loaded_dict[key], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "proprio_optimizer_state_dict" in loaded_dict:
                self.proprio_optimizer.load_state_dict(loaded_dict["proprio_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> RepresentationActorCritic:
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "RepresentationTeacherStudentPPO":
        alg_class: type[RepresentationTeacherStudentPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        model_class: type[RepresentationActorCritic] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic", "proprio_encoder", "privileged_encoder"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("RND is not supported by RepresentationTeacherStudentPPO.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("Symmetry augmentation is not supported by RepresentationTeacherStudentPPO.")
        cfg["algorithm"]["symmetry_cfg"] = None
        if cfg["algorithm"].get("share_cnn_encoders", False):
            raise ValueError("CNN encoder sharing is not supported by RepresentationTeacherStudentPPO.")
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        model = model_class(obs, cfg["obs_groups"], env.num_actions, **cfg["actor"]).to(device)
        print(f"Representation Actor-Critic Model: {model}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(model, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        model_params = [self._raw_actor.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])

    def reduce_parameters(self, parameters=None) -> None:
        all_params = list(self.actor.ppo_parameters() if parameters is None else parameters)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def _make_teacher_mask(
        self, num_envs: int, teacher_student_ratio: float | None
    ) -> torch.Tensor | None:
        if teacher_student_ratio is None:
            return None
        if teacher_student_ratio <= 0.0:
            raise ValueError(f"teacher_student_ratio must be positive, got {teacher_student_ratio}")
        if num_envs < 2:
            raise ValueError("Concurrent teacher-student training requires at least two environments")
        teacher_fraction = teacher_student_ratio / (teacher_student_ratio + 1.0)
        num_teacher_envs = min(max(int(num_envs * teacher_fraction), 1), num_envs - 1)
        teacher_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        teacher_mask[:num_teacher_envs] = True
        return teacher_mask
