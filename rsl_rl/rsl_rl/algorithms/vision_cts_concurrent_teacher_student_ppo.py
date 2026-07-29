"""Concurrent teacher-student PPO used by the Vision-CTS baseline."""

# ruff: file-ignore[missing-type-kwargs, undocumented-public-method, undocumented-public-init]

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.representation_teacher_student_ppo import RepresentationTeacherStudentPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config
from rsl_rl.models.vision_cts_actor_critic import VisionCTSActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups


class VisionCTSConcurrentTeacherStudentPPO(RepresentationTeacherStudentPPO):
    """PPO with a fixed concurrent teacher/student environment partition."""

    def __init__(
        self,
        model: VisionCTSActorCritic,
        storage: RolloutStorage,
        representation_learning_rate: float = 1.0e-3,
        num_representation_epochs: int = 1,
        num_representation_mini_batches: int = 4,
        representation_chunk_length: int = 24,
        teacher_student_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            storage,
            proprio_encoder_learning_rate=representation_learning_rate,
            num_proprio_encoder_substeps=1,
            **kwargs,
        )
        if num_representation_epochs <= 0 or num_representation_mini_batches <= 0:
            raise ValueError("VisionCTS representation epochs and mini-batches must be positive")
        if representation_chunk_length <= 0:
            raise ValueError("VisionCTS representation_chunk_length must be positive")
        if teacher_student_ratio <= 0.0:
            raise ValueError("teacher_student_ratio must be positive")
        self.num_representation_epochs = num_representation_epochs
        self.num_representation_mini_batches = num_representation_mini_batches
        self.representation_chunk_length = representation_chunk_length
        self.teacher_student_ratio = teacher_student_ratio
        self.teacher_mask = self._make_teacher_mask(storage.num_envs, teacher_student_ratio, self.device)
        self._teacher_reward_sum = 0.0
        self._student_reward_sum = 0.0
        self._reward_sample_count = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        actor_hidden_state = self.actor.get_hidden_state()
        self.transition.hidden_states = (actor_hidden_state, None)
        self.transition.teacher_mask = self.teacher_mask
        output = self.actor.act_mixed(
            obs,
            self.teacher_mask,
            hidden_state=actor_hidden_state,
            stochastic_output=True,
            update_hidden_state=True,
            return_student_latent=True,
        )
        if not isinstance(output, tuple):
            raise RuntimeError("VisionCTS mixed rollout must return actions and student latents")
        self.transition.actions, self.transition.student_latent = (value.detach() for value in output)
        self.transition.values = self.actor.evaluate_mixed(
            obs,
            self.teacher_mask,
            hidden_state=actor_hidden_state,
            student_latent=self.transition.student_latent,
        ).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(
            parameter.detach() for parameter in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        flat_rewards = rewards.view(-1)
        self._teacher_reward_sum += flat_rewards[self.teacher_mask.view(-1)].mean().item()
        self._student_reward_sum += flat_rewards[~self.teacher_mask.view(-1)].mean().item()
        self._reward_sample_count += 1
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        storage = self.storage
        last_values = self.actor.evaluate_mixed(
            obs,
            self.teacher_mask,
            hidden_state=self.actor.get_hidden_state(),
        ).detach()
        advantage = 0
        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
            not_done = 1.0 - storage.dones[step].float()
            delta = storage.rewards[step] + not_done * self.gamma * next_values - storage.values[step]
            advantage = delta + not_done * self.gamma * self.lam * advantage
            storage.returns[step] = advantage + storage.values[step]
        storage.advantages = storage.returns - storage.values
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = self._normalize_grouped_advantages(storage.advantages, storage.teacher_masks)

    def update(self) -> dict[str, float]:
        losses = self._update_ppo_phase()
        representation_losses = self._update_representation_phase()
        losses["representation"] = representation_losses["representation_total"]
        for name, value in representation_losses.items():
            if name not in {"representation_total", "height_total"}:
                losses[name] = value
        if self._reward_sample_count:
            losses["CTS/teacher_mean_step_reward"] = self._teacher_reward_sum / self._reward_sample_count
            losses["CTS/student_mean_step_reward"] = self._student_reward_sum / self._reward_sample_count
        self._teacher_reward_sum = 0.0
        self._student_reward_sum = 0.0
        self._reward_sample_count = 0
        self.storage.clear()
        return losses

    def _update_ppo_phase(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = self._normalize_grouped_advantages(batch.advantages, batch.teacher_mask)
            if batch.teacher_mask is None or batch.student_latents is None:
                raise RuntimeError("VisionCTS PPO requires stored teacher masks and student latents")
            self.actor.act_mixed(
                batch.observations,
                batch.teacher_mask,
                stochastic_output=True,
                student_latent=batch.student_latents.detach(),
            )
            values = self.actor.evaluate_mixed(
                batch.observations,
                batch.teacher_mask,
                student_latent=batch.student_latents.detach(),
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            entropy = self.actor.output_entropy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(
                        batch.old_distribution_params, self.actor.output_distribution_params
                    ).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = self._grouped_objective(torch.max(surrogate, surrogate_clipped), batch.teacher_mask)
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = self._grouped_objective(
                    torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)),
                    batch.teacher_mask,
                )
            else:
                value_loss = self._grouped_objective((batch.returns - values).pow(2), batch.teacher_mask)
            entropy_loss = self._grouped_objective(entropy, batch.teacher_mask)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor.ppo_parameters())
            nn.utils.clip_grad_norm_(self.actor.ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_loss.item()
            updates += 1
        return {
            "value": mean_value_loss / updates,
            "surrogate": mean_surrogate_loss / updates,
            "entropy": mean_entropy / updates,
        }

    def _update_representation_phase(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        updates = 0
        generator = self.storage.representation_chunk_generator(
            self.num_representation_mini_batches,
            self.num_representation_epochs,
            self.representation_chunk_length,
            student_only=True,
        )
        for batch in generator:
            losses = self.actor.compute_representation_losses_sequence(
                batch.observations, batch.dones, hidden_state=batch.hidden_states[0]
            )
            self.proprio_optimizer.zero_grad()
            losses["representation_total"].backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor.representation_parameters())
            nn.utils.clip_grad_norm_(self.actor.representation_parameters(), self.max_grad_norm)
            self.proprio_optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item()
            updates += 1
        return {name: value / updates for name, value in totals.items()}

    @staticmethod
    def _make_teacher_mask(num_envs: int, ratio: float, device: str) -> torch.Tensor:
        if num_envs < 2:
            raise ValueError("VisionCTS requires at least two environments")
        teacher_count = round(num_envs * ratio / (1.0 + ratio))
        teacher_count = min(max(teacher_count, 1), num_envs - 1)
        mask = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        mask[torch.randperm(num_envs, device=device)[:teacher_count]] = True
        return mask

    @staticmethod
    def _grouped_objective(values: torch.Tensor, teacher_mask: torch.Tensor | None) -> torch.Tensor:
        if teacher_mask is None:
            return values.mean()
        mask = teacher_mask.view(-1)
        return 0.5 * (values.view(-1)[mask].mean() + values.view(-1)[~mask].mean())

    @staticmethod
    def _normalize_grouped_advantages(advantages: torch.Tensor, teacher_mask: torch.Tensor | None) -> torch.Tensor:
        if teacher_mask is None:
            return (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        normalized = advantages.clone()
        mask = teacher_mask.to(dtype=torch.bool)
        for group_mask in (mask, ~mask):
            values = advantages[group_mask]
            normalized[group_mask] = (values - values.mean()) / (values.std() + 1.0e-8)
        return normalized

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> VisionCTSConcurrentTeacherStudentPPO:
        algorithm_class: type[VisionCTSConcurrentTeacherStudentPPO] = resolve_callable(
            cfg["algorithm"].pop("class_name")
        )  # type: ignore
        model_class: type[VisionCTSActorCritic] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        default_sets = [
            "teacher_actor",
            "critic",
            "student_history",
            "actor_command",
            "privileged_encoder",
            "depth_encoder",
            "height_encoder",
        ]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("RND is not supported by VisionCTS")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("Symmetry augmentation is not supported by VisionCTS")
        cfg["algorithm"]["symmetry_cfg"] = None
        if cfg["algorithm"].get("share_cnn_encoders", False):
            raise ValueError("CNN encoder sharing is not supported by VisionCTS")
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        model = model_class(obs, cfg["obs_groups"], env.num_actions, **cfg["actor"]).to(device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm = algorithm_class(model, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        algorithm.actor = compile_model(algorithm._raw_actor, cfg.get("torch_compile_mode"))  # type: ignore
        algorithm.critic = algorithm.actor
        return algorithm
