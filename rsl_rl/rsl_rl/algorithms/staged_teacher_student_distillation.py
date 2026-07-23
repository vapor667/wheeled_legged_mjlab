"""Warm-start and DAgger training for a frozen privileged representation teacher."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.models import DepthLinVelStudentActor
from rsl_rl.models.representation_velocity_predictor_actor_critic import (
    RepresentationVelocityPredictorActorCritic,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class StagedTeacherStudentDistillation(Distillation):
    """Train a depth student against a frozen privileged teacher in two explicit stages."""

    def __init__(
        self,
        student: DepthLinVelStudentActor,
        teacher: RepresentationVelocityPredictorActorCritic,
        storage: RolloutStorage,
        training_stage: str = "warm_start",
        num_learning_epochs: int = 1,
        gradient_length: int = 12,
        actor_learning_rate: float = 1.0e-3,
        encoder_learning_rate: float = 2.0e-4,
        latent_loss_coef: float = 1.0,
        lin_vel_loss_coef: float = 1.0,
        kl_loss_coef: float = 1.0,
        action_loss_coef: float = 0.2,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,
    ) -> None:
        del kwargs
        if training_stage not in {"warm_start", "dagger"}:
            raise ValueError("training_stage must be 'warm_start' or 'dagger'.")
        super().__init__(
            student,
            teacher,
            storage,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=actor_learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.training_stage = training_stage
        self.actor_learning_rate = actor_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.latent_loss_coef = latent_loss_coef
        self.lin_vel_loss_coef = lin_vel_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.action_loss_coef = action_loss_coef
        self._raw_teacher.requires_grad_(False)
        self._raw_teacher.eval()
        self._rollout_initial_hidden_state: torch.Tensor | None = None
        self._configure_optimizer(resolve_optimizer(optimizer))

    def _configure_optimizer(self, optimizer_cls: type[torch.optim.Optimizer]) -> None:
        actor_parameters = list(self._raw_student.actor_parameters())
        encoder_parameters = list(self._raw_student.encoder_parameters())
        if self.training_stage == "warm_start":
            for parameter in actor_parameters:
                parameter.requires_grad_(False)
            for parameter in encoder_parameters:
                parameter.requires_grad_(True)
            parameter_groups = [{"params": encoder_parameters, "lr": self.encoder_learning_rate}]
        else:
            for parameter in actor_parameters + encoder_parameters:
                parameter.requires_grad_(True)
            parameter_groups = [
                {"params": actor_parameters, "lr": self.actor_learning_rate},
                {"params": encoder_parameters, "lr": self.encoder_learning_rate},
            ]
        self.optimizer = optimizer_cls(parameter_groups)
        self.learning_rate = self.actor_learning_rate

    def initialize_student_from_teacher(self) -> None:
        """Initialize the frozen warm-start actor head after a teacher checkpoint is loaded."""
        self._raw_student.initialize_from_teacher(self._raw_teacher)
        self._configure_optimizer(type(self.optimizer))

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.storage.step == 0:
            hidden_state = self.student.get_hidden_state()
            self._rollout_initial_hidden_state = (
                hidden_state.detach().clone() if isinstance(hidden_state, torch.Tensor) else None
            )
        with torch.no_grad():
            teacher_actions = self.teacher.act_teacher(obs, stochastic_output=True)
            self.transition.privileged_actions = self.teacher.output_mean.detach()
        if self.training_stage == "warm_start":
            self.transition.actions = teacher_actions.detach()
        else:
            self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        del extras
        self.student.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        del obs

    def update(self) -> dict[str, float]:
        losses = {"latent": 0.0, "lin_vel": 0.0, "kl": 0.0, "action": 0.0, "behavior": 0.0}
        count = 0
        for _ in range(self.num_learning_epochs):
            hidden_state = self._clone_rollout_initial_hidden_state()
            accumulated_loss: torch.Tensor | None = None
            accumulated_steps = 0
            for batch in self.storage.generator():
                _, student_latent, predicted_lin_vel, next_hidden = self.student.forward_with_state(
                    batch.observations, hidden_state, stochastic_output=True
                )
                with torch.no_grad():
                    self.teacher.act_teacher(batch.observations, stochastic_output=True)
                    teacher_latent = self.teacher.get_privileged_latent(batch.observations)
                    teacher_mean = self.teacher.output_mean.detach()
                    teacher_params = tuple(parameter.detach() for parameter in self.teacher.output_distribution_params)
                    lin_vel_target = self.teacher.get_lin_vel_target(batch.observations)

                latent_loss = F.mse_loss(student_latent, teacher_latent)
                lin_vel_loss = F.mse_loss(predicted_lin_vel, lin_vel_target)
                action_loss = F.huber_loss(self.student.output_mean, teacher_mean)
                kl_loss = self.student.get_kl_divergence(
                    teacher_params, self.student.output_distribution_params
                ).mean()
                behavior_loss = (
                    self.latent_loss_coef * latent_loss
                    + self.lin_vel_loss_coef * lin_vel_loss
                    + self.action_loss_coef * action_loss
                )
                if self.training_stage == "dagger":
                    behavior_loss = behavior_loss + self.kl_loss_coef * kl_loss

                accumulated_loss = behavior_loss if accumulated_loss is None else accumulated_loss + behavior_loss
                accumulated_steps += 1
                count += 1
                losses["latent"] += latent_loss.item()
                losses["lin_vel"] += lin_vel_loss.item()
                losses["kl"] += kl_loss.item()
                losses["action"] += action_loss.item()
                losses["behavior"] += behavior_loss.item()

                hidden_state = self._reset_hidden_state(next_hidden, batch.dones.view(-1))
                if accumulated_steps == self.gradient_length:
                    self._optimizer_step(accumulated_loss)
                    accumulated_loss = None
                    accumulated_steps = 0
                    hidden_state = hidden_state.detach()

            if accumulated_loss is not None:
                self._optimizer_step(accumulated_loss)

        self.storage.clear()
        return {name: value / max(count, 1) for name, value in losses.items()}

    def _clone_rollout_initial_hidden_state(self) -> torch.Tensor | None:
        if self._rollout_initial_hidden_state is None:
            return None
        return self._rollout_initial_hidden_state.detach().clone()

    @staticmethod
    def _reset_hidden_state(hidden_state: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        return hidden_state.masked_fill(dones.to(dtype=torch.bool).view(-1, 1), 0.0)

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.student.detach_hidden_state()

    def train_mode(self) -> None:
        self.student.train()
        self.teacher.eval()

    def eval_mode(self) -> None:
        self.student.eval()
        self.teacher.eval()

    def save(self) -> dict:
        return {
            "student_state_dict": self._raw_student.state_dict(),
            "teacher_state_dict": self._raw_teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stage": self.training_stage,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            if "student_state_dict" in loaded_dict:
                # A warm-start checkpoint has one optimizer parameter group while
                # DAgger has two.  Carry model weights across the stage boundary,
                # but deliberately start a fresh optimizer in the new stage.
                same_stage = loaded_dict.get("training_stage") == self.training_stage
                load_cfg = {
                    "student": True,
                    "teacher": True,
                    "optimizer": same_stage,
                    "iteration": True,
                }
            else:
                load_cfg = {"teacher": True, "iteration": False}
        if load_cfg.get("actor"):
            load_cfg = {**load_cfg, "student": True}
        if load_cfg.get("student") and "student_state_dict" in loaded_dict:
            self._raw_student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            teacher_state = loaded_dict.get("teacher_state_dict", loaded_dict.get("actor_state_dict"))
            if teacher_state is None:
                raise KeyError("Checkpoint does not contain a teacher actor state dict.")
            self._raw_teacher.load_state_dict(teacher_state, strict=strict)
            self.teacher_loaded = True
            if "student_state_dict" not in loaded_dict:
                # Both warm-start and a manually started DAgger run must begin
                # from the teacher policy head, never from a random actor head.
                # A resumed warm-start checkpoint already contains this copy.
                self.initialize_student_from_teacher()
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> DepthLinVelStudentActor:
        return self._raw_student

    def compile(self, mode: str | None = None) -> None:
        self.student = compile_model(self._raw_student, mode)  # type: ignore[assignment]
        self.teacher = compile_model(self._raw_teacher, mode)  # type: ignore[assignment]

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "StagedTeacherStudentDistillation":
        algorithm_class: type[StagedTeacherStudentDistillation] = resolve_callable(
            cfg["algorithm"].pop("class_name")
        )
        teacher_class: type[RepresentationVelocityPredictorActorCritic] = resolve_callable(
            cfg["teacher"].pop("class_name")
        )
        student_class: type[DepthLinVelStudentActor] = resolve_callable(cfg["student"].pop("class_name"))
        required_sets = ["teacher", "student_history", "student_command", "student_depth"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], required_sets)
        cfg["algorithm"]["rnd_cfg"] = None
        cfg["algorithm"]["symmetry_cfg"] = None
        teacher = teacher_class(obs, cfg["obs_groups"], env.num_actions, **cfg["teacher"]).to(device)
        student = student_class(obs, cfg["obs_groups"], env.num_actions, **cfg["student"]).to(device)
        storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm = algorithm_class(
            student, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm
