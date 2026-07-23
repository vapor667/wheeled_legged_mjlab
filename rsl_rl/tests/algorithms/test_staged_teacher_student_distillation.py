"""Tests for stage-2 warm-start and stage-3 pure-DAgger distillation."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.staged_teacher_student_distillation import StagedTeacherStudentDistillation
from rsl_rl.models import DepthLinVelStudentActor
from rsl_rl.models.representation_velocity_predictor_actor_critic import (
    RepresentationVelocityPredictorActorCritic,
)
from rsl_rl.storage import RolloutStorage


NUM_ENVS = 4
NUM_ACTIONS = 2
NUM_STEPS = 4


def make_obs() -> TensorDict:
    return TensorDict(
        {
            "proprio_history": torch.randn(NUM_ENVS, 5, 12),
            "actor_command": torch.randn(NUM_ENVS, 3),
            "lin_vel_target": torch.randn(NUM_ENVS, 3),
            "teacher_lin_vel": torch.randn(NUM_ENVS, 3),
            "critic": torch.randn(NUM_ENVS, 16),
            "privileged_encoder": torch.randn(NUM_ENVS, 11),
            "dynamics_context": torch.randn(NUM_ENVS, 2),
            "depth_camera": torch.randn(NUM_ENVS, 1, 32, 24),
        },
        batch_size=[NUM_ENVS],
    )


def obs_groups() -> dict[str, list[str]]:
    return {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "teacher_lin_vel": ["teacher_lin_vel"],
        "critic": ["critic", "dynamics_context"],
        "privileged_encoder": ["privileged_encoder", "dynamics_context"],
        "teacher": ["critic", "dynamics_context"],
        "student_history": ["proprio_history"],
        "student_command": ["actor_command"],
        "student_depth": ["depth_camera"],
    }


def distribution_cfg() -> dict:
    return {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


def make_teacher(obs: TensorDict) -> RepresentationVelocityPredictorActorCritic:
    return RepresentationVelocityPredictorActorCritic(
        obs,
        obs_groups(),
        NUM_ACTIONS,
        hidden_dims=(16, 8),
        encoder_hidden_dims=(16, 8),
        latent_dim=4,
        latent_dynamics_hidden_dims=(8,),
        latent_dynamics_horizons=(1,),
        distribution_cfg=distribution_cfg(),
    )


def make_student(obs: TensorDict) -> DepthLinVelStudentActor:
    return DepthLinVelStudentActor(
        obs,
        obs_groups(),
        NUM_ACTIONS,
        hidden_dims=(16, 8),
        encoder_hidden_dims=(16, 8),
        latent_dim=4,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        distribution_cfg=distribution_cfg(),
    )


def make_algorithm(training_stage: str) -> tuple[StagedTeacherStudentDistillation, object]:
    torch.manual_seed(7)
    obs = make_obs()
    teacher = make_teacher(obs)
    student = make_student(obs)
    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    algorithm = StagedTeacherStudentDistillation(
        student,
        teacher,
        storage,
        training_stage=training_stage,
        gradient_length=2,
        actor_learning_rate=1.0e-3,
        encoder_learning_rate=1.0e-3,
    )
    algorithm.initialize_student_from_teacher()
    return algorithm, obs


def fill_rollout(algorithm: StagedTeacherStudentDistillation, obs) -> None:
    for step in range(NUM_STEPS):
        algorithm.act(obs)
        algorithm.process_env_step(
            obs,
            rewards=torch.ones(NUM_ENVS),
            dones=torch.tensor([False, step == 1, False, False]),
            extras={},
        )


def any_parameter_changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], parameter) for name, parameter in module.named_parameters())


def test_warm_start_freezes_teacher_and_actor_while_updating_student_encoder() -> None:
    algorithm, obs = make_algorithm("warm_start")
    actor_before = {name: parameter.detach().clone() for name, parameter in algorithm.student.actor_head.named_parameters()}
    encoder_before = {
        name: parameter.detach().clone() for name, parameter in algorithm.student.proprio_encoder.named_parameters()
    }
    teacher_before = {name: parameter.detach().clone() for name, parameter in algorithm.teacher.named_parameters()}

    fill_rollout(algorithm, obs)
    losses = algorithm.update()

    assert all(not parameter.requires_grad for parameter in algorithm.student.actor_parameters())
    assert not any_parameter_changed(actor_before, algorithm.student.actor_head)
    assert any_parameter_changed(encoder_before, algorithm.student.proprio_encoder)
    assert not any_parameter_changed(teacher_before, algorithm.teacher)
    assert losses["latent"] > 0.0
    assert losses["lin_vel"] > 0.0
    assert losses["behavior"] > 0.0


def test_dagger_unfreezes_student_actor_and_keeps_all_auxiliary_losses() -> None:
    algorithm, obs = make_algorithm("dagger")
    actor_before = {name: parameter.detach().clone() for name, parameter in algorithm.student.actor_head.named_parameters()}
    teacher_before = {name: parameter.detach().clone() for name, parameter in algorithm.teacher.named_parameters()}

    fill_rollout(algorithm, obs)
    losses = algorithm.update()

    assert all(parameter.requires_grad for parameter in algorithm.student.actor_parameters())
    assert any_parameter_changed(actor_before, algorithm.student.actor_head)
    assert not any_parameter_changed(teacher_before, algorithm.teacher)
    assert losses["latent"] > 0.0
    assert losses["lin_vel"] > 0.0
    assert losses["kl"] >= 0.0
    assert losses["action"] > 0.0


def test_dagger_can_load_a_warm_start_checkpoint_without_reusing_optimizer_state() -> None:
    warm_start, obs = make_algorithm("warm_start")
    fill_rollout(warm_start, obs)
    warm_start.update()
    checkpoint = warm_start.save()

    dagger, _ = make_algorithm("dagger")
    load_iteration = dagger.load(checkpoint, load_cfg=None, strict=True)

    assert load_iteration is True
    assert dagger.teacher_loaded
    assert len(dagger.optimizer.param_groups) == 2
    for student_parameter, saved_parameter in zip(
        dagger.student.parameters(), checkpoint["student_state_dict"].values(), strict=True
    ):
        assert torch.equal(student_parameter, saved_parameter)


def test_teacher_only_load_initializes_the_dagger_actor_from_the_teacher() -> None:
    dagger, _ = make_algorithm("dagger")
    teacher_checkpoint = {"actor_state_dict": dagger.teacher.state_dict()}
    for parameter in dagger.student.actor_head.parameters():
        parameter.data.zero_()

    dagger.load(teacher_checkpoint, load_cfg={"teacher": True}, strict=True)

    for student_parameter, teacher_parameter in zip(
        dagger.student.actor_head.parameters(), dagger.teacher.actor_head.parameters(), strict=True
    ):
        assert torch.equal(student_parameter, teacher_parameter)
