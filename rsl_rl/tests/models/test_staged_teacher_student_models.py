"""Tests for the standalone staged teacher--student models."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import DepthLinVelStudentActor
from rsl_rl.models.representation_velocity_predictor_actor_critic import (
    RepresentationVelocityPredictorActorCritic,
)


NUM_ENVS = 4
NUM_ACTIONS = 2
HISTORY_LENGTH = 5
PROPRIO_DIM = 12
COMMAND_DIM = 3
LATENT_DIM = 4
DEPTH_SHAPE = (1, 32, 24)


def make_obs(include_privileged: bool = True) -> TensorDict:
    data = {
        "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
        "actor_command": torch.randn(NUM_ENVS, COMMAND_DIM),
        "depth_camera": torch.randn(NUM_ENVS, *DEPTH_SHAPE),
    }
    if include_privileged:
        data.update(
            {
                "lin_vel_target": torch.randn(NUM_ENVS, 3),
                "teacher_lin_vel": torch.randn(NUM_ENVS, 3),
                "critic": torch.randn(NUM_ENVS, 16),
                "privileged_encoder": torch.randn(NUM_ENVS, 11),
                "dynamics_context": torch.randn(NUM_ENVS, 2),
            }
        )
    return TensorDict(data, batch_size=[NUM_ENVS])


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


def make_teacher(obs: TensorDict | None = None) -> RepresentationVelocityPredictorActorCritic:
    return RepresentationVelocityPredictorActorCritic(
        make_obs() if obs is None else obs,
        obs_groups(),
        NUM_ACTIONS,
        hidden_dims=(16, 8),
        encoder_hidden_dims=(16, 8),
        latent_dim=LATENT_DIM,
        latent_dynamics_hidden_dims=(8,),
        latent_dynamics_horizons=(1,),
        distribution_cfg=distribution_cfg(),
    )


def make_student(obs: TensorDict | None = None) -> DepthLinVelStudentActor:
    return DepthLinVelStudentActor(
        make_obs() if obs is None else obs,
        obs_groups(),
        NUM_ACTIONS,
        hidden_dims=(16, 8),
        encoder_hidden_dims=(16, 8),
        latent_dim=LATENT_DIM,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        distribution_cfg=distribution_cfg(),
    )


def test_teacher_actor_uses_noisy_ground_truth_velocity_and_privileged_latent() -> None:
    obs = make_obs()
    obs["lin_vel_target"] = torch.full((NUM_ENVS, 3), 100.0)
    obs["teacher_lin_vel"] = torch.full((NUM_ENVS, 3), -2.0)
    teacher = make_teacher(obs)
    captured: dict[str, torch.Tensor] = {}

    def capture_actor(actor_obs: torch.Tensor, latent: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        del stochastic_output
        captured["actor_obs"] = actor_obs.detach().clone()
        captured["latent"] = latent.detach().clone()
        return torch.zeros(NUM_ENVS, NUM_ACTIONS)

    teacher._actor = capture_actor  # type: ignore[method-assign]
    actions = teacher.act_teacher(obs, stochastic_output=True)

    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert torch.equal(captured["actor_obs"][:, :3], obs["teacher_lin_vel"])
    assert not torch.equal(captured["actor_obs"][:, :3], obs["lin_vel_target"])
    assert torch.equal(captured["latent"], teacher.get_privileged_latent(obs))


def test_student_only_requires_deployable_observations_and_exports_recurrent_state() -> None:
    teacher = make_teacher()
    student = make_student()
    student.initialize_from_teacher(teacher)
    inference_obs = make_obs(include_privileged=False)

    actions, latent, predicted_lin_vel = student.forward_with_outputs(inference_obs)
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert latent.shape == (NUM_ENVS, LATENT_DIM)
    assert predicted_lin_vel.shape == (NUM_ENVS, 3)

    onnx_policy = student.as_onnx(verbose=False)
    hidden_state = torch.zeros(NUM_ENVS, 8)
    onnx_actions, onnx_lin_vel, next_hidden = onnx_policy(
        inference_obs["proprio_history"],
        inference_obs["actor_command"],
        inference_obs["depth_camera"],
        hidden_state,
    )
    assert onnx_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert onnx_lin_vel.shape == (NUM_ENVS, 3)
    assert next_hidden.shape == (NUM_ENVS, 8)
    assert onnx_policy.input_names == ["proprio_history", "actor_command", "depth", "hidden_state_in"]
    assert onnx_policy.output_names == ["actions", "predicted_lin_vel", "hidden_state_out"]


def test_functional_student_step_preserves_rollout_state_and_exposes_bptt_state() -> None:
    student = make_student()
    obs = make_obs(include_privileged=False)

    _, _, _, next_hidden = student.forward_with_state(obs)

    assert student.get_hidden_state() is None
    assert next_hidden.requires_grad
    assert next_hidden.shape == (NUM_ENVS, 8)


def test_student_initialization_copies_the_teacher_actor_head() -> None:
    teacher = make_teacher()
    student = make_student()

    student.initialize_from_teacher(teacher)

    for student_parameter, teacher_parameter in zip(
        student.actor_head.parameters(), teacher.actor_head.parameters(), strict=True
    ):
        assert torch.equal(student_parameter, teacher_parameter)


def test_teacher_predictor_supports_the_stage_one_autoregressive_rollout_loss() -> None:
    obs = make_obs()
    teacher = RepresentationVelocityPredictorActorCritic(
        obs,
        obs_groups(),
        NUM_ACTIONS,
        hidden_dims=(16, 8),
        encoder_hidden_dims=(16, 8),
        latent_dim=LATENT_DIM,
        latent_dynamics_hidden_dims=(8,),
        latent_dynamics_horizons=(1, 5, 10),
        distribution_cfg=distribution_cfg(),
    )
    latent = teacher.get_privileged_latent(obs)
    normalized_lin_vel = teacher.get_normalized_lin_vel_target(obs)
    action_sequence = torch.randn(3, NUM_ENVS, NUM_ACTIONS)

    predicted_latents, predicted_lin_vel = teacher.rollout_privileged_state(
        latent, normalized_lin_vel, action_sequence
    )

    assert predicted_latents.shape == (3, NUM_ENVS, LATENT_DIM)
    assert predicted_lin_vel.shape == (3, NUM_ENVS, 3)
