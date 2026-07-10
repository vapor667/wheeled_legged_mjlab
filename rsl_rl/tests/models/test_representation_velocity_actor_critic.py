# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D103

"""Tests for the velocity representation actor-critic model."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import DepthRepresentationVelocityActorCritic, RepresentationVelocityActorCritic

NUM_ENVS = 4
PROPRIO_DIM = 28
COMMAND_DIM = 3
LIN_VEL_DIM = 3
CRITIC_DIM = 16
PRIVILEGED_DIM = 11
HISTORY_LENGTH = 5
LATENT_DIM = 4
NUM_ACTIONS = 2
DEPTH_SHAPE = (1, 32, 24)
AME_MAP_SHAPE = (5, 4)
AME_MAP_DIM = AME_MAP_SHAPE[0] * AME_MAP_SHAPE[1] * 3
AME_DOWNSAMPLED_TOKENS = ((AME_MAP_SHAPE[0] + 1) // 2) * ((AME_MAP_SHAPE[1] + 1) // 2)
AME_MODEL_DIM = 8
AME_NUM_HEADS = 2


def make_rep_obs(include_privileged: bool = True) -> TensorDict:
    data = {
        "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
        "actor_command": torch.randn(NUM_ENVS, COMMAND_DIM),
    }
    if include_privileged:
        data.update(
            {
                "lin_vel_target": torch.randn(NUM_ENVS, LIN_VEL_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "privileged_encoder": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
            }
        )
    return TensorDict(data, batch_size=[NUM_ENVS])


def make_model(obs: TensorDict | None = None, *, obs_normalization: bool = False) -> RepresentationVelocityActorCritic:
    obs = make_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder", "actor_command"],
    }
    return RepresentationVelocityActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        obs_normalization=obs_normalization,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def make_depth_rep_obs(include_privileged: bool = True, include_depth: bool = True) -> TensorDict:
    data = {
        "proprio_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM),
        "actor_command": torch.randn(NUM_ENVS, COMMAND_DIM),
    }
    if include_depth:
        data["depth_camera"] = torch.randn(NUM_ENVS, *DEPTH_SHAPE)
    if include_privileged:
        data.update(
            {
                "lin_vel_target": torch.randn(NUM_ENVS, LIN_VEL_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "privileged_encoder": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
            }
        )
    return TensorDict(data, batch_size=[NUM_ENVS])


def make_ame_depth_rep_obs(include_privileged: bool = True, include_depth: bool = True) -> TensorDict:
    obs = make_depth_rep_obs(include_privileged=False, include_depth=include_depth)
    if include_privileged:
        obs.update(
            {
                "lin_vel_target": torch.randn(NUM_ENVS, LIN_VEL_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "privileged_encoder": torch.randn(NUM_ENVS, AME_MAP_DIM),
                "privileged_query": torch.randn(NUM_ENVS, PROPRIO_DIM + COMMAND_DIM),
            }
        )
    return obs


def make_depth_model(obs: TensorDict | None = None) -> DepthRepresentationVelocityActorCritic:
    obs = make_depth_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder", "actor_command"],
        "depth_encoder": ["depth_camera"],
    }
    return DepthRepresentationVelocityActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def make_ame_depth_model(obs: TensorDict | None = None) -> DepthRepresentationVelocityActorCritic:
    obs = make_ame_depth_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder"],
        "privileged_query": ["privileged_query"],
        "depth_encoder": ["depth_camera"],
    }
    return DepthRepresentationVelocityActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        ame_map_scan_shape=(*AME_MAP_SHAPE, 3),
        ame_d_model=AME_MODEL_DIM,
        ame_num_heads=AME_NUM_HEADS,
        ame_return_attention_in_eval=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def make_ame_global_depth_model(obs: TensorDict | None = None) -> DepthRepresentationVelocityActorCritic:
    obs = make_ame_depth_rep_obs() if obs is None else obs
    obs_groups = {
        "proprio_history": ["proprio_history"],
        "actor_command": ["actor_command"],
        "lin_vel_target": ["lin_vel_target"],
        "critic": ["critic"],
        "privileged_encoder": ["privileged_encoder"],
        "privileged_query": ["privileged_query"],
        "depth_encoder": ["depth_camera"],
    }
    return DepthRepresentationVelocityActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        latent_dim=LATENT_DIM,
        depth_feature_dim=8,
        depth_gru_hidden_dim=8,
        depth_channels=(4, 4),
        ame_map_scan_shape=(*AME_MAP_SHAPE, 3),
        ame_d_model=AME_MODEL_DIM,
        ame_num_heads=AME_NUM_HEADS,
        ame_attach_global_context=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def test_teacher_student_value_and_velocity_paths_have_expected_shapes() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    teacher_actions = model.act_teacher(obs, stochastic_output=True)
    student_actions = model(obs)
    values = model.evaluate_teacher(obs)
    privileged_latent = model.get_privileged_latent(obs)
    proprio_latent, predicted_lin_vel = model.get_proprio_outputs(obs)

    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert values.shape == (NUM_ENVS, 1)
    assert privileged_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert proprio_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert predicted_lin_vel.shape == (NUM_ENVS, LIN_VEL_DIM)


def test_current_proprio_is_latest_frame_of_history() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    assert torch.equal(model.get_current_proprio(obs), obs["proprio_history"][:, -1, :])
    expected_proprio_obs = torch.cat(
        (obs["proprio_history"].flatten(start_dim=1), obs["actor_command"]),
        dim=-1,
    )
    assert torch.equal(model.get_proprio_obs(obs), expected_proprio_obs)


def test_teacher_actor_uses_predicted_velocity_and_privileged_latent() -> None:
    obs = make_rep_obs()
    obs["lin_vel_target"] = torch.full((NUM_ENVS, LIN_VEL_DIM), 100.0)
    obs["proprio_history"][:, -1, :] = torch.arange(
        NUM_ENVS * PROPRIO_DIM,
        dtype=obs["proprio_history"].dtype,
    ).reshape(NUM_ENVS, PROPRIO_DIM)
    model = make_model(obs)
    captured: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        expected_predicted_lin_vel = model.get_predicted_lin_vel(obs)

    def capture_actor(actor_obs: torch.Tensor, latent: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        captured["actor_obs"] = actor_obs.detach().clone()
        captured["latent"] = latent.detach().clone()
        captured["stochastic_output"] = torch.tensor(stochastic_output)
        return torch.zeros(NUM_ENVS, NUM_ACTIONS)

    model._actor = capture_actor  # type: ignore[method-assign]

    actions = model.act_teacher(obs, stochastic_output=True)

    lin_vel_slice = captured["actor_obs"][:, :LIN_VEL_DIM]
    proprio_slice = captured["actor_obs"][:, LIN_VEL_DIM : LIN_VEL_DIM + PROPRIO_DIM]
    command_slice = captured["actor_obs"][:, LIN_VEL_DIM + PROPRIO_DIM :]
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert torch.equal(lin_vel_slice, expected_predicted_lin_vel)
    assert not torch.equal(lin_vel_slice, obs["lin_vel_target"])
    assert torch.equal(proprio_slice, obs["proprio_history"][:, -1, :])
    assert torch.equal(command_slice, obs["actor_command"])
    assert torch.equal(captured["latent"], model.get_privileged_latent(obs))
    assert captured["stochastic_output"].item() is True


def test_student_losses_only_update_student_encoder_side() -> None:
    obs = make_rep_obs()
    model = make_model(obs)

    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert any(param.grad is not None for param in model.student_latent_head.parameters())
    assert any(param.grad is not None for param in model.lin_vel_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_normalization_update_supports_velocity_inputs() -> None:
    obs = make_rep_obs()
    model = make_model(obs, obs_normalization=True)

    model.update_normalization(obs)
    teacher_actions = model.act_teacher(obs)
    student_actions = model(obs)

    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)


def test_student_inference_only_requires_history_and_command() -> None:
    model = make_model(make_rep_obs())
    inference_obs = make_rep_obs(include_privileged=False)

    actions = model(inference_obs)

    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)


def test_exported_student_policy_uses_history_and_command_inputs() -> None:
    model = make_model(make_rep_obs())
    proprio_history = torch.randn(NUM_ENVS, HISTORY_LENGTH, PROPRIO_DIM)
    actor_command = torch.randn(NUM_ENVS, COMMAND_DIM)
    inference_obs = TensorDict(
        {"proprio_history": proprio_history, "actor_command": actor_command},
        batch_size=[NUM_ENVS],
    )
    expected_actions = model(inference_obs)
    expected_predicted_lin_vel = model.get_predicted_lin_vel(inference_obs)

    jit_policy = model.as_jit()
    onnx_policy = model.as_onnx(verbose=False)
    jit_actions, jit_predicted_lin_vel = jit_policy(proprio_history, actor_command)
    onnx_actions, onnx_predicted_lin_vel = onnx_policy(proprio_history, actor_command)

    assert torch.allclose(jit_actions, expected_actions)
    assert torch.allclose(jit_predicted_lin_vel, expected_predicted_lin_vel)
    assert torch.allclose(onnx_actions, expected_actions)
    assert torch.allclose(onnx_predicted_lin_vel, expected_predicted_lin_vel)
    assert onnx_policy.input_names == ["proprio_history", "actor_command"]
    assert onnx_policy.output_names == ["actions", "predicted_lin_vel"]
    assert onnx_policy.get_dummy_inputs()[0].shape == (1, HISTORY_LENGTH, PROPRIO_DIM)
    assert onnx_policy.get_dummy_inputs()[1].shape == (1, COMMAND_DIM)


def test_depth_teacher_student_value_and_velocity_paths_have_expected_shapes() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)

    teacher_actions = model.act_teacher(obs, stochastic_output=True)
    student_actions = model(obs)
    values = model.evaluate_teacher(obs)
    privileged_latent = model.get_privileged_latent(obs)
    proprio_latent, predicted_lin_vel = model.get_proprio_outputs(obs)

    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert values.shape == (NUM_ENVS, 1)
    assert privileged_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert proprio_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert predicted_lin_vel.shape == (NUM_ENVS, LIN_VEL_DIM)


def test_depth_ame_teacher_encoder_paths_have_expected_shapes() -> None:
    obs = make_ame_depth_rep_obs()
    model = make_ame_depth_model(obs)

    teacher_actions = model.act_teacher(obs, stochastic_output=True)
    values = model.evaluate_teacher(obs)
    privileged_latent = model.get_privileged_latent(obs)
    query_obs = model.get_privileged_query_obs(obs)

    assert model.use_ame_teacher_encoder is True
    assert model.privileged_encoder.map_scan_shape == (*AME_MAP_SHAPE, 3)
    assert model.privileged_encoder.use_xyz_cnn_input is True
    assert model.privileged_encoder.cnn_downsample is True
    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert values.shape == (NUM_ENVS, 1)
    assert privileged_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert query_obs.shape == (NUM_ENVS, PROPRIO_DIM + COMMAND_DIM)


def test_depth_ame_teacher_query_uses_clean_privileged_query_group() -> None:
    obs = make_ame_depth_rep_obs()
    model = make_ame_depth_model(obs)

    query_obs = model.get_privileged_query_obs(obs)

    assert torch.equal(query_obs, obs["privileged_query"])


def test_depth_ame_eval_path_caches_attention_weights() -> None:
    obs = make_ame_depth_rep_obs()
    model = make_ame_depth_model(obs)

    model.train()
    model.get_privileged_latent(obs)
    assert model.privileged_encoder.last_attention_weights is None
    assert model.privileged_encoder.last_attention_points is None

    model.eval()
    model.get_privileged_latent(obs)

    attention_weights = model.privileged_encoder.last_attention_weights
    attention_points = model.privileged_encoder.last_attention_points
    assert attention_weights is not None
    assert attention_points is not None
    assert attention_weights.shape == (
        NUM_ENVS,
        AME_NUM_HEADS,
        1,
        AME_DOWNSAMPLED_TOKENS,
    )
    assert attention_points.shape == (NUM_ENVS, AME_DOWNSAMPLED_TOKENS, 3)


def test_depth_ame_global_context_path_has_expected_shape() -> None:
    obs = make_ame_depth_rep_obs()
    model = make_ame_global_depth_model(obs)

    privileged_latent = model.get_privileged_latent(obs)

    assert model.privileged_encoder.attach_global_context is True
    assert privileged_latent.shape == (NUM_ENVS, LATENT_DIM)


def test_depth_student_hidden_state_persists_and_resets_per_environment() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)

    model(obs)
    first_state = model.get_hidden_state().clone()
    model(obs)
    second_state = model.get_hidden_state().clone()
    assert not torch.equal(first_state, second_state)

    model.reset(torch.tensor([True, False, False, True]))
    reset_state = model.get_hidden_state()
    assert torch.equal(reset_state[[0, 3]], torch.zeros_like(reset_state[[0, 3]]))
    assert torch.equal(reset_state[[1, 2]], second_state[[1, 2]])

    model.reset()
    assert model.get_hidden_state() is None


def test_depth_student_losses_update_depth_and_student_encoder_side() -> None:
    obs = make_depth_rep_obs()
    model = make_depth_model(obs)

    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.depth_encoder.parameters())
    assert any(param.grad is not None for param in model.depth_gru.parameters())
    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert any(param.grad is not None for param in model.student_latent_head.parameters())
    assert any(param.grad is not None for param in model.lin_vel_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_ame_student_losses_detach_teacher_encoder_side() -> None:
    obs = make_ame_depth_rep_obs()
    model = make_ame_depth_model(obs)

    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses(obs)
    model.zero_grad()
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.depth_encoder.parameters())
    assert any(param.grad is not None for param in model.depth_gru.parameters())
    assert any(param.grad is not None for param in model.proprio_encoder.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_ame_critic_does_not_update_teacher_encoder_side() -> None:
    obs = make_ame_depth_rep_obs()
    model = make_ame_depth_model(obs)

    value_loss = model.evaluate_teacher(obs).pow(2).mean()
    model.zero_grad()
    value_loss.backward()

    assert any(param.grad is not None for param in model.critic_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_sequence_student_losses_use_continuous_depth_state() -> None:
    model = make_depth_model()
    time_steps = 3
    step_observations = [make_depth_rep_obs() for _ in range(time_steps)]
    obs = TensorDict(
        {
            key: torch.stack([step_obs[key] for step_obs in step_observations])
            for key in step_observations[0].keys()
        },
        batch_size=[time_steps, NUM_ENVS],
    )
    dones = torch.zeros(time_steps, NUM_ENVS, 1)
    dones[1, 0] = 1.0
    hidden_state = torch.zeros(NUM_ENVS, model.depth_gru_hidden_dim)

    model.zero_grad()
    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses_sequence(
        obs,
        dones,
        hidden_state,
    )
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.depth_gru.parameters())
    assert all(param.grad is None for param in model.actor_head.parameters())
    assert all(param.grad is None for param in model.critic_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_ame_sequence_student_losses_use_attention_teacher_target() -> None:
    model = make_ame_depth_model()
    time_steps = 3
    step_observations = [make_ame_depth_rep_obs() for _ in range(time_steps)]
    obs = TensorDict(
        {
            key: torch.stack([step_obs[key] for step_obs in step_observations])
            for key in step_observations[0].keys()
        },
        batch_size=[time_steps, NUM_ENVS],
    )
    dones = torch.zeros(time_steps, NUM_ENVS, 1)
    hidden_state = torch.zeros(NUM_ENVS, model.depth_gru_hidden_dim)

    model.zero_grad()
    student_loss, representation_loss, lin_vel_loss = model.compute_student_losses_sequence(
        obs,
        dones,
        hidden_state,
    )
    student_loss.backward()

    assert torch.allclose(student_loss, representation_loss + lin_vel_loss)
    assert any(param.grad is not None for param in model.depth_gru.parameters())
    assert all(param.grad is None for param in model.actor_head.parameters())
    assert all(param.grad is None for param in model.critic_head.parameters())
    assert all(param.grad is None for param in model.privileged_encoder.parameters())


def test_depth_student_inference_requires_depth_observations() -> None:
    model = make_depth_model()
    inference_obs = make_depth_rep_obs(include_privileged=False, include_depth=False)

    try:
        model(inference_obs)
    except ValueError as exc:
        assert "depth observation group" in str(exc)
    else:
        raise AssertionError("student inference without depth should fail")


def test_depth_onnx_wrapper_matches_policy_outputs() -> None:
    model = make_depth_model()
    model.eval()
    obs = make_depth_rep_obs()
    hidden_state = torch.zeros(NUM_ENVS, model.depth_gru_hidden_dim)
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()

    with torch.inference_mode():
        expected_actions = model(obs, hidden_state=hidden_state)
        expected_predicted_lin_vel = model.get_proprio_outputs(obs, hidden_state=hidden_state)[1]
        actions, predicted_lin_vel, hidden_state_out = onnx_model(
            obs["proprio_history"],
            obs["actor_command"],
            obs["depth_camera"],
            hidden_state,
        )

    assert torch.allclose(actions, expected_actions, atol=1e-6)
    assert torch.allclose(predicted_lin_vel, expected_predicted_lin_vel, atol=1e-6)
    assert hidden_state_out.shape == hidden_state.shape
    assert onnx_model.input_names == ["proprio_history", "actor_command", "depth", "hidden_state_in"]
    assert onnx_model.output_names == ["actions", "predicted_lin_vel", "hidden_state_out"]
    assert onnx_model.get_dummy_inputs()[2].shape == (1, *DEPTH_SHAPE)


def test_depth_jit_wrapper_scripts_and_runs_single_robot_policy() -> None:
    model = make_depth_model()
    obs = make_depth_rep_obs()

    jit_model = torch.jit.script(model.as_jit())
    actions, predicted_lin_vel = jit_model(
        obs["proprio_history"][:1],
        obs["actor_command"][:1],
        obs["depth_camera"][:1],
    )

    assert actions.shape == (1, NUM_ACTIONS)
    assert predicted_lin_vel.shape == (1, LIN_VEL_DIM)
