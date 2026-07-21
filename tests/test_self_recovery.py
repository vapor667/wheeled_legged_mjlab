from __future__ import annotations

import inspect
import math
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import wheeled_legged_mjlab  # noqa: F401
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    DEPTH_CAMERA_NAME,
    RECOVERY_FALLEN_FRACTION,
    RECOVERY_START_ITERATION,
    RECOVERY_START_STEP,
    RECOVERY_STEPS_PER_ITERATION,
    RECOVERY_UPRIGHT_GATE_HI,
    wf_tron1b_flat_env_cfg,
    wf_tron1b_flat_recovery_env_cfg,
    wf_tron1b_rough_recovery_env_cfg,
    wf_tron1b_rough_rep_ts_lin_vel_depth_recovery_env_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.rl_cfg import (
    wf_tron1b_ppo_runner_cfg,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.events import (
    reset_root_state_partial_fallen,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.metrics import (
    recovery_success_rate,
    time_to_recover,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.rewards import (
    _apply_upright_gate,
    _upright_gate,
    righting_progress,
    upward,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.terminations import (
    bad_orientation_until_step,
    illegal_contact_until_step,
)


def _orientation_env(projected_gravity_z: list[float]) -> SimpleNamespace:
    projected_gravity = torch.zeros(len(projected_gravity_z), 3)
    projected_gravity[:, 2] = torch.tensor(projected_gravity_z)
    robot = SimpleNamespace(
        data=SimpleNamespace(projected_gravity_b=projected_gravity)
    )
    return SimpleNamespace(
        num_envs=len(projected_gravity_z),
        device="cpu",
        step_dt=0.1,
        common_step_counter=0,
        scene={"robot": robot},
    )


def test_signed_upright_rewards_distinguish_inversion() -> None:
    env = _orientation_env([-1.0, 0.0, 1.0])

    assert torch.equal(upward(env), torch.tensor([4.0, 1.0, 0.0]))
    assert torch.allclose(
        _upright_gate(env, hi=RECOVERY_UPRIGHT_GATE_HI),
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_recovery_starts_after_5000_runner_iterations() -> None:
    runner_cfg = wf_tron1b_ppo_runner_cfg()

    assert RECOVERY_START_ITERATION == 5_000
    assert RECOVERY_STEPS_PER_ITERATION == runner_cfg.num_steps_per_env == 24
    assert RECOVERY_START_STEP == 120_000


def test_upright_gate_and_upward_switch_at_recovery_boundary() -> None:
    env = _orientation_env([0.0])
    value = torch.ones(1)

    env.common_step_counter = RECOVERY_START_STEP - 1
    assert _apply_upright_gate(
        value,
        env,
        RECOVERY_UPRIGHT_GATE_HI,
        upright_gate_start_step=RECOVERY_START_STEP,
    ).item() == 1.0
    assert upward(env, recovery_start_step=RECOVERY_START_STEP).item() == 0.0

    env.common_step_counter = RECOVERY_START_STEP
    assert _apply_upright_gate(
        value,
        env,
        RECOVERY_UPRIGHT_GATE_HI,
        upright_gate_start_step=RECOVERY_START_STEP,
    ).item() == 0.0
    assert upward(env, recovery_start_step=RECOVERY_START_STEP).item() == 1.0


def test_righting_progress_rewards_only_positive_change_and_resets() -> None:
    env = _orientation_env([1.0])
    term = righting_progress(None, env)

    assert term(env).item() == 0.0
    env.scene["robot"].data.projected_gravity_b[:, 2] = 0.5
    assert term(env, max_progress=0.1).item() == 1.0
    env.scene["robot"].data.projected_gravity_b[:, 2] = 0.8
    assert term(env, max_progress=0.1).item() == 0.0

    term.reset(torch.tensor([0]))
    assert term(env).item() == 0.0


def test_partial_fallen_reset_splits_requested_environment_ids() -> None:
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        common_step_counter=RECOVERY_START_STEP,
    )
    env_ids = torch.tensor([4, 6, 8, 10])
    normal_pose = {"yaw": (-math.pi, math.pi)}
    fallen_pose = {"roll": (-math.pi, math.pi)}
    asset_cfg = SimpleNamespace(name="robot")

    with (
        patch(
            "wheeled_legged_mjlab.tasks.velocity.mdp.events.torch.rand",
            return_value=torch.tensor([0.1, 0.9, 0.2, 0.8]),
        ),
        patch(
            "wheeled_legged_mjlab.tasks.velocity.mdp.events.reset_root_state_uniform"
        ) as reset_mock,
    ):
        reset_root_state_partial_fallen(
            env,
            env_ids,
            pose_range=normal_pose,
            fallen_pose_range=fallen_pose,
            velocity_range={"x": (-0.3, 0.3)},
            fallen_velocity_range={"x": (-0.5, 0.5)},
            fallen_fraction=0.3,
            recovery_start_step=RECOVERY_START_STEP,
            asset_cfg=asset_cfg,
        )

    assert reset_mock.call_count == 2
    upright_call, fallen_call = reset_mock.call_args_list
    assert upright_call.args[1].tolist() == [6, 10]
    assert upright_call.kwargs["pose_range"] == normal_pose
    assert fallen_call.args[1].tolist() == [4, 8]
    assert fallen_call.kwargs["pose_range"] == fallen_pose


def test_partial_fallen_reset_is_disabled_before_recovery_start() -> None:
    env = SimpleNamespace(num_envs=3, device="cpu", common_step_counter=119_999)
    env_ids = torch.arange(3)

    with patch(
        "wheeled_legged_mjlab.tasks.velocity.mdp.events.reset_root_state_uniform"
    ) as reset_mock:
        reset_root_state_partial_fallen(
            env,
            env_ids,
            pose_range={"yaw": (-math.pi, math.pi)},
            fallen_pose_range={"roll": (-math.pi, math.pi)},
            fallen_fraction=1.0,
            recovery_start_step=RECOVERY_START_STEP,
        )

    assert reset_mock.call_count == 1
    assert reset_mock.call_args.args[1].tolist() == [0, 1, 2]
    assert reset_mock.call_args.kwargs["pose_range"] == {
        "yaw": (-math.pi, math.pi)
    }


def test_recovery_metrics_report_upright_state_and_first_recovery_time() -> None:
    env = _orientation_env([1.0, -1.0])
    metric = time_to_recover(None, env)

    assert recovery_success_rate(env).tolist() == [0.0, 1.0]
    assert torch.allclose(metric(env), torch.tensor([0.1, 0.0]))
    env.scene["robot"].data.projected_gravity_b[0, 2] = -1.0
    assert torch.allclose(metric(env), torch.tensor([0.2, 0.0]))
    assert torch.allclose(metric(env), torch.tensor([0.2, 0.0]))

    metric.reset(torch.tensor([0]))
    assert metric(env)[0].item() == 0.0


def test_fall_and_illegal_contact_stop_resetting_at_recovery_start() -> None:
    env = _orientation_env([0.0])
    env.scene["illegal_ground_contact"] = SimpleNamespace(
        data=SimpleNamespace(
            force_history=None,
            found=torch.tensor([[True]]),
        )
    )

    env.common_step_counter = RECOVERY_START_STEP - 1
    assert bad_orientation_until_step(
        env,
        limit_angle=math.radians(65.0),
        deactivate_after_step=RECOVERY_START_STEP,
    ).item()
    assert illegal_contact_until_step(
        env,
        sensor_name="illegal_ground_contact",
        deactivate_after_step=RECOVERY_START_STEP,
    ).item()

    env.common_step_counter = RECOVERY_START_STEP
    assert not bad_orientation_until_step(
        env,
        limit_angle=math.radians(65.0),
        deactivate_after_step=RECOVERY_START_STEP,
    ).item()
    assert not illegal_contact_until_step(
        env,
        sensor_name="illegal_ground_contact",
        deactivate_after_step=RECOVERY_START_STEP,
    ).item()


def test_recovery_configs_are_isolated_and_manager_terms_are_wired() -> None:
    baseline = wf_tron1b_flat_env_cfg()
    flat = wf_tron1b_flat_recovery_env_cfg()
    rough = wf_tron1b_rough_recovery_env_cfg()
    play = wf_tron1b_flat_recovery_env_cfg(play=True)

    assert "fell_over" in baseline.terminations
    assert "illegal_contact" in baseline.terminations
    assert "upward" not in baseline.rewards
    assert set(flat.terminations) == {
        "non_finite_physics",
        "time_out",
        "fell_over",
        "illegal_contact",
    }
    assert flat.terminations["fell_over"].func is bad_orientation_until_step
    assert flat.terminations["illegal_contact"].func is illegal_contact_until_step
    assert (
        flat.terminations["fell_over"].params["deactivate_after_step"]
        == RECOVERY_START_STEP
    )
    assert "fell_over_limit_angle" in flat.curriculum
    assert "recovery_phase" in flat.curriculum
    assert "terrain_levels" in rough.curriculum
    assert rough.terminations["out_of_terrain_bounds"].params["margin"] == 1.5
    assert flat.events["reset_base"].params["fallen_fraction"] == (
        RECOVERY_FALLEN_FRACTION
    )
    assert play.events["reset_base"].params["fallen_fraction"] == 1.0
    assert flat.events["reset_base"].params["recovery_start_step"] == (
        RECOVERY_START_STEP
    )
    assert flat.events["reset_leg_joints"].params["position_range"] == (-0.3, 0.5)
    assert flat.events["reset_leg_joints"].params["recovery_position_range"] == (
        -0.6,
        0.8,
    )
    assert flat.rewards["upward"].weight == 1.0
    assert flat.rewards["righting_progress"].weight == 1.0
    assert flat.rewards["base_height"].weight == -50.0
    assert flat.rewards["upright"].weight == 1.0
    assert flat.rewards["base_height"].params["post_recovery_scale"] == 0.2
    assert flat.rewards["upright"].params["post_recovery_scale"] == 0.2
    assert flat.rewards["track_linear_velocity"].params["upright_gate_hi"] == (
        RECOVERY_UPRIGHT_GATE_HI
    )
    assert (
        flat.rewards["track_linear_velocity"].params["upright_gate_start_step"]
        == RECOVERY_START_STEP
    )
    assert play.rewards["upward"].params["recovery_start_step"] == 0
    assert play.terminations["fell_over"].params["deactivate_after_step"] == 0
    assert set(flat.metrics) == {
        "mean_action_acc",
        "recovery_success_rate",
        "time_to_recover",
    }

    for cfg in (flat, rough):
        for term in cfg.rewards.values():
            callable_obj = (
                term.func.__call__ if inspect.isclass(term.func) else term.func
            )
            signature = inspect.signature(callable_obj)
            assert set(term.params) <= set(signature.parameters)


def test_recovery_tasks_are_registered() -> None:
    tasks = set(list_tasks())
    assert "Mjlab-Velocity-Flat-WF-Tron1B-Recovery" in tasks
    assert "Mjlab-Velocity-Rough-WF-Tron1B-Recovery" in tasks

    cfg = load_env_cfg("Mjlab-Velocity-Flat-WF-Tron1B-Recovery")
    assert "upward" in cfg.rewards


def test_full_depth_predict_recovery_task_is_registered_and_wired() -> None:
    task_id = (
        "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth-Predict-Recovery"
    )
    assert task_id in set(list_tasks())

    cfg = load_env_cfg(task_id)
    play_cfg = wf_tron1b_rough_rep_ts_lin_vel_depth_recovery_env_cfg(play=True)
    agent = asdict(load_rl_cfg(task_id))

    depth_term = cfg.observations[DEPTH_CAMERA_NAME].terms[DEPTH_CAMERA_NAME]
    assert depth_term.func is mdp.async_depth_buffer
    assert "upward" in cfg.rewards
    assert "recovery_phase" in cfg.curriculum
    assert cfg.events["reset_base"].params["recovery_start_step"] == (
        RECOVERY_START_STEP
    )
    assert cfg.terminations["fell_over"].func is bad_orientation_until_step
    assert play_cfg.events["reset_base"].params["recovery_start_step"] == 0

    assert agent["actor"]["class_name"].endswith(
        ":DepthRepresentationVelocityPredictorActorCritic"
    )
    assert agent["algorithm"]["class_name"].endswith(
        ":RepresentationVelocityPredictorTeacherStudentPPO"
    )
    assert agent["obs_groups"]["depth_encoder"] == (DEPTH_CAMERA_NAME,)
