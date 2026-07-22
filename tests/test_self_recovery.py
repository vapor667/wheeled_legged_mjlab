from __future__ import annotations

import inspect
import math
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import wheeled_legged_mjlab  # noqa: F401
from scripts.rsl_rl.train import _sync_env_step_counter_from_runner
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    DEPTH_CAMERA_NAME,
    RECOVERY_FALLEN_FRACTION,
    RECOVERY_FALLEN_FRACTION_RAMP_STEPS,
    RECOVERY_MAX_FALLEN_TERRAIN_LEVEL,
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
    recovery_attempt_rate,
    recovery_episode_outcome,
    upright_time_fraction,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.recovery import (
    recovery_started_fallen,
    set_recovery_started_fallen,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.rewards import (
    _apply_upright_gate,
    _upright_gate,
    righting_progress,
    upward,
)
from wheeled_legged_mjlab.tasks.velocity.mdp.terminations import (
    bad_orientation_except_recovery,
    illegal_contact_except_recovery,
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


def test_resume_synchronizes_environment_curriculum_clock() -> None:
    base_env = SimpleNamespace(common_step_counter=0)
    wrapped_env = SimpleNamespace(unwrapped=base_env)
    runner = SimpleNamespace(current_learning_iteration=4_800)

    restored = _sync_env_step_counter_from_runner(
        wrapped_env, runner, RECOVERY_STEPS_PER_ITERATION
    )

    assert restored == 115_200
    assert base_env.common_step_counter == restored


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
        num_envs=12,
        device="cpu",
        common_step_counter=(
            RECOVERY_START_STEP + RECOVERY_FALLEN_FRACTION_RAMP_STEPS
        ),
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
    assert recovery_started_fallen(env)[env_ids].tolist() == [True, False, True, False]


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
    assert not recovery_started_fallen(env).any()


def test_partial_fallen_reset_ramps_and_uses_only_easy_terrain() -> None:
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_type="generator"),
        terrain_levels=torch.tensor([0, 6, 5]),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        common_step_counter=(
            RECOVERY_START_STEP + RECOVERY_FALLEN_FRACTION_RAMP_STEPS
        ),
        scene=SimpleNamespace(terrain=terrain),
    )

    with (
        patch(
            "wheeled_legged_mjlab.tasks.velocity.mdp.events.torch.rand",
            return_value=torch.zeros(3),
        ),
        patch(
            "wheeled_legged_mjlab.tasks.velocity.mdp.events.reset_root_state_uniform"
        ) as reset_mock,
    ):
        reset_root_state_partial_fallen(
            env,
            torch.arange(3),
            pose_range={"yaw": (-math.pi, math.pi)},
            fallen_pose_range={"roll": (-math.pi, math.pi)},
            fallen_fraction=RECOVERY_FALLEN_FRACTION,
            recovery_start_step=RECOVERY_START_STEP,
            fallen_fraction_ramp_steps=RECOVERY_FALLEN_FRACTION_RAMP_STEPS,
            max_fallen_terrain_level=RECOVERY_MAX_FALLEN_TERRAIN_LEVEL,
        )

    assert reset_mock.call_args_list[0].args[1].tolist() == [1]
    assert reset_mock.call_args_list[1].args[1].tolist() == [0, 2]
    assert recovery_started_fallen(env).tolist() == [True, False, True]


def test_recovery_metrics_report_upright_state_and_first_recovery_time() -> None:
    env = _orientation_env([1.0, -1.0])
    set_recovery_started_fallen(
        env, torch.arange(2), torch.tensor([True, False])
    )
    success_metric = recovery_episode_outcome(None, env)
    time_metric = recovery_episode_outcome(None, env)

    assert upright_time_fraction(env).tolist() == [0.0, 1.0]
    assert recovery_attempt_rate(env).tolist() == [1.0, 0.0]
    assert success_metric(env, output="success").tolist() == [0.0, 0.0]
    assert time_metric(env, output="success_time").tolist() == [0.0, 0.0]
    env.scene["robot"].data.projected_gravity_b[0, 2] = -1.0
    assert success_metric(env, output="success").tolist() == [1.0, 0.0]
    assert torch.allclose(
        time_metric(env, output="success_time"), torch.tensor([0.2, 0.0])
    )

    success_metric.reset(torch.tensor([0]))
    assert success_metric(env, output="success")[0].item() == 1.0


def test_only_explicit_recovery_cohort_ignores_fall_and_contact() -> None:
    env = _orientation_env([0.0])
    env.scene["illegal_ground_contact"] = SimpleNamespace(
        data=SimpleNamespace(
            force_history=None,
            found=torch.tensor([[True]]),
        )
    )

    set_recovery_started_fallen(env, torch.tensor([0]), torch.tensor([False]))
    assert bad_orientation_except_recovery(
        env,
        limit_angle=math.radians(65.0),
    ).item()
    assert illegal_contact_except_recovery(
        env,
        sensor_name="illegal_ground_contact",
    ).item()

    set_recovery_started_fallen(env, torch.tensor([0]), torch.tensor([True]))
    assert not bad_orientation_except_recovery(
        env,
        limit_angle=math.radians(65.0),
    ).item()
    assert not illegal_contact_except_recovery(
        env,
        sensor_name="illegal_ground_contact",
    ).item()


def test_recovery_cohort_does_not_update_terrain_curriculum() -> None:
    class Terrain:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                terrain_generator=SimpleNamespace(
                    size=(8.0, 8.0), sub_terrains={}
                )
            )
            self.terrain_levels = torch.ones(2, dtype=torch.long)
            self.terrain_types = torch.zeros(2, dtype=torch.long)
            self.terrain_origins = torch.empty(1, 0, 3)
            self.move_up = torch.zeros(2, dtype=torch.bool)
            self.move_down = torch.zeros(2, dtype=torch.bool)

        def update_env_origins(self, env_ids, move_up, move_down) -> None:
            del env_ids
            self.move_up = move_up.clone()
            self.move_down = move_down.clone()

    root_pos = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))

    class Scene:
        def __init__(self) -> None:
            self.terrain = Terrain()
            self.env_origins = torch.zeros(2, 3)

        def __getitem__(self, name: str):
            assert name == "robot"
            return robot

    scene = Scene()
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        command_manager=SimpleNamespace(
            get_command=lambda _name: torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        ),
        max_episode_length_s=20.0,
    )
    set_recovery_started_fallen(
        env, torch.arange(2), torch.tensor([True, False])
    )

    mdp.terrain_levels_vel(env, torch.arange(2), command_name="base_velocity")

    assert scene.terrain.move_up.tolist() == [False, False]
    assert scene.terrain.move_down.tolist() == [False, True]


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
    assert flat.terminations["fell_over"].func is bad_orientation_except_recovery
    assert (
        flat.terminations["illegal_contact"].func
        is illegal_contact_except_recovery
    )
    assert "fell_over_limit_angle" in flat.curriculum
    assert "recovery_phase" in flat.curriculum
    assert "terrain_levels" in rough.curriculum
    assert rough.terminations["out_of_terrain_bounds"].params["margin"] == 1.5
    assert flat.events["reset_base"].params["fallen_fraction"] == (
        RECOVERY_FALLEN_FRACTION
    )
    assert flat.events["reset_base"].params["fallen_fraction_ramp_steps"] == (
        RECOVERY_FALLEN_FRACTION_RAMP_STEPS
    )
    assert rough.events["reset_base"].params["max_fallen_terrain_level"] == (
        RECOVERY_MAX_FALLEN_TERRAIN_LEVEL
    )
    assert play.events["reset_base"].params["fallen_fraction"] == 1.0
    assert play.events["reset_base"].params["fallen_fraction_ramp_steps"] == 0
    assert flat.events["reset_base"].params["recovery_start_step"] == (
        RECOVERY_START_STEP
    )
    assert flat.events["reset_leg_joints"].params["position_range"] == (-0.3, 0.5)
    assert flat.events["reset_leg_joints"].params["recovery_position_range"] == (
        -0.6,
        0.8,
    )
    assert flat.rewards["upward"].weight == 0.25
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
    assert set(flat.metrics) == {
        "mean_action_acc",
        "upright_time_fraction",
        "recovery_attempt_rate",
        "recovery_success_unconditional",
        "recovery_time_success_unconditional",
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
    assert cfg.terminations["fell_over"].func is bad_orientation_except_recovery
    assert play_cfg.events["reset_base"].params["recovery_start_step"] == 0

    assert agent["actor"]["class_name"].endswith(
        ":DepthRepresentationVelocityPredictorActorCritic"
    )
    assert agent["algorithm"]["class_name"].endswith(
        ":RepresentationVelocityPredictorTeacherStudentPPO"
    )
    assert agent["obs_groups"]["depth_encoder"] == (DEPTH_CAMERA_NAME,)
    assert agent["actor"]["distribution_cfg"] == {
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
        "std_range": (0.05, 0.8),
    }
    assert agent["algorithm"]["entropy_coef"] == 0.002
