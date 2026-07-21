"""Event functions for the task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_partial_fallen(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    fallen_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    fallen_velocity_range: dict[str, tuple[float, float]] | None = None,
    fallen_fraction: float = 0.3,
    recovery_start_step: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset a random subset with arbitrary attitude and the rest normally."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)

    effective_fraction = (
        fallen_fraction if env.common_step_counter >= recovery_start_step else 0.0
    )
    fallen_mask = torch.rand(len(env_ids), device=env.device) < effective_fraction
    upright_env_ids = env_ids[~fallen_mask]
    fallen_env_ids = env_ids[fallen_mask]

    if upright_env_ids.numel() > 0:
        reset_root_state_uniform(
            env,
            upright_env_ids,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )
    if fallen_env_ids.numel() > 0:
        reset_root_state_uniform(
            env,
            fallen_env_ids,
            pose_range=fallen_pose_range,
            velocity_range=fallen_velocity_range,
            asset_cfg=asset_cfg,
        )


def reset_joints_by_offset_after_step(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    recovery_position_range: tuple[float, float],
    recovery_start_step: int,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Use the wider recovery joint reset range only after its curriculum starts."""
    active_position_range = (
        recovery_position_range
        if env.common_step_counter >= recovery_start_step
        else position_range
    )
    reset_joints_by_offset(
        env,
        env_ids,
        position_range=active_position_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )


def _replace_non_finite_(tensor: torch.Tensor, env_ids: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        return
    tensor[env_ids] = torch.nan_to_num(tensor[env_ids], nan=0.0, posinf=0.0, neginf=0.0)


def clear_non_finite_sim_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Clear stale non-finite physics/sensor buffers after resetting envs."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)

    data = env.sim.data
    for name in (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "sensordata",
        "actuator_force",
        "qfrc_actuator",
    ):
        tensor = getattr(data, name, None)
        if tensor is not None:
            _replace_non_finite_(tensor, env_ids)

    for sensor in env.scene.sensors.values():
        if not isinstance(sensor, ContactSensor):
            continue
        for name in (
            "force",
            "torque",
            "dist",
            "pos",
            "normal",
            "tangent",
            "current_air_time",
            "last_air_time",
            "current_contact_time",
            "last_contact_time",
            "force_history",
            "torque_history",
            "dist_history",
        ):
            tensor = getattr(sensor.data, name, None)
            if tensor is not None:
                _replace_non_finite_(tensor, env_ids)


def prepare_quantities(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Compute the nominal foot position in the body frame.

    This function computes the nominal foot position in the body frame. This function is only suitable for TRON robot.

    The computed nominal foot position is stored in the following attributes of env:
        - env._nominal_foot_position_b: Nominal foot positions in body frame
        - env._wheels_link_ids: Body indices of wheel links
        - env._wheels_joint_ids: Joint indices of wheel joints
        - env._foot_radius: Radius of the foot/wheel (0.127m)
    """
    asset: Entity = env.scene[asset_cfg.name]

    wheel_link_idx, _ = asset.find_bodies("wheel_[RL]_Link")
    wheel_joint_ids, _ = asset.find_joints("wheel_[RL]_Joint")
    base_idx, _ = asset.find_bodies("base_Link")

    wheels_pos_w = asset.data.body_link_pos_w[:, wheel_link_idx, :]
    base_pos_w = asset.data.body_link_pos_w[:, base_idx, :]
    base_quat = asset.data.body_link_quat_w[:, base_idx, :]

    nominal_foot_position_b = torch.zeros(len(wheel_link_idx), 3, device=env.device)

    for j in range(env.num_envs):
        if torch.any(asset.data.joint_pos[j, :] > 5e-2):
            continue
        for i in range(len(wheel_link_idx)):
            nominal_foot_position_b[i, :] = quat_apply_inverse(
                base_quat[j, 0, :], wheels_pos_w[j, i, :] - base_pos_w[j, 0, :]
            )
        break

    assert (nominal_foot_position_b != 0.0).any(), "Failed to compute nominal foot positions"

    env._nominal_foot_position_b = nominal_foot_position_b  # type: ignore
    env._wheels_link_ids = wheel_link_idx  # type: ignore
    env._wheels_joint_ids = wheel_joint_ids  # type: ignore
    env._foot_radius = 0.127  # type: ignore
