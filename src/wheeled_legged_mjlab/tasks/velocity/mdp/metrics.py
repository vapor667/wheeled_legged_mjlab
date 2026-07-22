"""Metrics for evaluating self-recovery behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .recovery import recovery_started_fallen

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def upright_time_fraction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_threshold: float = 0.9,
    recovery_start_step: int = 0,
) -> torch.Tensor:
    """Return instantaneous upright occupancy across all environments."""
    asset: Entity = env.scene[asset_cfg.name]
    up = -asset.data.projected_gravity_b[:, 2]
    active = float(env.common_step_counter >= recovery_start_step)
    return (up > upright_threshold).float() * active


def recovery_attempt_rate(
    env: ManagerBasedRlEnv,
    recovery_start_step: int = 0,
) -> torch.Tensor:
    """Return the episode-level indicator for an explicit fallen reset."""
    active = float(env.common_step_counter >= recovery_start_step)
    return recovery_started_fallen(env).float() * active


class recovery_episode_outcome:
    """Track recovery success and its success-weighted completion time."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._recovered = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._elapsed = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        upright_threshold: float = 0.9,
        recovery_start_step: int = 0,
        output: Literal["success", "success_time"] = "success",
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        up = -asset.data.projected_gravity_b[:, 2]
        started_fallen = recovery_started_fallen(env)

        if env.common_step_counter < recovery_start_step:
            self.reset(None)
            return torch.zeros_like(up)

        pending = started_fallen & ~self._recovered
        self._elapsed[pending] += env.step_dt
        self._recovered[pending & (up > upright_threshold)] = True
        success = started_fallen & self._recovered
        if output == "success_time":
            return torch.where(
                success, self._elapsed, torch.zeros_like(self._elapsed)
            )
        return success.float()

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._recovered[env_ids] = False
        self._elapsed[env_ids] = 0.0
