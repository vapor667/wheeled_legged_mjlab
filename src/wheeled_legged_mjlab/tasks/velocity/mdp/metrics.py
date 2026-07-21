"""Metrics for evaluating self-recovery behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def recovery_success_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_threshold: float = 0.9,
    recovery_start_step: int = 0,
) -> torch.Tensor:
    """Return one for environments whose signed upright direction is recovered."""
    asset: Entity = env.scene[asset_cfg.name]
    up = -asset.data.projected_gravity_b[:, 2]
    active = float(env.common_step_counter >= recovery_start_step)
    return (up > upright_threshold).float() * active


class time_to_recover:
    """Track seconds from an initially non-upright state to first recovery."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._started_fallen = torch.zeros_like(self._initialized)
        self._recovered = torch.zeros_like(self._initialized)
        self._elapsed = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        upright_threshold: float = 0.9,
        recovery_start_step: int = 0,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        up = -asset.data.projected_gravity_b[:, 2]

        if env.common_step_counter < recovery_start_step:
            self.reset(None)
            return torch.zeros_like(up)

        new_episode = ~self._initialized
        self._started_fallen[new_episode] = up[new_episode] <= upright_threshold
        self._initialized[new_episode] = True

        pending = self._started_fallen & ~self._recovered
        self._elapsed[pending] += env.step_dt
        self._recovered[pending & (up > upright_threshold)] = True
        return torch.where(
            self._started_fallen,
            self._elapsed,
            torch.zeros_like(self._elapsed),
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._started_fallen[env_ids] = False
        self._recovered[env_ids] = False
        self._elapsed[env_ids] = 0.0
