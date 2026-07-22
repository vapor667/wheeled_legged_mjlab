"""Shared per-environment state for self-recovery episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_RECOVERY_STARTED_FALLEN_ATTR = "_recovery_started_fallen"


def recovery_started_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the persistent mask for episodes created by a fallen reset."""
  mask = getattr(env, _RECOVERY_STARTED_FALLEN_ATTR, None)
  if mask is None:
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _RECOVERY_STARTED_FALLEN_ATTR, mask)
  return mask


def set_recovery_started_fallen(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  fallen_mask: torch.Tensor,
) -> None:
  """Record the reset cohort while preserving other running environments."""
  recovery_started_fallen(env)[env_ids] = fallen_mask
