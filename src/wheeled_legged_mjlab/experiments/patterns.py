"""Sensor ray patterns used by experiment overrides."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AsymmetricGridPatternCfg:
    """Grid pattern with independent forward/backward and left/right extents."""

    x_back: float
    x_front: float
    y_left: float
    y_right: float
    resolution: float
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the sample grid shape as ``(rows_y, cols_x)``."""
        return (
            _num_samples(self.y_left, self.y_right, self.resolution),
            _num_samples(self.x_back, self.x_front, self.resolution),
        )

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate local ray offsets and directions for ``RayCastSensor``."""
        del mj_model
        _validate_extent("x_back", self.x_back)
        _validate_extent("x_front", self.x_front)
        _validate_extent("y_left", self.y_left)
        _validate_extent("y_right", self.y_right)
        if self.resolution <= 0.0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")

        x = torch.arange(
            -self.x_back,
            self.x_front + self.resolution * 0.5,
            self.resolution,
            device=device,
            dtype=torch.float32,
        )
        y = torch.arange(
            -self.y_right,
            self.y_left + self.resolution * 0.5,
            self.resolution,
            device=device,
            dtype=torch.float32,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

        local_offsets = torch.zeros((grid_x.numel(), 3), device=device, dtype=torch.float32)
        local_offsets[:, 0] = grid_x.flatten()
        local_offsets[:, 1] = grid_y.flatten()

        direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
        direction = direction / direction.norm()
        local_directions = direction.unsqueeze(0).expand(local_offsets.shape[0], 3).clone()
        return local_offsets, local_directions


def _validate_extent(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _num_samples(negative_extent: float, positive_extent: float, resolution: float) -> int:
    if resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if negative_extent < 0.0 or positive_extent < 0.0:
        raise ValueError("grid extents must be non-negative")
    return int(torch.arange(-negative_extent, positive_extent + resolution * 0.5, resolution).numel())
