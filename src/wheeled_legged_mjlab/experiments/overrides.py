"""Apply experiment sweep overrides to loaded task configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from mjlab.sensor import GridPatternCfg

from wheeled_legged_mjlab.experiments.patterns import AsymmetricGridPatternCfg
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import DEPTH_CAMERA_NAME


def apply_experiment_overrides(env_cfg: Any, rl_cfg: Any, overrides: Mapping[str, Any] | None) -> None:
    """Mutate loaded env/rl configs according to a sweep variant."""
    if not overrides:
        return

    env_overrides = _as_mapping(overrides.get("env", {}), "env")
    rl_overrides = _as_mapping(overrides.get("rl", {}), "rl")

    if env_overrides:
        _apply_env_overrides(env_cfg, env_overrides)
    if rl_overrides:
        _apply_rl_overrides(rl_cfg, rl_overrides)


def _apply_env_overrides(env_cfg: Any, overrides: Mapping[str, Any]) -> None:
    if "depth_camera_width" in overrides:
        _depth_camera(env_cfg).width = int(overrides["depth_camera_width"])
    if "depth_camera_height" in overrides:
        _depth_camera(env_cfg).height = int(overrides["depth_camera_height"])

    if "depth" in overrides:
        _apply_depth_observation(env_cfg, _as_mapping(overrides["depth"], "env.depth"))
    if "terrain_scan" in overrides:
        _apply_terrain_scan(env_cfg, _as_mapping(overrides["terrain_scan"], "env.terrain_scan"))


def _apply_depth_observation(env_cfg: Any, depth_cfg: Mapping[str, Any]) -> None:
    mode = depth_cfg.get("mode")
    if mode is None:
        depth_group = env_cfg.observations[DEPTH_CAMERA_NAME]
        depth_term = depth_group.terms[DEPTH_CAMERA_NAME]
        if "capture_frequency_hz" in depth_cfg:
            depth_term.params["capture_frequency_hz"] = float(depth_cfg["capture_frequency_hz"])
        if "buffer_size" in depth_cfg:
            depth_term.params["buffer_size"] = int(depth_cfg["buffer_size"])
        if "update_period" in depth_cfg:
            depth_term.params["update_period"] = int(depth_cfg["update_period"])
        return

    depth_group = env_cfg.observations[DEPTH_CAMERA_NAME]
    depth_term = depth_group.terms[DEPTH_CAMERA_NAME]
    if mode == "async":
        depth_term.func = mdp.async_depth_buffer
        depth_term.params = {
            "sensor_name": DEPTH_CAMERA_NAME,
            "capture_frequency_hz": float(depth_cfg.get("capture_frequency_hz", 25.0)),
        }
    elif mode == "buffer":
        depth_term.func = mdp.depth_buffer
        depth_term.params = {
            "sensor_name": DEPTH_CAMERA_NAME,
            "buffer_size": int(depth_cfg.get("buffer_size", 5)),
            "update_period": int(depth_cfg.get("update_period", 5)),
        }
    else:
        raise ValueError(f"Unsupported env.depth.mode={mode!r}; expected 'async' or 'buffer'")


def _apply_terrain_scan(env_cfg: Any, scan_cfg: Mapping[str, Any]) -> None:
    sensor = _sensor_by_name(env_cfg, "terrain_scan")
    resolution = float(scan_cfg.get("resolution", getattr(sensor.pattern, "resolution", 0.1)))

    if {"x_back", "x_front", "y_left", "y_right"} & set(scan_cfg):
        y_half = scan_cfg.get("y_half")
        y_left = scan_cfg.get("y_left", y_half)
        y_right = scan_cfg.get("y_right", y_half)
        if y_left is None or y_right is None:
            raise ValueError("terrain_scan asymmetric override requires y_half or both y_left and y_right")
        pattern = AsymmetricGridPatternCfg(
            x_back=float(scan_cfg.get("x_back", 0.5)),
            x_front=float(scan_cfg.get("x_front", 0.5)),
            y_left=float(y_left),
            y_right=float(y_right),
            resolution=resolution,
        )
        grid_shape = pattern.grid_shape
    else:
        size = scan_cfg.get("size", getattr(sensor.pattern, "size", (1.0, 1.0)))
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("terrain_scan.size must be a 2-item list or tuple")
        pattern = GridPatternCfg(size=(float(size[0]), float(size[1])), resolution=resolution)
        grid_shape = _centered_grid_shape(pattern.size, pattern.resolution)

    sensor.pattern = pattern
    _update_roughness_grid_shapes(env_cfg, grid_shape)


def _apply_rl_overrides(rl_cfg: Any, overrides: Mapping[str, Any]) -> None:
    actor = rl_cfg.actor
    for name in (
        "hidden_dims",
        "encoder_hidden_dims",
        "latent_dim",
        "depth_feature_dim",
        "depth_gru_hidden_dim",
        "depth_channels",
    ):
        if name in overrides:
            value = overrides[name]
            if name.endswith("dims") or name == "depth_channels":
                value = tuple(int(item) for item in value)
            elif name.endswith("dim"):
                value = int(value)
            setattr(actor, name, value)

    algorithm_overrides = _as_mapping(overrides.get("algorithm", {}), "rl.algorithm")
    for name, value in algorithm_overrides.items():
        setattr(rl_cfg.algorithm, name, value)


def _update_roughness_grid_shapes(env_cfg: Any, grid_shape: tuple[int, int]) -> None:
    for group_name in ("critic", "privileged_encoder"):
        group = env_cfg.observations.get(group_name)
        if group is None:
            continue
        term = group.terms.get("roughness_indicator")
        if term is not None and "grid_shape" in term.params:
            term.params["grid_shape"] = grid_shape

    for reward in env_cfg.rewards.values():
        params = getattr(reward, "params", None)
        if isinstance(params, dict) and "grid_shape" in params:
            params["grid_shape"] = grid_shape


def _depth_camera(env_cfg: Any) -> Any:
    return _sensor_by_name(env_cfg, DEPTH_CAMERA_NAME)


def _sensor_by_name(env_cfg: Any, name: str) -> Any:
    for sensor in env_cfg.scene.sensors:
        if getattr(sensor, "name", None) == name:
            return sensor
    raise KeyError(f"Sensor {name!r} not found in env_cfg.scene.sensors")


def _centered_grid_shape(size: tuple[float, float], resolution: float) -> tuple[int, int]:
    x_count = _centered_axis_count(size[0], resolution)
    y_count = _centered_axis_count(size[1], resolution)
    return (y_count, x_count)


def _centered_axis_count(size: float, resolution: float) -> int:
    import torch

    return int(torch.arange(-size / 2, size / 2 + resolution * 0.5, resolution).numel())


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value
