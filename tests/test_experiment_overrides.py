"""Tests for experiment sweep overrides."""

from __future__ import annotations

from dataclasses import asdict

import torch

from mjlab.tasks.registry import load_rl_cfg

import wheeled_legged_mjlab  # noqa: F401
from wheeled_legged_mjlab.experiments import AsymmetricGridPatternCfg, apply_experiment_overrides
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    DEPTH_CAMERA_NAME,
    wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg,
)


def test_asymmetric_grid_pattern_generates_expected_extent_and_shape() -> None:
    pattern = AsymmetricGridPatternCfg(
        x_back=0.2,
        x_front=0.4,
        y_left=0.3,
        y_right=0.1,
        resolution=0.1,
    )

    offsets, directions = pattern.generate_rays(None, "cpu")

    assert pattern.grid_shape == (5, 7)
    assert offsets.shape == (35, 3)
    assert torch.isclose(offsets[:, 0].min(), torch.tensor(-0.2))
    assert torch.isclose(offsets[:, 0].max(), torch.tensor(0.4))
    assert torch.isclose(offsets[:, 1].min(), torch.tensor(-0.1))
    assert torch.isclose(offsets[:, 1].max(), torch.tensor(0.3))
    assert torch.allclose(directions[0], torch.tensor([0.0, 0.0, -1.0]))


def test_apply_experiment_overrides_updates_depth_scan_and_model_cfgs() -> None:
    env_cfg = wf_tron1b_rough_rep_ts_lin_vel_depth_env_cfg()
    rl_cfg = load_rl_cfg("Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-Depth")

    apply_experiment_overrides(
        env_cfg,
        rl_cfg,
        {
            "env": {
                "depth_camera_width": 32,
                "depth_camera_height": 48,
                "depth": {
                    "mode": "buffer",
                    "buffer_size": 3,
                    "update_period": 2,
                },
                "terrain_scan": {
                    "x_back": 0.3,
                    "x_front": 1.2,
                    "y_half": 0.5,
                    "resolution": 0.1,
                },
            },
            "rl": {
                "hidden_dims": [768, 512, 256],
                "encoder_hidden_dims": [768, 512, 256],
                "latent_dim": 64,
                "depth_feature_dim": 128,
                "depth_gru_hidden_dim": 128,
                "depth_channels": [32, 64, 64],
            },
        },
    )

    depth_sensor = next(sensor for sensor in env_cfg.scene.sensors if sensor.name == DEPTH_CAMERA_NAME)
    assert depth_sensor.width == 32
    assert depth_sensor.height == 48

    depth_term = env_cfg.observations[DEPTH_CAMERA_NAME].terms[DEPTH_CAMERA_NAME]
    assert depth_term.func is mdp.depth_buffer
    assert depth_term.params == {
        "sensor_name": DEPTH_CAMERA_NAME,
        "buffer_size": 3,
        "update_period": 2,
    }

    terrain_sensor = next(sensor for sensor in env_cfg.scene.sensors if sensor.name == "terrain_scan")
    assert isinstance(terrain_sensor.pattern, AsymmetricGridPatternCfg)
    assert terrain_sensor.pattern.grid_shape == (11, 16)
    assert env_cfg.observations["critic"].terms["roughness_indicator"].params["grid_shape"] == (11, 16)
    assert env_cfg.observations["privileged_encoder"].terms["roughness_indicator"].params["grid_shape"] == (11, 16)
    assert all(
        reward.params.get("grid_shape") == (11, 16)
        for reward in env_cfg.rewards.values()
        if isinstance(reward.params, dict) and "grid_shape" in reward.params
    )

    agent = asdict(rl_cfg)
    assert agent["actor"]["hidden_dims"] == (768, 512, 256)
    assert agent["actor"]["encoder_hidden_dims"] == (768, 512, 256)
    assert agent["actor"]["latent_dim"] == 64
    assert agent["actor"]["depth_feature_dim"] == 128
    assert agent["actor"]["depth_gru_hidden_dim"] == 128
    assert agent["actor"]["depth_channels"] == (32, 64, 64)
