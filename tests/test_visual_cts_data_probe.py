"""Probe current Visual-CTS data prerequisites without changing training behavior."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import wheeled_legged_mjlab  # noqa: F401
from wheeled_legged_mjlab.tasks.velocity import mdp
from wheeled_legged_mjlab.tasks.velocity.config.wf_tron1b.env_cfgs import (
    DEPTH_BUFFER_SIZE,
    DEPTH_BUFFER_UPDATE_PERIOD,
    DEPTH_CAMERA_HEIGHT,
    DEPTH_CAMERA_NAME,
    DEPTH_CAMERA_WIDTH,
    TERRAIN_SCAN_GRID_SHAPE,
)


TASK_ID = "Mjlab-Velocity-Rough-WF-Tron1B-RepTS-Depth"


class VisualCTSDataProbeTests(unittest.TestCase):
    def test_depth_height_and_history_are_available_but_not_training_inputs(self) -> None:
        self.assertIn(TASK_ID, set(list_tasks()))

        env_cfg = load_env_cfg(TASK_ID)
        agent = asdict(load_rl_cfg(TASK_ID))

        depth_group = env_cfg.observations[DEPTH_CAMERA_NAME]
        depth_term = depth_group.terms[DEPTH_CAMERA_NAME]
        self.assertIs(depth_term.func, mdp.depth_buffer)
        self.assertEqual(
            depth_term.params,
            {
                "sensor_name": DEPTH_CAMERA_NAME,
                "buffer_size": DEPTH_BUFFER_SIZE,
                "update_period": DEPTH_BUFFER_UPDATE_PERIOD,
            },
        )
        self.assertFalse(depth_group.enable_corruption)

        actor_history_cfg = env_cfg.observations["actor_history"]
        self.assertEqual(actor_history_cfg.history_length, 5)
        self.assertTrue(actor_history_cfg.flatten_history_dim)

        actor_terms = env_cfg.observations["actor"].terms
        actor_history_terms = actor_history_cfg.terms
        critic_terms = env_cfg.observations["critic"].terms
        self.assertNotIn("height_scan", actor_terms)
        self.assertNotIn("height_scan", actor_history_terms)
        self.assertIn("height_scan", critic_terms)

        expected_depth_buffer_shape = (
            DEPTH_BUFFER_SIZE,
            DEPTH_CAMERA_HEIGHT,
            DEPTH_CAMERA_WIDTH,
        )
        expected_height_scan_dim = math.prod(TERRAIN_SCAN_GRID_SHAPE)
        self.assertEqual(expected_depth_buffer_shape, (5, 32, 24))
        self.assertEqual(expected_height_scan_dim, 121)

        training_obs_groups = {
            group for groups in agent["obs_groups"].values() for group in groups
        }
        self.assertNotIn(DEPTH_CAMERA_NAME, training_obs_groups)
        self.assertEqual(
            agent["obs_groups"],
            {
                "actor": ("actor",),
                "critic": ("critic",),
                "proprio_encoder": ("actor_history",),
                "privileged_encoder": ("critic",),
            },
        )

    @unittest.skipUnless(
        os.environ.get("RUN_VISUAL_CTS_LIVE_PROBE") == "1",
        "set RUN_VISUAL_CTS_LIVE_PROBE=1 to initialize MuJoCo and inspect live observation shapes",
    )
    def test_live_observation_shapes(self) -> None:
        import warp as wp
        from mjlab.envs import ManagerBasedRlEnv

        warp_cache_dir = Path(os.environ.get("WARP_KERNEL_CACHE_DIR", "/tmp/warp-kernel-cache"))
        warp_cache_dir.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(warp_cache_dir)

        env_cfg = load_env_cfg(TASK_ID)
        env_cfg.scene.num_envs = int(os.environ.get("VISUAL_CTS_PROBE_NUM_ENVS", "2"))
        device = os.environ.get("VISUAL_CTS_PROBE_DEVICE", "cpu")
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        try:
            obs, _ = env.reset()
            self.assertIn("actor", obs)
            self.assertIn("actor_history", obs)
            self.assertIn("critic", obs)
            self.assertIn(DEPTH_CAMERA_NAME, obs)

            actor_shape = tuple(obs["actor"].shape)
            actor_history_shape = tuple(obs["actor_history"].shape)
            critic_shape = tuple(obs["critic"].shape)
            depth_shape = tuple(obs[DEPTH_CAMERA_NAME].shape)
            print("actor_shape", actor_shape)
            print("actor_history_shape", actor_history_shape)
            print("critic_shape", critic_shape)
            print("depth_camera_shape", depth_shape)

            self.assertEqual(actor_history_shape[0], actor_shape[0])
            self.assertEqual(
                actor_history_shape[1],
                actor_shape[1] * env_cfg.observations["actor_history"].history_length,
            )
            self.assertEqual(critic_shape[0], actor_shape[0])
            self.assertEqual(depth_shape[0], actor_shape[0])

            expected_depth_elements = (
                DEPTH_BUFFER_SIZE * DEPTH_CAMERA_HEIGHT * DEPTH_CAMERA_WIDTH
            )
            if len(depth_shape) == 4:
                self.assertEqual(
                    depth_shape[1:],
                    (DEPTH_BUFFER_SIZE, DEPTH_CAMERA_HEIGHT, DEPTH_CAMERA_WIDTH),
                )
            else:
                self.assertEqual(depth_shape[1], expected_depth_elements)
        finally:
            env.close()

    def test_visual_cts_probe_is_not_git_ignored(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "check-ignore", "tests/test_visual_cts_data_probe.py"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1, result.stdout)


if __name__ == "__main__":
    unittest.main()
