# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for visual representation training through teacher-student PPO."""

from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import RepresentationTeacherStudentPPO
from rsl_rl.models import VisualRepresentationActorCritic
from rsl_rl.storage import RolloutStorage


NUM_ENVS = 4
NUM_STEPS = 4
ACTOR_DIM = 6
ACTOR_HISTORY_DIM = 30
CRITIC_DIM = 16
DEPTH_SHAPE = (5, 32, 24)
HEIGHT_SCAN_START = 4
HEIGHT_DIM = 8
NUM_ACTIONS = 2


class VisualRepresentationTeacherStudentPPOTests(unittest.TestCase):
    def make_obs(self) -> TensorDict:
        return TensorDict(
            {
                "actor": torch.randn(NUM_ENVS, ACTOR_DIM),
                "actor_history": torch.randn(NUM_ENVS, ACTOR_HISTORY_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "depth_camera": torch.randn(NUM_ENVS, *DEPTH_SHAPE),
            },
            batch_size=[NUM_ENVS],
        )

    def make_model(self, obs: TensorDict) -> VisualRepresentationActorCritic:
        return VisualRepresentationActorCritic(
            obs,
            {
                "actor": ["actor"],
                "critic": ["critic"],
                "proprio_encoder": ["actor_history"],
                "privileged_encoder": ["critic"],
                "depth_encoder": ["depth_camera"],
            },
            NUM_ACTIONS,
            hidden_dims=(16,),
            encoder_hidden_dims=(16,),
            latent_dim=4,
            height_latent_dim=4,
            height_scan_start=HEIGHT_SCAN_START,
            height_dim=HEIGHT_DIM,
            height_teacher_hidden_dims=(16,),
            height_proprio_feature_dim=8,
            height_depth_feature_dim=8,
            height_gru_hidden_dim=16,
            height_proprio_hidden_dims=(16,),
            height_depth_channels=(4, 4),
            height_decoder_hidden_dims=(16,),
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )

    def build_algorithm(self) -> tuple[RepresentationTeacherStudentPPO, TensorDict]:
        torch.manual_seed(17)
        obs = self.make_obs()
        model = self.make_model(obs)
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        alg = RepresentationTeacherStudentPPO(
            model,
            storage,
            num_learning_epochs=2,
            num_mini_batches=2,
            learning_rate=1.0e-3,
            proprio_encoder_learning_rate=1.0e-3,
            schedule="fixed",
            desired_kl=0.01,
        )
        return alg, obs

    def fill_rollout(self, alg: RepresentationTeacherStudentPPO, obs: TensorDict) -> None:
        for _ in range(NUM_STEPS):
            alg.act(obs)
            next_obs = self.make_obs()
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            alg.process_env_step(next_obs, rewards, dones, {})
            obs = next_obs
        alg.compute_returns(obs)

    def test_update_optimizes_visual_student_representation_parameters(self) -> None:
        alg, obs = self.build_algorithm()
        self.fill_rollout(alg, obs)

        actor_before = self.clone_named_parameters(alg.actor.actor_head)
        critic_before = self.clone_named_parameters(alg.actor.critic_head)
        privileged_before = self.clone_named_parameters(alg.actor.privileged_encoder)
        proprio_before = self.clone_named_parameters(alg.actor.proprio_encoder)
        height_student_before = self.clone_named_parameters(alg.actor.height_pair.student_height_estimator)

        losses = alg.update()

        self.assertIn("representation", losses)
        self.assertTrue(self.any_param_changed(actor_before, alg.actor.actor_head))
        self.assertTrue(self.any_param_changed(critic_before, alg.actor.critic_head))
        self.assertTrue(self.any_param_changed(privileged_before, alg.actor.privileged_encoder))
        self.assertTrue(self.any_param_changed(proprio_before, alg.actor.proprio_encoder))
        self.assertTrue(self.any_param_changed(height_student_before, alg.actor.height_pair.student_height_estimator))

    def clone_named_parameters(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in module.named_parameters()}

    def any_param_changed(self, before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
        return any(not torch.equal(before[name], param) for name, param in module.named_parameters())


if __name__ == "__main__":
    unittest.main()
