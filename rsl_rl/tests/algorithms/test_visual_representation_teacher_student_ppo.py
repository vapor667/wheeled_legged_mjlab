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
HISTORY_LENGTH = 5
CRITIC_DIM = 16
DEPTH_SHAPE = (1, 32, 24)
HEIGHT_SCAN_START = 4
HEIGHT_DIM = 8
NUM_ACTIONS = 2


class VisualRepresentationTeacherStudentPPOTests(unittest.TestCase):
    def make_obs(self) -> TensorDict:
        return TensorDict(
            {
                "actor": torch.randn(NUM_ENVS, ACTOR_DIM),
                "actor_history": torch.randn(NUM_ENVS, HISTORY_LENGTH, ACTOR_DIM),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                "depth_camera": torch.randn(NUM_ENVS, *DEPTH_SHAPE),
                "height_scan": torch.randn(NUM_ENVS, HEIGHT_DIM),
            },
            batch_size=[NUM_ENVS],
        )

    def make_model(self, obs: TensorDict) -> VisualRepresentationActorCritic:
        return VisualRepresentationActorCritic(
            obs,
            {
                "teacher_actor": ["actor"],
                "critic": ["critic"],
                "student_history": ["actor_history"],
                "privileged_encoder": ["critic"],
                "depth_encoder": ["depth_camera"],
                "height_encoder": ["height_scan"],
            },
            NUM_ACTIONS,
            hidden_dims=(16,),
            encoder_hidden_dims=(16,),
            latent_dim=4,
            height_latent_dim=4,
            height_scan_start=None,
            height_dim=HEIGHT_DIM,
            height_teacher_hidden_dims=(16,),
            height_proprio_feature_dim=8,
            height_depth_feature_dim=8,
            height_gru_hidden_dim=16,
            height_proprio_hidden_dims=(16,),
            height_depth_channels=(4, 4),
            height_decoder_hidden_dims=(16,),
            privileged_decoder_hidden_dims=(16,),
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
            teacher_student_ratio=1.0,
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
        privileged_decoder_before = self.clone_named_parameters(alg.actor.privileged_decoder)
        height_student_before = self.clone_named_parameters(alg.actor.height_pair.student_height_estimator)

        losses = alg.update()

        self.assertTrue(
            {
                "representation",
                "privileged_latent",
                "privileged_reconstruction",
                "privileged_total",
                "height_latent",
                "height_reconstruction",
                "CTS/teacher_mean_step_reward",
                "CTS/student_mean_step_reward",
            }
            <= set(losses)
        )
        self.assertTrue(self.any_param_changed(actor_before, alg.actor.actor_head))
        self.assertTrue(self.any_param_changed(critic_before, alg.actor.critic_head))
        self.assertTrue(self.any_param_changed(privileged_before, alg.actor.privileged_encoder))
        self.assertTrue(self.any_param_changed(proprio_before, alg.actor.proprio_encoder))
        self.assertTrue(self.any_param_changed(privileged_decoder_before, alg.actor.privileged_decoder))
        self.assertTrue(self.any_param_changed(height_student_before, alg.actor.height_pair.student_height_estimator))

    def test_rollout_preserves_equal_teacher_student_mask_and_gru_states(self) -> None:
        alg, obs = self.build_algorithm()
        self.fill_rollout(alg, obs)

        expected_mask = torch.tensor([False, True, False, True])
        self.assertTrue(torch.equal(alg.teacher_mask, expected_mask))
        self.assertIsNotNone(alg.storage.teacher_masks)
        for step_mask in alg.storage.teacher_masks:
            self.assertTrue(torch.equal(step_mask.squeeze(-1), expected_mask))
        self.assertIsNotNone(alg.storage.saved_hidden_state_a)
        self.assertIsNotNone(alg.storage.student_latents)
        self.assertEqual(
            tuple(alg.storage.student_latents.shape),
            (NUM_STEPS, NUM_ENVS, alg.actor.latent_dim + alg.actor.height_latent_dim),
        )
        self.assertEqual(
            tuple(alg.storage.saved_hidden_state_a[0].shape),
            (NUM_STEPS, NUM_ENVS, alg.actor.height_pair.student_height_estimator.gru_hidden_dim),
        )

    def test_ppo_and_representation_phases_update_disjoint_parameters(self) -> None:
        alg, obs = self.build_algorithm()
        self.fill_rollout(alg, obs)

        actor_before = self.clone_named_parameters(alg.actor.actor_head)
        student_privileged_before = self.clone_named_parameters(alg.actor.proprio_encoder)
        student_height_before = self.clone_named_parameters(alg.actor.height_pair.student_height_estimator)
        alg._update_ppo_phase()

        self.assertTrue(self.any_param_changed(actor_before, alg.actor.actor_head))
        self.assertFalse(self.any_param_changed(student_privileged_before, alg.actor.proprio_encoder))
        self.assertFalse(
            self.any_param_changed(student_height_before, alg.actor.height_pair.student_height_estimator)
        )

        actor_before = self.clone_named_parameters(alg.actor.actor_head)
        critic_before = self.clone_named_parameters(alg.actor.critic_head)
        teacher_privileged_before = self.clone_named_parameters(alg.actor.privileged_encoder)
        teacher_height_before = self.clone_named_parameters(alg.actor.height_pair.teacher_height_encoder)
        student_height_before = self.clone_named_parameters(alg.actor.height_pair.student_height_estimator)
        alg._update_representation_phase()

        self.assertFalse(self.any_param_changed(actor_before, alg.actor.actor_head))
        self.assertFalse(self.any_param_changed(critic_before, alg.actor.critic_head))
        self.assertFalse(self.any_param_changed(teacher_privileged_before, alg.actor.privileged_encoder))
        self.assertFalse(
            self.any_param_changed(teacher_height_before, alg.actor.height_pair.teacher_height_encoder)
        )
        self.assertTrue(
            self.any_param_changed(student_height_before, alg.actor.height_pair.student_height_estimator)
        )

    def test_ppo_phase_uses_cached_student_latents_without_estimator_forward(self) -> None:
        alg, obs = self.build_algorithm()
        self.fill_rollout(alg, obs)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("student estimator must not run during PPO update")

        alg.actor.proprio_encoder.forward = fail_if_called
        alg.actor.height_pair.student_height_estimator.forward = fail_if_called
        alg._update_ppo_phase()

    def test_ppo_batches_are_teacher_student_stratified(self) -> None:
        alg, obs = self.build_algorithm()
        self.fill_rollout(alg, obs)

        batches = list(alg.storage.mini_batch_generator(num_mini_batches=2, num_epochs=1))
        self.assertEqual(len(batches), 2)
        for batch in batches:
            mask = batch.teacher_mask.view(-1)
            self.assertEqual(int(mask.sum()), int((~mask).sum()))

    def test_representation_chunks_only_contain_student_trajectories(self) -> None:
        alg, obs = self.build_algorithm()
        self.fill_rollout(alg, obs)

        batches = list(
            alg.storage.representation_chunk_generator(
                num_mini_batches=2,
                num_epochs=1,
                chunk_length=2,
                student_only=True,
            )
        )
        self.assertTrue(batches)
        for batch in batches:
            self.assertEqual(batch.observations.batch_size[0], 2)
            self.assertFalse(batch.teacher_mask.any())

    def test_teacher_student_mask_balances_contiguous_terrain_type_ranges(self) -> None:
        alg, _ = self.build_algorithm()
        mask = alg._make_teacher_mask(2048, teacher_student_ratio=1.0)

        self.assertIsNotNone(mask)
        self.assertEqual(int(mask.sum()), 1024)
        terrain_type_counts = (342, 341, 341, 171, 171, 682)
        start = 0
        for count in terrain_type_counts:
            terrain_mask = mask[start : start + count]
            num_teachers = int(terrain_mask.sum())
            num_students = count - num_teachers
            self.assertLessEqual(abs(num_teachers - num_students), 1)
            self.assertGreater(num_teachers, 0)
            self.assertGreater(num_students, 0)
            start += count
        self.assertEqual(start, mask.numel())

    def test_teacher_student_mask_preserves_arbitrary_ratio_count(self) -> None:
        alg, _ = self.build_algorithm()
        for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
            mask = alg._make_teacher_mask(101, teacher_student_ratio=ratio)
            expected = min(max(int(101 * ratio / (ratio + 1.0)), 1), 100)
            self.assertEqual(int(mask.sum()), expected)

    def clone_named_parameters(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in module.named_parameters()}

    def any_param_changed(self, before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
        return any(not torch.equal(before[name], param) for name, param in module.named_parameters())


if __name__ == "__main__":
    unittest.main()
