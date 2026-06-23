# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the visual representation actor-critic skeleton."""

from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict

from rsl_rl.models import VisualRepresentationActorCritic


BATCH_SIZE = 4
ACTOR_DIM = 33
ACTOR_HISTORY_DIM = 165
CRITIC_DIM = 170
DEPTH_SHAPE = (5, 32, 24)
HEIGHT_SCAN_START = 49
HEIGHT_DIM = 121
LATENT_DIM = 8
HEIGHT_LATENT_DIM = 8
NUM_ACTIONS = 8


class VisualRepresentationActorCriticTests(unittest.TestCase):
    def make_obs(self, *, include_critic: bool = True, include_depth: bool = True) -> TensorDict:
        data = {
            "actor": torch.randn(BATCH_SIZE, ACTOR_DIM),
            "actor_history": torch.randn(BATCH_SIZE, ACTOR_HISTORY_DIM),
        }
        if include_critic:
            data["critic"] = torch.randn(BATCH_SIZE, CRITIC_DIM)
        if include_depth:
            data["depth_camera"] = torch.randn(BATCH_SIZE, *DEPTH_SHAPE)
        return TensorDict(data, batch_size=[BATCH_SIZE])

    def make_model(self) -> VisualRepresentationActorCritic:
        obs = self.make_obs()
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
            hidden_dims=(32,),
            encoder_hidden_dims=(32,),
            latent_dim=LATENT_DIM,
            height_latent_dim=HEIGHT_LATENT_DIM,
            height_scan_start=HEIGHT_SCAN_START,
            height_dim=HEIGHT_DIM,
            height_teacher_hidden_dims=(32,),
            height_proprio_feature_dim=16,
            height_depth_feature_dim=16,
            height_gru_hidden_dim=32,
            height_proprio_hidden_dims=(32,),
            height_depth_channels=(8, 8),
            height_decoder_hidden_dims=(32,),
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )

    def test_student_teacher_value_and_height_loss_shapes(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        student_actions = model(obs)
        teacher_actions = model.act_teacher(obs, stochastic_output=True)
        values = model.evaluate_teacher(obs)
        losses = model.compute_visual_height_loss(obs)

        self.assertEqual(student_actions.shape, (BATCH_SIZE, NUM_ACTIONS))
        self.assertEqual(teacher_actions.shape, (BATCH_SIZE, NUM_ACTIONS))
        self.assertEqual(values.shape, (BATCH_SIZE, 1))
        self.assertEqual(set(losses), {"height_latent", "height_reconstruction", "height_total"})
        for loss in losses.values():
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))

    def test_visual_representation_loss_shapes_and_compatibility(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        losses = model.compute_visual_representation_loss(obs)
        representation_loss = model.compute_representation_loss(obs)

        self.assertEqual(
            set(losses),
            {
                "privileged_latent",
                "height_latent",
                "height_reconstruction",
                "height_total",
                "representation_total",
            },
        )
        for loss in losses.values():
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(representation_loss, losses["representation_total"]))

    def test_visual_representation_loss_updates_only_student_estimators(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        model.zero_grad()
        losses = model.compute_visual_representation_loss(obs)
        losses["representation_total"].backward()

        self.assertTrue(self.has_any_grad(model.proprio_encoder.parameters()))
        self.assertTrue(self.has_any_grad(model.height_pair.student_height_estimator.parameters()))
        self.assertFalse(self.has_any_grad(model.privileged_encoder.parameters()))
        self.assertFalse(self.has_any_grad(model.height_pair.teacher_height_encoder.parameters()))

    def test_actor_head_uses_privileged_and_height_latents(self) -> None:
        model = self.make_model()

        self.assertEqual(model.actor_head[0].in_features, ACTOR_DIM + LATENT_DIM + HEIGHT_LATENT_DIM)

    def test_student_inference_does_not_require_critic_observations(self) -> None:
        model = self.make_model()
        inference_obs = self.make_obs(include_critic=False)

        actions = model(inference_obs)

        self.assertEqual(actions.shape, (BATCH_SIZE, NUM_ACTIONS))

    def test_student_inference_requires_depth_observations(self) -> None:
        model = self.make_model()
        inference_obs = self.make_obs(include_critic=False, include_depth=False)

        with self.assertRaisesRegex(ValueError, "depth observation group"):
            model(inference_obs)

    def test_teacher_paths_require_critic_observations(self) -> None:
        model = self.make_model()
        inference_obs = self.make_obs(include_critic=False)

        with self.assertRaisesRegex(ValueError, "critic"):
            model.act_teacher(inference_obs)

    def has_any_grad(self, parameters) -> bool:
        return any(param.grad is not None and torch.any(param.grad != 0) for param in parameters)


if __name__ == "__main__":
    unittest.main()
