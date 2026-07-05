# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the visual representation actor-critic skeleton."""

from __future__ import annotations

import tempfile
import unittest

import onnx
import torch
from tensordict import TensorDict

from rsl_rl.models import VisualRepresentationActorCritic


BATCH_SIZE = 4
ACTOR_DIM = 33
HISTORY_LENGTH = 5
CRITIC_DIM = 170
DEPTH_SHAPE = (1, 32, 24)
HEIGHT_SCAN_START = 49
HEIGHT_DIM = 121
LATENT_DIM = 8
HEIGHT_LATENT_DIM = 8
NUM_ACTIONS = 8


class VisualRepresentationActorCriticTests(unittest.TestCase):
    def make_obs(self, *, include_critic: bool = True, include_depth: bool = True) -> TensorDict:
        data = {
            "actor": torch.randn(BATCH_SIZE, ACTOR_DIM),
            "actor_history": torch.randn(BATCH_SIZE, HISTORY_LENGTH, ACTOR_DIM),
            "height_scan": torch.randn(BATCH_SIZE, HEIGHT_DIM),
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
                "teacher_actor": ["actor"],
                "critic": ["critic"],
                "student_history": ["actor_history"],
                "privileged_encoder": ["critic"],
                "depth_encoder": ["depth_camera"],
                "height_encoder": ["height_scan"],
            },
            NUM_ACTIONS,
            hidden_dims=(32,),
            encoder_hidden_dims=(32,),
            latent_dim=LATENT_DIM,
            height_latent_dim=HEIGHT_LATENT_DIM,
            height_scan_start=None,
            height_dim=HEIGHT_DIM,
            height_teacher_hidden_dims=(32,),
            height_proprio_feature_dim=16,
            height_depth_feature_dim=16,
            height_gru_hidden_dim=32,
            height_proprio_hidden_dims=(32,),
            height_depth_channels=(8, 8),
            height_decoder_hidden_dims=(32,),
            privileged_decoder_hidden_dims=(32,),
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )

    def test_student_teacher_value_and_height_loss_shapes(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        student_actions = model(obs)
        teacher_actions = model.act_teacher(obs, stochastic_output=True)
        values = model.evaluate_teacher(obs)
        losses = model.compute_visual_height_loss(obs)
        _, privileged_hat = model.get_student_privileged_output(obs)

        self.assertEqual(student_actions.shape, (BATCH_SIZE, NUM_ACTIONS))
        self.assertEqual(teacher_actions.shape, (BATCH_SIZE, NUM_ACTIONS))
        self.assertEqual(values.shape, (BATCH_SIZE, 1))
        self.assertEqual(privileged_hat.shape, (BATCH_SIZE, CRITIC_DIM))
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
                "privileged_reconstruction",
                "privileged_total",
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
        self.assertTrue(
            torch.allclose(
                losses["privileged_total"],
                losses["privileged_latent"] + losses["privileged_reconstruction"],
            )
        )
        self.assertTrue(
            torch.allclose(
                losses["representation_total"],
                losses["privileged_total"] + losses["height_total"],
            )
        )

    def test_visual_representation_loss_updates_only_student_estimators(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        model.zero_grad()
        losses = model.compute_visual_representation_loss(obs)
        losses["representation_total"].backward()

        self.assertTrue(self.has_any_grad(model.proprio_encoder.parameters()))
        self.assertTrue(self.has_any_grad(model.privileged_decoder.parameters()))
        self.assertTrue(self.has_any_grad(model.height_pair.student_height_estimator.parameters()))
        self.assertFalse(self.has_any_grad(model.privileged_encoder.parameters()))
        self.assertFalse(self.has_any_grad(model.height_pair.teacher_height_encoder.parameters()))

    def test_actor_head_uses_privileged_and_height_latents(self) -> None:
        model = self.make_model()

        self.assertEqual(model.actor_head[0].in_features, ACTOR_DIM + LATENT_DIM + HEIGHT_LATENT_DIM)

    def test_student_actor_uses_latest_noisy_history_frame(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        self.assertTrue(
            torch.equal(
                model.get_student_actor_obs(obs),
                obs["actor_history"][:, -1, :],
            )
        )
        self.assertTrue(
            torch.equal(
                model.get_teacher_actor_obs(obs),
                obs["actor"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.get_proprio_obs(obs),
                obs["actor_history"].flatten(start_dim=1),
            )
        )

    def test_student_inference_does_not_require_critic_observations(self) -> None:
        model = self.make_model()
        inference_obs = self.make_obs(include_critic=False)

        actions = model(inference_obs)

        self.assertEqual(actions.shape, (BATCH_SIZE, NUM_ACTIONS))

    def test_student_inference_does_not_execute_privileged_decoder(self) -> None:
        model = self.make_model()
        inference_obs = self.make_obs(include_critic=False)

        def fail_if_called(_):
            raise AssertionError("privileged decoder is training-only")

        model.privileged_decoder.forward = fail_if_called
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

    def test_height_scan_uses_explicit_observation_group(self) -> None:
        model = self.make_model()
        obs = self.make_obs()
        obs["height_scan"] = torch.arange(BATCH_SIZE * HEIGHT_DIM).reshape(BATCH_SIZE, HEIGHT_DIM).float()

        self.assertTrue(torch.equal(model.get_height_scan(obs), obs["height_scan"]))

    def test_mixed_actor_selects_teacher_and_student_paths(self) -> None:
        model = self.make_model()
        obs = self.make_obs()
        hidden_state = torch.zeros(BATCH_SIZE, model.height_pair.student_height_estimator.gru_hidden_dim)
        teacher_mask = torch.tensor([True, True, False, False])

        teacher_actions = model.act_teacher(obs)
        student_actions = model(obs, hidden_state=hidden_state)
        mixed_actions = model.act_mixed(obs, teacher_mask, hidden_state=hidden_state)

        expected = torch.where(teacher_mask[:, None], teacher_actions, student_actions)
        self.assertTrue(torch.allclose(mixed_actions, expected))

    def test_mixed_ppo_path_detaches_student_estimators(self) -> None:
        model = self.make_model()
        obs = self.make_obs()
        hidden_state = torch.zeros(BATCH_SIZE, model.height_pair.student_height_estimator.gru_hidden_dim)
        student_mask = torch.zeros(BATCH_SIZE, dtype=torch.bool)

        model.zero_grad()
        actions = model.act_mixed(obs, student_mask, hidden_state=hidden_state)
        values = model.evaluate_mixed(obs, student_mask, hidden_state=hidden_state)
        (actions.sum() + values.sum()).backward()

        self.assertTrue(self.has_any_grad(model.actor_head.parameters()))
        self.assertTrue(self.has_any_grad(model.critic_head.parameters()))
        self.assertFalse(self.has_any_grad(model.proprio_encoder.parameters()))
        self.assertFalse(self.has_any_grad(model.height_pair.student_height_estimator.parameters()))

    def test_student_gru_state_persists_and_resets_per_environment(self) -> None:
        model = self.make_model()
        obs = self.make_obs()

        model(obs)
        first_state = model.get_hidden_state().clone()
        model(obs)
        second_state = model.get_hidden_state().clone()
        self.assertFalse(torch.equal(first_state, second_state))

        model.reset(torch.tensor([True, False, False, True]))
        reset_state = model.get_hidden_state()
        self.assertTrue(torch.equal(reset_state[[0, 3]], torch.zeros_like(reset_state[[0, 3]])))
        self.assertTrue(torch.equal(reset_state[[1, 2]], second_state[[1, 2]]))

        model.reset()
        self.assertIsNone(model.get_hidden_state())

    def test_visual_student_onnx_wrapper_matches_policy_and_exports(self) -> None:
        model = self.make_model()
        model.eval()
        obs = self.make_obs()
        hidden_state = torch.zeros(BATCH_SIZE, model.height_pair.student_height_estimator.gru_hidden_dim)
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

        with torch.inference_mode():
            expected_actions = model(obs, hidden_state=hidden_state)
            actions, hidden_state_out = onnx_model(
                obs["actor_history"],
                obs["depth_camera"],
                hidden_state,
            )

        self.assertTrue(torch.allclose(actions, expected_actions, atol=1e-6))
        self.assertEqual(hidden_state_out.shape, hidden_state.shape)
        self.assertEqual(onnx_model.input_names, ["student_history", "depth", "hidden_state_in"])
        self.assertEqual(onnx_model.output_names, ["actions", "hidden_state_out"])
        self.assertFalse(hasattr(onnx_model, "privileged_decoder"))
        self.assertFalse(hasattr(onnx_model, "height_decoder"))

        with tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                output_file.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
                external_data=onnx_model.use_external_data,
            )
            exported_model = onnx.load(output_file.name)
            onnx.checker.check_model(exported_model)
            self.assertTrue(all(not initializer.external_data for initializer in exported_model.graph.initializer))

    def has_any_grad(self, parameters) -> bool:
        return any(param.grad is not None and torch.any(param.grad != 0) for param in parameters)


if __name__ == "__main__":
    unittest.main()
