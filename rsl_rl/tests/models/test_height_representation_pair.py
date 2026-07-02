# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the standalone teacher/student height representation pair."""

from __future__ import annotations

import unittest

import torch

from rsl_rl.models import HeightRepresentationPair


BATCH_SIZE = 4
HEIGHT_DIM = 121
PROPRIO_HISTORY_DIM = 165
DEPTH_SHAPE = (1, 32, 24)
HEIGHT_LATENT_DIM = 16
GRU_HIDDEN_DIM = 32


class HeightRepresentationPairTests(unittest.TestCase):
    def make_pair(self) -> HeightRepresentationPair:
        return HeightRepresentationPair(
            height_dim=HEIGHT_DIM,
            proprio_history_dim=PROPRIO_HISTORY_DIM,
            depth_shape=DEPTH_SHAPE,
            height_latent_dim=HEIGHT_LATENT_DIM,
            teacher_hidden_dims=(32,),
            proprio_feature_dim=16,
            depth_feature_dim=16,
            gru_hidden_dim=GRU_HIDDEN_DIM,
            proprio_hidden_dims=(32,),
            depth_channels=(8, 8),
            decoder_hidden_dims=(32,),
        )

    def make_inputs(self):
        return (
            torch.randn(BATCH_SIZE, HEIGHT_DIM),
            torch.randn(BATCH_SIZE, PROPRIO_HISTORY_DIM),
            torch.randn(BATCH_SIZE, *DEPTH_SHAPE),
        )

    def test_forward_and_loss_shapes(self) -> None:
        pair = self.make_pair()
        height_scan, proprio_history, depth = self.make_inputs()

        output = pair(height_scan, proprio_history, depth)
        losses = pair.compute_height_loss(output, height_scan)

        self.assertEqual(output.teacher_height_latent.shape, (BATCH_SIZE, HEIGHT_LATENT_DIM))
        self.assertEqual(output.student_height_latent.shape, (BATCH_SIZE, HEIGHT_LATENT_DIM))
        self.assertEqual(output.height_hat.shape, (BATCH_SIZE, HEIGHT_DIM))
        self.assertEqual(output.next_hidden_state.shape, (BATCH_SIZE, GRU_HIDDEN_DIM))
        self.assertTrue(
            torch.allclose(output.teacher_height_latent.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(output.student_height_latent.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6)
        )
        self.assertEqual(set(losses), {"height_latent", "height_reconstruction", "height_total"})
        for loss in losses.values():
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))

    def test_height_loss_detaches_teacher_height_latent_target(self) -> None:
        pair = self.make_pair()
        height_scan, proprio_history, depth = self.make_inputs()

        output = pair(height_scan, proprio_history, depth)
        losses = pair.compute_height_loss(output, height_scan)
        pair.zero_grad()
        losses["height_total"].backward()

        self.assertTrue(all(param.grad is None for param in pair.teacher_height_encoder.parameters()))
        self.assertTrue(any(param.grad is not None for param in pair.student_height_estimator.parameters()))

    def test_invalid_height_scan_shapes_fail_loudly(self) -> None:
        pair = self.make_pair()
        height_scan, proprio_history, depth = self.make_inputs()

        with self.assertRaisesRegex(ValueError, "height_scan"):
            pair(torch.randn(BATCH_SIZE, 1, HEIGHT_DIM), proprio_history, depth)
        with self.assertRaisesRegex(ValueError, "expected height_scan dim"):
            pair(torch.randn(BATCH_SIZE, HEIGHT_DIM + 1), proprio_history, depth)
        output = pair(height_scan, proprio_history, depth)
        with self.assertRaisesRegex(ValueError, "expected height_scan dim"):
            pair.compute_height_loss(output, torch.randn(BATCH_SIZE, HEIGHT_DIM + 1))


if __name__ == "__main__":
    unittest.main()
