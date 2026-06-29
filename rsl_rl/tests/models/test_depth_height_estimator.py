# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the standalone depth-to-height estimator."""

from __future__ import annotations

import unittest

import torch

from rsl_rl.models import DepthHeightEstimator


BATCH_SIZE = 4
PROPRIO_HISTORY_DIM = 165
DEPTH_SHAPE = (5, 32, 24)
HEIGHT_DIM = 121
HEIGHT_LATENT_DIM = 16
GRU_HIDDEN_DIM = 32


class DepthHeightEstimatorTests(unittest.TestCase):
    def make_estimator(self) -> DepthHeightEstimator:
        return DepthHeightEstimator(
            proprio_history_dim=PROPRIO_HISTORY_DIM,
            depth_shape=DEPTH_SHAPE,
            height_dim=HEIGHT_DIM,
            height_latent_dim=HEIGHT_LATENT_DIM,
            proprio_feature_dim=16,
            depth_feature_dim=16,
            gru_hidden_dim=GRU_HIDDEN_DIM,
            proprio_hidden_dims=(32,),
            depth_channels=(8, 8),
            decoder_hidden_dims=(32,),
        )

    def test_forward_shapes_without_initial_hidden_state(self) -> None:
        estimator = self.make_estimator()
        proprio_history = torch.randn(BATCH_SIZE, PROPRIO_HISTORY_DIM)
        depth = torch.randn(BATCH_SIZE, *DEPTH_SHAPE)

        height_latent, height_hat, next_hidden = estimator(proprio_history, depth)

        self.assertEqual(height_latent.shape, (BATCH_SIZE, HEIGHT_LATENT_DIM))
        self.assertEqual(height_hat.shape, (BATCH_SIZE, HEIGHT_DIM))
        self.assertEqual(next_hidden.shape, (BATCH_SIZE, GRU_HIDDEN_DIM))
        self.assertTrue(torch.allclose(height_latent.norm(dim=-1), torch.ones(BATCH_SIZE), atol=1e-6))

    def test_forward_uses_provided_hidden_state_and_updates_it(self) -> None:
        estimator = self.make_estimator()
        proprio_history = torch.randn(BATCH_SIZE, PROPRIO_HISTORY_DIM)
        depth = torch.randn(BATCH_SIZE, *DEPTH_SHAPE)
        hidden = estimator.get_initial_state(BATCH_SIZE, dtype=proprio_history.dtype)

        _, _, next_hidden = estimator(proprio_history, depth, hidden)

        self.assertEqual(next_hidden.shape, hidden.shape)
        self.assertFalse(torch.equal(next_hidden, hidden))

    def test_invalid_shapes_fail_loudly(self) -> None:
        estimator = self.make_estimator()
        depth = torch.randn(BATCH_SIZE, *DEPTH_SHAPE)
        proprio_history = torch.randn(BATCH_SIZE, PROPRIO_HISTORY_DIM)

        with self.assertRaisesRegex(ValueError, "proprio_history"):
            estimator(torch.randn(BATCH_SIZE, 5, PROPRIO_HISTORY_DIM), depth)
        with self.assertRaisesRegex(ValueError, "expected proprio_history dim"):
            estimator(torch.randn(BATCH_SIZE, PROPRIO_HISTORY_DIM + 1), depth)
        with self.assertRaisesRegex(ValueError, "expected depth shape"):
            estimator(proprio_history, torch.randn(BATCH_SIZE, 1, *DEPTH_SHAPE))
        with self.assertRaisesRegex(ValueError, "expected hidden_state shape"):
            estimator(proprio_history, depth, torch.randn(BATCH_SIZE, GRU_HIDDEN_DIM + 1))


if __name__ == "__main__":
    unittest.main()
