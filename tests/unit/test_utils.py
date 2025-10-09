#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import io
import sys
import time
import unittest

import pytest
import torch
from fvdb.utils.tests import (
    ScopedTimer,
    generate_chebyshev_spaced_ijk,
    generate_chebyshev_spaced_ijk_batch,
    generate_hermit_impulses_dense,
    generate_hermit_impulses_dense_batch,
)
from parameterized import parameterized


class TestScopedTimer(unittest.TestCase):
    def test_split_without_context_raises(self):
        t = ScopedTimer()
        with pytest.raises(RuntimeError):
            _ = t.split()

    def test_timer_elapsed_time_basic(self):
        with ScopedTimer() as timer:
            time.sleep(0.01)
        assert timer.elapsed_time is not None
        assert timer.elapsed_time > 0.0

    def test_timer_split_positive(self):
        with ScopedTimer() as timer:
            time.sleep(0.002)
            s1 = timer.split()
            time.sleep(0.002)
            s2 = timer.split()
        assert s1 > 0.0 and s2 > 0.0

    def test_timer_prints_message_on_exit_cpu(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            with ScopedTimer(message="CPU scope"):
                time.sleep(0.001)
        finally:
            sys.stdout = old_stdout

        out = buf.getvalue()
        assert "CPU scope:" in out
        # Ensure we printed a floating seconds value
        assert "seconds" in out

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_timer_cuda_timing(self):
        device = torch.device("cuda")
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        with ScopedTimer(cuda=True) as timer:
            _ = a @ b
        assert timer.elapsed_time is not None and timer.elapsed_time > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_timer_prints_message_on_exit_cuda(self):
        device = torch.device("cuda")
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            with ScopedTimer(message="GPU scope", cuda=True):
                _ = a @ b
        finally:
            sys.stdout = old_stdout

        out = buf.getvalue()
        assert "GPU scope:" in out
        assert "seconds" in out


all_device_combos = [
    ["cpu"],
    ["cuda"],
]


class TestGenerateChebyshevSpacedIJK(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_generate_chebyshev_spaced_ijk(self, device):
        num_candidates = 1000
        volume_shape = [100, 200, 300]
        min_separation = [7, 3, 5]
        ijk = generate_chebyshev_spaced_ijk(num_candidates, volume_shape, min_separation, device=device)
        self.assertGreater(len(ijk), 0)
        self.assertLessEqual(len(ijk), num_candidates)
        self.assertTrue(torch.all(ijk >= 0))
        self.assertTrue(torch.all(ijk < torch.tensor(volume_shape, device=device)))

        for point_idx in range(len(ijk)):
            test_i, test_j, test_k = ijk[point_idx].tolist()
            for other_point_idx in range(0, point_idx):
                other_i, other_j, other_k = ijk[other_point_idx].tolist()
                self.assertGreaterEqual(abs(test_i - other_i), min_separation[0])
                self.assertGreaterEqual(abs(test_j - other_j), min_separation[1])
                self.assertGreaterEqual(abs(test_k - other_k), min_separation[2])

    @parameterized.expand(all_device_combos)
    def test_generate_chebyshev_spaced_ijk_batch(self, device):
        num_candidates = 1000
        batch_size = 4
        volume_shapes = [[50, 100, 75], [200, 150, 300], [80, 80, 120], [160, 240, 200]]
        min_separations = [[5, 8, 3], [10, 6, 12], [4, 4, 7], [8, 15, 9]]
        ijks = generate_chebyshev_spaced_ijk_batch(
            batch_size, num_candidates, volume_shapes, min_separations, device=device
        )
        self.assertEqual(len(ijks), batch_size)
        for i in range(batch_size):
            ijk = ijks[i].jdata
            self.assertGreater(len(ijk), 0)
            self.assertLessEqual(len(ijk), num_candidates)
            self.assertTrue(torch.all(ijk >= 0))
            self.assertTrue(torch.all(ijk < torch.tensor(volume_shapes[i], device=device)))
            for point_idx in range(len(ijk)):
                test_i, test_j, test_k = ijk[point_idx].tolist()
                for other_point_idx in range(0, point_idx):
                    other_i, other_j, other_k = ijk[other_point_idx].tolist()
                    self.assertGreaterEqual(abs(test_i - other_i), min_separations[i][0])
                    self.assertGreaterEqual(abs(test_j - other_j), min_separations[i][1])
                    self.assertGreaterEqual(abs(test_k - other_k), min_separations[i][2])


class TestGenerateHermitImpulsesDense(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_generate_hermit_impulses_dense(self, device):
        num_candidates = 500
        volume_shape = [80, 120, 100]
        kernel_size = [5, 7, 5]
        impulse_value = 2.5
        coords, vals = generate_hermit_impulses_dense(
            num_candidates, volume_shape, kernel_size, impulse_value, device=device
        )

        # Check output shapes
        self.assertEqual(vals.shape, tuple(volume_shape))
        self.assertGreater(len(coords), 0)
        self.assertLessEqual(len(coords), num_candidates)

        # Check impulse values are correctly placed
        for i, j, k in coords.tolist():
            self.assertEqual(vals[i, j, k].item(), impulse_value)

        # Check only impulse coordinates have non-zero values
        mask = torch.zeros_like(vals, dtype=torch.bool)
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        self.assertTrue(torch.all(vals[~mask] == 0))

    @parameterized.expand(all_device_combos)
    def test_generate_hermit_impulses_dense_batch(self, device):
        num_candidates = 400
        batch_size = 3
        volume_shape = [60, 80, 70]
        kernel_size = [6, 4, 6]
        impulse_value = 1.5
        coords_batch, vals_batch = generate_hermit_impulses_dense_batch(
            batch_size, num_candidates, volume_shape, kernel_size, impulse_value, device=device
        )

        # Check batch output shape
        self.assertEqual(vals_batch.shape, tuple([batch_size] + volume_shape))
        self.assertEqual(len(coords_batch), batch_size)

        for i in range(batch_size):
            coords = coords_batch[i].jdata
            vals = vals_batch[i]

            self.assertGreater(len(coords), 0)
            self.assertLessEqual(len(coords), num_candidates)

            # Check impulse values are correctly placed
            for ii, jj, kk in coords.tolist():
                self.assertEqual(vals[ii, jj, kk].item(), impulse_value)

            # Check only impulse coordinates have non-zero values
            mask = torch.zeros_like(vals, dtype=torch.bool)
            mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            self.assertTrue(torch.all(vals[~mask] == 0))
