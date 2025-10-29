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
    fourier_anti_symmetric_kernel,
    generate_chebyshev_spaced_ijk,
    generate_chebyshev_spaced_ijk_batch,
    generate_hermit_impulses_dense,
    generate_hermit_impulses_dense_batch,
    has_any_symmetry,
    has_any_symmetry_witnessed,
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
            batch_size, num_candidates, volume_shape, kernel_size, impulse_value=impulse_value, device=device
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


class TestHasAnySymmetry(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank1_symmetric(self, device):
        """Test symmetric rank-1 tensors."""
        # Constant tensor is symmetric under any transformation
        constant = torch.ones(5, device=device)
        self.assertTrue(has_any_symmetry(constant))

        # Palindromic tensor is symmetric under flip
        palindromic = torch.tensor([1, 2, 3, 2, 1], device=device)
        self.assertTrue(has_any_symmetry(palindromic))

        # Even-length palindromic tensor
        palindromic_even = torch.tensor([1, 2, 3, 3, 2, 1], device=device)
        self.assertTrue(has_any_symmetry(palindromic_even))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank1_asymmetric(self, device):
        """Test asymmetric rank-1 tensors."""
        # Strictly increasing tensor is asymmetric
        increasing = torch.tensor([1, 2, 3, 4, 5], device=device)
        self.assertFalse(has_any_symmetry(increasing))

        # Random-like pattern is asymmetric
        random_like = torch.tensor([1, 3, 2, 5, 4], device=device)
        self.assertFalse(has_any_symmetry(random_like))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank2_symmetric(self, device):
        """Test symmetric rank-2 tensors."""
        # Constant matrix is symmetric under any transformation
        constant = torch.ones(3, 4, device=device)
        self.assertTrue(has_any_symmetry(constant))

        # Diagonal matrix is symmetric under transpose
        diagonal = torch.diag(torch.tensor([1, 2, 3], device=device))
        self.assertTrue(has_any_symmetry(diagonal))

        # Matrix symmetric about center
        symmetric = torch.tensor([[1, 2, 1], [2, 3, 2], [1, 2, 1]], device=device)
        self.assertTrue(has_any_symmetry(symmetric))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank2_asymmetric(self, device):
        """Test asymmetric rank-2 tensors."""
        # Strictly increasing matrix is asymmetric
        increasing = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
        self.assertFalse(has_any_symmetry(increasing))

        # Random-like pattern is asymmetric
        random_like = torch.tensor([[1, 3, 2], [5, 4, 6], [2, 1, 3]], device=device)
        self.assertFalse(has_any_symmetry(random_like))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank3_symmetric(self, device):
        """Test symmetric rank-3 tensors."""
        # Constant tensor is symmetric
        constant = torch.ones(2, 3, 4, device=device)
        self.assertTrue(has_any_symmetry(constant))

        # Tensor symmetric in first two dimensions
        symmetric_2d = torch.randn(3, 3, 2, device=device)
        symmetric_2d = symmetric_2d + symmetric_2d.transpose(0, 1)  # Make symmetric
        self.assertTrue(has_any_symmetry(symmetric_2d))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank3_asymmetric(self, device):
        """Test asymmetric rank-3 tensors."""
        # Strictly increasing tensor is asymmetric
        increasing = torch.arange(24, device=device).reshape(2, 3, 4)
        self.assertFalse(has_any_symmetry(increasing))

        # Random tensor is typically asymmetric
        random_tensor = torch.randn(2, 3, 4, device=device)
        self.assertFalse(has_any_symmetry(random_tensor))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_rank4_and_5(self, device):
        """Test higher rank tensors."""
        # Rank 4: Constant tensor is symmetric
        constant_4d = torch.ones(2, 3, 4, 5, device=device)
        self.assertTrue(has_any_symmetry(constant_4d))

        # Rank 4: Asymmetric tensor
        increasing_4d = torch.arange(120, device=device).reshape(2, 3, 4, 5)
        self.assertFalse(has_any_symmetry(increasing_4d))

        # Rank 5: Constant tensor is symmetric
        constant_5d = torch.ones(2, 2, 3, 3, 4, device=device)
        self.assertTrue(has_any_symmetry(constant_5d))

        # Rank 5: Asymmetric tensor
        increasing_5d = torch.arange(144, device=device).reshape(2, 2, 3, 3, 4)
        self.assertFalse(has_any_symmetry(increasing_5d))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_length1_axes(self, device):
        """Test behavior with length-1 axes."""
        # Tensor with length-1 axes should be symmetric under axis reordering
        tensor_1d = torch.tensor([[1, 2, 3]], device=device)  # Shape (1, 3)
        self.assertTrue(has_any_symmetry(tensor_1d))

        # Test ignore_length1_axes parameter
        tensor_1d = torch.tensor([[1, 2, 3]], device=device)
        self.assertTrue(has_any_symmetry(tensor_1d, ignore_length1_axes=True))
        self.assertFalse(has_any_symmetry(tensor_1d, ignore_length1_axes=False))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_witnessed(self, device):
        """Test witness functionality."""
        # Constant tensor should return witness
        constant = torch.ones(3, 4, device=device)
        has_sym, witness = has_any_symmetry_witnessed(constant)
        self.assertTrue(has_sym)
        self.assertIsNotNone(witness)
        self.assertIn("perm", witness)
        self.assertIn("flip_after_permute", witness)
        self.assertIn("original_shape", witness)
        self.assertIn("transformed_shape", witness)

        # Asymmetric tensor should return empty dict witness
        increasing = torch.arange(12, device=device).reshape(3, 4)
        has_sym, witness = has_any_symmetry_witnessed(increasing)
        self.assertFalse(has_sym)
        self.assertEqual(witness, {})

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_tolerance(self, device):
        """Test tolerance parameters."""
        # Create tensor that's almost symmetric
        base = torch.tensor([[1, 2, 3], [2, 1, 2], [3, 2, 1]], device=device, dtype=torch.float32)
        # Use deterministic noise to ensure test reproducibility
        torch.manual_seed(42)
        noisy = base + 1e-7 * torch.randn_like(base)

        # Should be symmetric with default tolerance
        self.assertTrue(has_any_symmetry(noisy))

        # Should not be symmetric with very strict tolerance
        self.assertFalse(has_any_symmetry(noisy, rtol=1e-10, atol=1e-10))

    @parameterized.expand(all_device_combos)
    def test_has_any_symmetry_edge_cases(self, device):
        """Test edge cases."""
        # Scalar tensor
        scalar = torch.tensor(5.0, device=device)
        self.assertTrue(has_any_symmetry(scalar))

        # Single element tensor
        single = torch.tensor([5.0], device=device)
        self.assertTrue(has_any_symmetry(single))

        # Rank 0 tensor
        rank0 = torch.tensor(5.0, device=device)
        self.assertTrue(has_any_symmetry(rank0))


class TestFourierAntiSymmetricKernel(unittest.TestCase):
    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_rank1(self, device):
        """Test rank-1 kernels (1D convolution)."""
        # Test basic 1D kernel
        kernel_1d = fourier_anti_symmetric_kernel((5,), device=device)
        self.assertEqual(kernel_1d.shape, (5,))
        self.assertFalse(has_any_symmetry(kernel_1d))

        # Test different sizes
        for size in [1, 3, 7, 11]:
            kernel = fourier_anti_symmetric_kernel((size,), device=device)
            self.assertEqual(kernel.shape, (size,))
            if size > 1:  # Size-1 axes can't reveal flips
                self.assertFalse(has_any_symmetry(kernel))

        # Test determinism
        kernel1 = fourier_anti_symmetric_kernel((7,), device=device)
        kernel2 = fourier_anti_symmetric_kernel((7,), device=device)
        self.assertTrue(torch.allclose(kernel1, kernel2))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_rank2(self, device):
        """Test rank-2 kernels (2D convolution)."""
        # Test basic 2D kernel
        kernel_2d = fourier_anti_symmetric_kernel((3, 5), device=device)
        self.assertEqual(kernel_2d.shape, (3, 5))
        self.assertFalse(has_any_symmetry(kernel_2d))

        # Test square kernel
        kernel_square = fourier_anti_symmetric_kernel((5, 5), device=device)
        self.assertEqual(kernel_square.shape, (5, 5))
        self.assertFalse(has_any_symmetry(kernel_square))

        # Test different sizes
        for h, w in [(3, 3), (3, 5), (5, 3), (7, 7)]:
            kernel = fourier_anti_symmetric_kernel((h, w), device=device)
            self.assertEqual(kernel.shape, (h, w))
            self.assertFalse(has_any_symmetry(kernel))

        # Test determinism
        kernel1 = fourier_anti_symmetric_kernel((4, 6), device=device)
        kernel2 = fourier_anti_symmetric_kernel((4, 6), device=device)
        self.assertTrue(torch.allclose(kernel1, kernel2))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_rank3(self, device):
        """Test rank-3 kernels (3D convolution)."""
        # Test basic 3D kernel
        kernel_3d = fourier_anti_symmetric_kernel((3, 5, 7), device=device)
        self.assertEqual(kernel_3d.shape, (3, 5, 7))
        self.assertFalse(has_any_symmetry(kernel_3d))

        # Test cubic kernel
        kernel_cubic = fourier_anti_symmetric_kernel((5, 5, 5), device=device)
        self.assertEqual(kernel_cubic.shape, (5, 5, 5))
        self.assertFalse(has_any_symmetry(kernel_cubic))

        # Test different sizes
        for d, h, w in [(3, 3, 3), (3, 5, 7), (5, 3, 7), (7, 5, 3)]:
            kernel = fourier_anti_symmetric_kernel((d, h, w), device=device)
            self.assertEqual(kernel.shape, (d, h, w))
            self.assertFalse(has_any_symmetry(kernel))

        # Test determinism
        kernel1 = fourier_anti_symmetric_kernel((4, 6, 8), device=device)
        kernel2 = fourier_anti_symmetric_kernel((4, 6, 8), device=device)
        self.assertTrue(torch.allclose(kernel1, kernel2))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_rank4(self, device):
        """Test rank-4 kernels (convolution with channels)."""
        # Test basic 4D kernel: (out_channels, in_channels, height, width)
        kernel_4d = fourier_anti_symmetric_kernel((16, 8, 3, 3), device=device)
        self.assertEqual(kernel_4d.shape, (16, 8, 3, 3))
        self.assertFalse(has_any_symmetry(kernel_4d))

        # Test different channel configurations
        for out_ch, in_ch, h, w in [(1, 1, 3, 3), (4, 2, 5, 5), (8, 16, 3, 7)]:
            kernel = fourier_anti_symmetric_kernel((out_ch, in_ch, h, w), device=device)
            self.assertEqual(kernel.shape, (out_ch, in_ch, h, w))
            self.assertFalse(has_any_symmetry(kernel))

        # Test determinism
        kernel1 = fourier_anti_symmetric_kernel((4, 8, 5, 5), device=device)
        kernel2 = fourier_anti_symmetric_kernel((4, 8, 5, 5), device=device)
        self.assertTrue(torch.allclose(kernel1, kernel2))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_rank5(self, device):
        """Test rank-5 kernels (3D convolution with channels)."""
        # Test basic 5D kernel: (out_channels, in_channels, depth, height, width)
        kernel_5d = fourier_anti_symmetric_kernel((8, 4, 3, 3, 3), device=device)
        self.assertEqual(kernel_5d.shape, (8, 4, 3, 3, 3))
        self.assertFalse(has_any_symmetry(kernel_5d))

        # Test different configurations
        for out_ch, in_ch, d, h, w in [(1, 1, 3, 3, 3), (2, 4, 5, 5, 5), (4, 8, 3, 7, 5)]:
            kernel = fourier_anti_symmetric_kernel((out_ch, in_ch, d, h, w), device=device)
            self.assertEqual(kernel.shape, (out_ch, in_ch, d, h, w))
            self.assertFalse(has_any_symmetry(kernel))

        # Test determinism
        kernel1 = fourier_anti_symmetric_kernel((2, 4, 5, 5, 5), device=device)
        kernel2 = fourier_anti_symmetric_kernel((2, 4, 5, 5, 5), device=device)
        self.assertTrue(torch.allclose(kernel1, kernel2))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_axis_identity(self, device):
        """Test that kernels distinguish different axes."""
        # Test that swapping equal-sized axes changes the kernel
        kernel_2d = fourier_anti_symmetric_kernel((5, 5), device=device)
        kernel_2d_transposed = kernel_2d.transpose(0, 1)

        # Should be different (not symmetric under transpose)
        self.assertFalse(torch.allclose(kernel_2d, kernel_2d_transposed))

        # Test 3D case
        kernel_3d = fourier_anti_symmetric_kernel((5, 5, 5), device=device)
        kernel_3d_permuted = kernel_3d.permute(1, 0, 2)  # Swap first two axes

        # Should be different
        self.assertFalse(torch.allclose(kernel_3d, kernel_3d_permuted))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_orientation_sensitivity(self, device):
        """Test that kernels are sensitive to axis flips."""
        # Test 1D flip sensitivity
        kernel_1d = fourier_anti_symmetric_kernel((7,), device=device)
        kernel_1d_flipped = kernel_1d.flip(0)

        # Should be different
        self.assertFalse(torch.allclose(kernel_1d, kernel_1d_flipped))

        # Test 2D flip sensitivity
        kernel_2d = fourier_anti_symmetric_kernel((5, 7), device=device)
        kernel_2d_flipped_h = kernel_2d.flip(0)
        kernel_2d_flipped_w = kernel_2d.flip(1)
        kernel_2d_flipped_both = kernel_2d.flip((0, 1))

        # All should be different
        self.assertFalse(torch.allclose(kernel_2d, kernel_2d_flipped_h))
        self.assertFalse(torch.allclose(kernel_2d, kernel_2d_flipped_w))
        self.assertFalse(torch.allclose(kernel_2d, kernel_2d_flipped_both))

        # Test 3D flip sensitivity
        kernel_3d = fourier_anti_symmetric_kernel((3, 5, 7), device=device)
        kernel_3d_flipped_d = kernel_3d.flip(0)
        kernel_3d_flipped_h = kernel_3d.flip(1)
        kernel_3d_flipped_w = kernel_3d.flip(2)

        # All should be different
        self.assertFalse(torch.allclose(kernel_3d, kernel_3d_flipped_d))
        self.assertFalse(torch.allclose(kernel_3d, kernel_3d_flipped_h))
        self.assertFalse(torch.allclose(kernel_3d, kernel_3d_flipped_w))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_dtype_support(self, device):
        """Test different dtype support."""
        # Test float32 (default)
        kernel_f32 = fourier_anti_symmetric_kernel((3, 5), dtype=torch.float32, device=device)
        self.assertEqual(kernel_f32.dtype, torch.float32)

        # Test float64
        kernel_f64 = fourier_anti_symmetric_kernel((3, 5), dtype=torch.float64, device=device)
        self.assertEqual(kernel_f64.dtype, torch.float64)

        # Test float16
        kernel_f16 = fourier_anti_symmetric_kernel((3, 5), dtype=torch.float16, device=device)
        self.assertEqual(kernel_f16.dtype, torch.float16)

        # All should be asymmetric
        self.assertFalse(has_any_symmetry(kernel_f32))
        self.assertFalse(has_any_symmetry(kernel_f64))
        self.assertFalse(has_any_symmetry(kernel_f16))

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_numerical_properties(self, device):
        """Test numerical properties of the kernel."""
        # Test that kernel values are bounded and reasonable
        kernel = fourier_anti_symmetric_kernel((5, 7, 9), device=device)

        # Values should be bounded (not too large)
        self.assertLess(torch.max(torch.abs(kernel)).item(), 10.0)

        # Values should not be all zero
        self.assertGreater(torch.max(torch.abs(kernel)).item(), 0.0)

        # Test differentiability (should not raise error)
        kernel.requires_grad_(True)
        loss = torch.sum(kernel**2)
        loss.backward()
        self.assertIsNotNone(kernel.grad)

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_edge_cases(self, device):
        """Test edge cases."""
        # Test minimum valid rank
        kernel_1d = fourier_anti_symmetric_kernel((1,), device=device)
        self.assertEqual(kernel_1d.shape, (1,))

        # Test maximum valid rank
        kernel_5d = fourier_anti_symmetric_kernel((1, 1, 1, 1, 1), device=device)
        self.assertEqual(kernel_5d.shape, (1, 1, 1, 1, 1))

        # Test invalid ranks
        with self.assertRaises(ValueError):
            fourier_anti_symmetric_kernel((), device=device)  # Rank 0

        with self.assertRaises(ValueError):
            fourier_anti_symmetric_kernel((1, 2, 3, 4, 5, 6), device=device)  # Rank 6

        # Test invalid dimensions
        with self.assertRaises(ValueError):
            fourier_anti_symmetric_kernel((0, 3), device=device)  # Zero dimension

        with self.assertRaises(ValueError):
            fourier_anti_symmetric_kernel((3, -1), device=device)  # Negative dimension

    @parameterized.expand(all_device_combos)
    def test_fourier_anti_symmetric_kernel_common_conv_shapes(self, device):
        """Test common convolution kernel shapes."""
        # Common 2D convolution shapes
        common_2d_shapes = [
            (1, 1, 3, 3),  # Basic 3x3
            (1, 1, 5, 5),  # 5x5
            (1, 1, 7, 7),  # 7x7
            (16, 8, 3, 3),  # With channels
            (32, 16, 5, 5),  # Larger channels
        ]

        for shape in common_2d_shapes:
            kernel = fourier_anti_symmetric_kernel(shape, device=device)
            self.assertEqual(kernel.shape, shape)
            self.assertFalse(has_any_symmetry(kernel))

        # Common 3D convolution shapes
        common_3d_shapes = [
            (1, 1, 3, 3, 3),  # Basic 3x3x3
            (1, 1, 5, 5, 5),  # 5x5x5
            (8, 4, 3, 3, 3),  # With channels
            (16, 8, 5, 5, 5),  # Larger channels
        ]

        for shape in common_3d_shapes:
            kernel = fourier_anti_symmetric_kernel(shape, device=device)
            self.assertEqual(kernel.shape, shape)
            self.assertFalse(has_any_symmetry(kernel))
