# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the PredGatherIGemm (CUTLASS IGEMM) sparse convolution backend.

The PredGatherIGemm backend has strict constraints compared to the default:
  - CUDA only (SM80+)
  - float32 only (uses TF32 tensor-core arithmetic internally)
  - Forward pass only (no backward, no transpose)
  - Uniform kernel sizes: 3, 5, 7
  - Uniform strides: 1, 2
  - Channel counts must be multiples of 32

Tests compare forward-pass results against:
  - Dense PyTorch conv3d ground truth (with TF32-aware tolerances)
  - GatherScatterDefault backend output (cross-backend validation)
"""

import unittest

import torch
from fvdb.convolution_plan import _PredGatherIGemmBackend
from fvdb.utils.tests import (
    fourier_anti_symmetric_kernel,
    generate_hermit_impulses_dense,
    has_any_symmetry,
)
from fvdb.utils.tests.convolution_utils import (
    conv_ground_truth_strided,
    create_grid_from_coords,
    diagnose_tensor_mismatch,
    disable_tf32,
    get_cluster_edge_aligned,
    sort_coords_by_ijk,
)
from parameterized import parameterized

from fvdb import ConvolutionPlan, JaggedTensor

# =============================================================================
# Configuration
# =============================================================================

# TF32 arithmetic (10-bit mantissa) vs FP32 (23-bit) tolerances.
# These match the C++ gtest tolerances for PredGatherIGemm validation.
TF32_RTOL = 5e-3
TF32_ATOL = 0.5

KERNEL_SIZES = [3, 5, 7]
STRIDES = [1, 2]

KERNEL_STRIDE_COMBOS = [[ks, st] for ks in KERNEL_SIZES for st in STRIDES]
KERNEL_ONLY_COMBOS = [[ks] for ks in KERNEL_SIZES]

IGEMM_CONFIG: dict = {"backend": "pred_gather_igemm"}


def _skip_if_no_sm80() -> bool:
    """Return True if no CUDA SM80+ device is available."""
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_capability(0)[0] < 8


# =============================================================================
# Test Class
# =============================================================================


class TestConvPredGatherIGemm(unittest.TestCase):
    """Forward-only tests for the PredGatherIGemm CUTLASS IGEMM convolution backend."""

    DEVICE = torch.device("cuda", 0)
    DTYPE = torch.float32
    CIN = 32
    COUT = 32
    VOLUME_SHAPE = (71, 34, 58)
    NUM_CANDIDATES = 1000

    def setUp(self):
        if _skip_if_no_sm80():
            self.skipTest("Requires CUDA SM80+")
        torch.random.manual_seed(2024)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _make_diagonal_kernel(self, kernel_size: tuple[int, int, int]) -> torch.Tensor:
        """Build a (COUT, CIN, *kernel_size) weight tensor with the anti-symmetric
        pattern on the diagonal (cout==cin) and zeros elsewhere."""
        kernel_3d = fourier_anti_symmetric_kernel(kernel_size, dtype=self.DTYPE, device=self.DEVICE)
        self.assertFalse(has_any_symmetry(kernel_3d))
        kernel_5d = torch.zeros(self.COUT, self.CIN, *kernel_size, device=self.DEVICE, dtype=self.DTYPE)
        for c in range(min(self.CIN, self.COUT)):
            kernel_5d[c, c] = kernel_3d
        return kernel_5d

    # =========================================================================
    # Single impulse forward (stride=1, vs dense conv3d)
    # =========================================================================

    @parameterized.expand(KERNEL_ONLY_COMBOS)
    def test_single_impulse_forward(self, ks: int):
        """
        Forward pass of a single impulse against dense conv3d ground truth.

        Places one voxel at a known coordinate, convolves with a diagonal
        anti-symmetric kernel, and verifies the sparse IGEMM output matches
        the dense PyTorch conv3d result (within TF32 tolerances).
        """
        kernel_size = (ks, ks, ks)
        half_k = ks // 2

        vol = tuple(ks + 2 for _ in range(3))
        coord_val = half_k + 1
        coord = torch.tensor([coord_val] * 3, device=self.DEVICE, dtype=torch.int32)

        grid = create_grid_from_coords(coord.unsqueeze(0), self.DEVICE)
        features = JaggedTensor(torch.ones((1, self.CIN), device=self.DEVICE, dtype=self.DTYPE))

        kernel_5d = self._make_diagonal_kernel(kernel_size)

        # Dense ground truth (TF32 disabled for full FP32 reference)
        dense_input = torch.zeros((1, self.CIN) + vol, device=self.DEVICE, dtype=self.DTYPE)
        dense_input[0, :, coord[0], coord[1], coord[2]] = 1.0

        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel_5d, padding="same")

        # Sparse IGEMM convolution
        dst_grid = grid.conv_grid(kernel_size=kernel_size, stride=1)
        dst_ijks = dst_grid.ijk.jdata

        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=IGEMM_CONFIG,
        )
        self.assertIsInstance(plan._backend, _PredGatherIGemmBackend)

        sparse_output = plan.execute(features, kernel_5d)

        dense_at_dst = dense_output[0, :, dst_ijks[:, 0], dst_ijks[:, 1], dst_ijks[:, 2]].T
        torch.testing.assert_close(sparse_output.jdata, dense_at_dst, rtol=TF32_RTOL, atol=TF32_ATOL)

    # =========================================================================
    # Multiple impulses forward (stride=1, vs dense conv3d)
    # =========================================================================

    @parameterized.expand(KERNEL_ONLY_COMBOS)
    def test_multiple_impulses_forward(self, ks: int):
        """
        Forward pass of multiple isolated impulses against dense conv3d ground truth.

        Uses hermit impulses (non-overlapping under the given kernel) so each
        impulse contributes independently to the output.
        """
        kernel_size = (ks, ks, ks)

        impulse_coords, impulse_field = generate_hermit_impulses_dense(
            num_candidates=self.NUM_CANDIDATES,
            volume_shape=self.VOLUME_SHAPE,
            kernel_size=kernel_size,
            impulse_value=1,
            dtype=self.DTYPE,
            device=self.DEVICE,
        )
        num_impulses = len(impulse_coords)
        self.assertGreater(num_impulses, 0)

        kernel_5d = self._make_diagonal_kernel(kernel_size)

        # Dense ground truth: tile single-channel impulse field to CIN channels
        dense_input = impulse_field.reshape(1, 1, *self.VOLUME_SHAPE).expand(1, self.CIN, -1, -1, -1).contiguous()
        with disable_tf32():
            dense_output = torch.nn.functional.conv3d(input=dense_input, weight=kernel_5d, padding="same")

        # Sparse IGEMM
        grid = create_grid_from_coords(impulse_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=kernel_size, stride=1)
        dst_ijks = dst_grid.ijk.jdata

        features = JaggedTensor(torch.ones((num_impulses, self.CIN), device=self.DEVICE, dtype=self.DTYPE))
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=1,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=IGEMM_CONFIG,
        )
        self.assertIsInstance(plan._backend, _PredGatherIGemmBackend)

        sparse_output = plan.execute(features, kernel_5d)

        dst_sorted_coords, dst_perm = sort_coords_by_ijk(dst_ijks)
        dense_values = dense_output[
            0,
            :,
            dst_sorted_coords[:, 0],
            dst_sorted_coords[:, 1],
            dst_sorted_coords[:, 2],
        ].T
        sparse_values = sparse_output.jdata[dst_perm]

        torch.testing.assert_close(sparse_values, dense_values, rtol=TF32_RTOL, atol=TF32_ATOL)

    # =========================================================================
    # Strided forward (stride=2, vs dense ground truth)
    # =========================================================================

    @parameterized.expand(KERNEL_ONLY_COMBOS)
    def test_strided_forward(self, ks: int):
        """
        Forward pass with stride=2 against dense ground truth.

        Uses conv_ground_truth_strided to build the FP32 reference and
        compares against the IGEMM output (TF32 tolerances).
        """
        kernel_size = (ks, ks, ks)
        stride = (2, 2, 2)

        cluster_coords = get_cluster_edge_aligned(kernel_size, self.DEVICE)
        grid = create_grid_from_coords(cluster_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=kernel_size, stride=stride)
        dst_coords = dst_grid.ijk.jdata
        num_voxels = len(cluster_coords)

        features_data = torch.randn((num_voxels, self.CIN), device=self.DEVICE, dtype=self.DTYPE)
        features = JaggedTensor(features_data)

        kernel_3d = fourier_anti_symmetric_kernel(kernel_size, dtype=self.DTYPE, device=self.DEVICE)
        kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0).expand(self.COUT, self.CIN, -1, -1, -1).clone()

        # Sparse IGEMM
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=stride,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=IGEMM_CONFIG,
        )
        self.assertIsInstance(plan._backend, _PredGatherIGemmBackend)

        sparse_output = plan.execute(features, kernel_5d)

        # Dense ground truth
        _, dense_output_at_dst = conv_ground_truth_strided(
            src_grid=grid,
            dst_grid=dst_grid,
            activation=JaggedTensor(features_data.clone()),
            weights=kernel_5d.clone(),
            stride=stride,
            allow_tf32=False,
        )

        _, perm = sort_coords_by_ijk(dst_coords)
        try:
            torch.testing.assert_close(
                sparse_output.jdata[perm],
                dense_output_at_dst[perm],
                rtol=TF32_RTOL,
                atol=TF32_ATOL,
            )
        except AssertionError:
            diag = diagnose_tensor_mismatch(
                f"Strided forward (ks={ks}, stride=2)",
                sparse_output.jdata[perm],
                dense_output_at_dst[perm],
                rtol=TF32_RTOL,
                atol=TF32_ATOL,
            )
            raise AssertionError(diag) from None

    # =========================================================================
    # Cross-backend: PredGatherIGemm vs GatherScatterDefault
    # =========================================================================

    @parameterized.expand(KERNEL_STRIDE_COMBOS)
    def test_matches_gather_scatter_default(self, ks: int, st: int):
        """
        PredGatherIGemm output matches GatherScatterDefault for all
        supported kernel/stride combinations.  This mirrors the C++ gtest
        cross-backend validation.
        """
        kernel_size = (ks, ks, ks)
        stride = (st, st, st)
        cin, cout = 64, 64

        cluster_coords = get_cluster_edge_aligned(kernel_size, self.DEVICE)
        grid = create_grid_from_coords(cluster_coords, self.DEVICE)
        dst_grid = grid.conv_grid(kernel_size=kernel_size, stride=stride)
        num_voxels = len(cluster_coords)

        torch.manual_seed(42)
        features = JaggedTensor(torch.randn((num_voxels, cin), device=self.DEVICE, dtype=self.DTYPE))
        weights = torch.randn((cout, cin, ks, ks, ks), device=self.DEVICE, dtype=self.DTYPE)

        # GatherScatterDefault (reference)
        gs_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=stride,
            source_grid=grid,
            target_grid=dst_grid,
        )
        gs_output = gs_plan.execute(features, weights)

        # PredGatherIGemm
        igemm_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=stride,
            source_grid=grid,
            target_grid=dst_grid,
            expert_config=IGEMM_CONFIG,
        )
        self.assertIsInstance(igemm_plan._backend, _PredGatherIGemmBackend)

        igemm_output = igemm_plan.execute(features, weights)

        self.assertEqual(igemm_output.jdata.shape, gs_output.jdata.shape)

        igemm_f64 = igemm_output.jdata.cpu().to(torch.float64)
        gs_f64 = gs_output.jdata.cpu().to(torch.float64)
        diff = (igemm_f64 - gs_f64).abs()

        self.assertTrue(
            torch.allclose(igemm_f64, gs_f64, rtol=TF32_RTOL, atol=TF32_ATOL),
            f"kernel={ks} stride={st}: max diff={diff.max().item():.6f}, mean diff={diff.mean().item():.6f}",
        )
