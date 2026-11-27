# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the masked implicit GEMM (masked_igemm) convolution backend.

The backend runs regular and transposed convolution with any stride and kernel size on CUDA
SM80+, with float16, bfloat16 and float32 features (float32 as TF32, or as 3xTF32 when TF32 is
off). Kernels of more than 32 taps use multi-word tap masks and compact neighbor maps. Channel
counts that are not multiples of 8 run zero-padded. Forward, input gradient and weight gradient
are checked against the default gather-scatter backend in float32, or float64 for 3xTF32.
"""

import contextlib
import math
import os
import unittest
import warnings

import torch
from parameterized import parameterized

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor, _fvdb_cpp, convolution_plan
from fvdb.convolution_plan import _MaskedIGemmBackend, _MatmulBackend
from fvdb.functional._dispatch import _get_grid_data

CONFIG = {"backend": "masked_igemm"}


def _has_sm80() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _random_grid(num_points: int, extent: int, batch: int = 1, seed: int = 0) -> GridBatch:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ijks = [
        torch.randint(-extent, extent, (num_points, 3), generator=gen, dtype=torch.int32).cuda() for _ in range(batch)
    ]
    return GridBatch.from_ijk(JaggedTensor(ijks))


def _shell_grid(res: int) -> GridBatch:
    ijk = torch.stack(torch.meshgrid(*[torch.arange(res)] * 3, indexing="ij"), dim=-1).reshape(-1, 3)
    dist = ((ijk.float() - res / 2 + 0.5) ** 2).sum(-1).sqrt()
    ijk = ijk[(dist <= res / 2) & (dist >= res / 2 - 1.25)]
    return GridBatch.from_ijk(JaggedTensor([ijk.int().cuda()]))


@contextlib.contextmanager
def _overrides(**kwargs):
    """Force masked_igemm choices that otherwise depend on grid size and the GPU."""
    _fvdb_cpp._masked_igemm_set_test_overrides(**kwargs)
    try:
        yield
    finally:
        _fvdb_cpp._masked_igemm_set_test_overrides()


@contextlib.contextmanager
def _tf32(enabled: bool):
    # Prefer fp32_precision (PyTorch 2.9+); reading allow_tf32 raises once it was used.
    matmul = torch.backends.cuda.matmul
    name = "fp32_precision" if hasattr(matmul, "fp32_precision") else "allow_tf32"
    previous = getattr(matmul, name)
    setattr(matmul, name, ("tf32" if enabled else "ieee") if name == "fp32_precision" else enabled)
    try:
        yield
    finally:
        setattr(matmul, name, previous)


def _build_topology(feature_grid: GridBatch, output_grid: GridBatch, kernel_size, stride=1):
    as_vec = lambda v: torch.tensor(v if isinstance(v, (tuple, list)) else (v,) * 3, dtype=torch.int32)
    return _fvdb_cpp.masked_igemm_build_topology(
        _get_grid_data(feature_grid), _get_grid_data(output_grid), as_vec(kernel_size), as_vec(stride)
    )


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


@unittest.skipUnless(_has_sm80(), "masked_igemm requires a CUDA SM80+ device")
class TestMaskedIGemm(unittest.TestCase):
    def _check(self, grid, cin, cout, kernel_size, dtype, tol=None):
        torch.manual_seed(0)
        tol = tol or (2e-2 if dtype == torch.bfloat16 else 4e-3)
        plan = ConvolutionPlan.from_grid_batch(kernel_size, 1, grid, grid, expert_config=CONFIG)
        self.assertIsInstance(plan._backend, _MaskedIGemmBackend)
        ref_plan = ConvolutionPlan.from_grid_batch(kernel_size, 1, grid, grid)

        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        x = torch.randn(grid.total_voxels, cin, device="cuda")
        w = torch.randn(cout, cin, *ks, device="cuda") / (cin * ks[0] * ks[1] * ks[2]) ** 0.5
        gy = torch.randn(grid.total_voxels, cout, device="cuda")

        xl = x.to(dtype).requires_grad_(True)
        wl = w.to(dtype).requires_grad_(True)
        y = plan.execute(grid.jagged_like(xl), wl).jdata
        y.backward(gy.to(dtype))

        # The reference runs in exact float32.
        xr = xl.detach().float().requires_grad_(True)
        wr = wl.detach().float().requires_grad_(True)
        with _tf32(False):
            yr = ref_plan.execute(grid.jagged_like(xr), wr).jdata
            yr.backward(gy.to(dtype).float())

        self.assertEqual(y.dtype, dtype)
        self.assertNotIn("gather_scatter", plan._backend.fallback, "covered inputs must run natively")
        self.assertLess(_rel_err(y, yr), tol, "forward")
        self.assertLess(_rel_err(xl.grad, xr.grad), tol, "input gradient")
        self.assertLess(_rel_err(wl.grad, wr.grad), tol, "weight gradient")

    @parameterized.expand(
        [
            (64, 64, torch.float16),
            (64, 128, torch.float16),
            (128, 64, torch.float16),
            (256, 128, torch.bfloat16),
            (192, 320, torch.float16),
            (32, 32, torch.float16),
            (32, 64, torch.float16),
            (64, 32, torch.bfloat16),
            (32, 96, torch.float16),
            (96, 160, torch.float16),
            (96, 96, torch.float16),
            (96, 64, torch.bfloat16),
        ]
    )
    def test_random_grid(self, cin, cout, dtype):
        self._check(_random_grid(3000, 12), cin, cout, 3, dtype)

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_shell(self, dtype):
        self._check(_shell_grid(48), 128, 128, 3, dtype)

    @parameterized.expand([(torch.float16,), (torch.float32,)])
    def test_shell_narrow(self, dtype):
        with _tf32(True):
            self._check(_shell_grid(48), 32, 32, 3, dtype, tol=5e-3)

    @parameterized.expand(
        [
            (8, 8, torch.float16),
            (16, 24, torch.float16),
            (40, 48, torch.float16),
            (48, 64, torch.bfloat16),
            (24, 136, torch.float16),
            (1, 32, torch.float16),
            (3, 16, torch.float16),
            (4, 64, torch.bfloat16),
            (17, 33, torch.float16),
            (100, 100, torch.float16),
            (8, 3, torch.float16),
            (40, 24, torch.float32),
            (3, 32, torch.float32),
        ]
    )
    def test_channels_not_multiple_of_32(self, cin, cout, dtype):
        # Multiples of 8 run on partial tiles; other counts run zero-padded to a multiple of 8.
        with _tf32(True):
            self._check(_random_grid(3000, 12), cin, cout, 3, dtype, tol=5e-3 if dtype == torch.float32 else None)

    @parameterized.expand([(8, 8), (16, 40), (3, 20), (40, 24)])
    def test_channels_not_multiple_of_32_unsplit(self, cin, cout):
        # Partial n-tiles must not store past C_out. Split-K adds the zero columns past C_out,
        # which hides an unguarded store, so force direct stores, with a bias.
        torch.manual_seed(0)
        grid = _random_grid(3000, 12)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        x = torch.randn(grid.total_voxels, cin, device="cuda", dtype=torch.float16)
        w = (torch.randn(cout, cin, 3, 3, 3, device="cuda") / (27 * cin) ** 0.5).half()
        b = torch.randn(cout, device="cuda", dtype=torch.float16)
        with _overrides(splits=1):
            y = plan.execute(x, w, b)
        with _tf32(False):
            yr = ref_plan.execute(x.float(), w.float(), b.float())
        self.assertNotIn("gather_scatter", plan._backend.fallback)
        self.assertLess(_rel_err(y, yr), 4e-3)

    @parameterized.expand(
        [
            ("gather_scatter", "default", False),
            ("gather_scatter_transposed", "default", True),
            ("masked_igemm", "masked_igemm", False),
            ("masked_igemm_transposed", "masked_igemm", True),
        ]
    )
    def test_plain_weights_get_contiguous_gradient(self, _, backend, transposed):
        # Module weights get a gradient in their (k2, k1, k0, C_in, C_out) storage; plain
        # (C_out, C_in, k0, k1, k2) weights get a contiguous gradient, so it can be flattened.
        grid = _random_grid(1500, 10)
        build = ConvolutionPlan.from_grid_batch_transposed if transposed else ConvolutionPlan.from_grid_batch
        plan = build((3, 3, 2), 2, grid, expert_config={"backend": backend})
        x = torch.randn(plan.source_grid_batch.total_voxels, 16, device="cuda", requires_grad=True)
        w = torch.randn(32, 16, 3, 3, 2, device="cuda", requires_grad=True)
        y = plan.execute(x, w)
        y.backward(torch.ones_like(y))
        self.assertTrue(w.grad.is_contiguous())
        self.assertEqual(w.grad.view(-1).numel(), w.numel())

    @parameterized.expand(
        [
            # Forced splits, so every GPU runs the per-split path: forward and input-gradient
            # split-K, and the weight gradient split by output rows.
            ("forced_split_k", 48, 256, 1, {"splits": 4}),
            ("forced_split_k_strided", 48, 256, 2, {"splits": 4}),
            ("forced_weight_row_splits", 96, 64, 1, {"weight_row_splits": 4}),
            ("forced_weight_row_splits_strided", 128, 64, 2, {"weight_row_splits": 4}),
            # The heuristics' own choices. On an RTX PRO 6000 these split the forward and input
            # gradient (C = 256) and the weight gradient by pairs (C = 64), and their results
            # differ run to run without the flag; other GPUs may choose differently.
            ("heuristic_c256", 48, 256, 1, {}),
            ("heuristic_c64", 96, 64, 1, {}),
            # 16-bit: the bias gradient must avoid cuBLAS, which older PyTorch versions reject in
            # deterministic mode without CUBLAS_WORKSPACE_CONFIG.
            ("float16", 96, 64, 1, {"weight_row_splits": 4}, torch.float16),
        ]
    )
    def test_deterministic_algorithms(self, _, res, channels, stride, overrides, dtype=torch.float32):
        # Split kernels add float partials with atomics in run-dependent order.
        # torch.use_deterministic_algorithms(True) gives each split its own slice, summed in a
        # fixed order, so every result must repeat bitwise.
        torch.manual_seed(0)
        grid = _shell_grid(res)
        target = grid if stride == 1 else grid.conv_grid(3, stride)
        plan = ConvolutionPlan.from_grid_batch(3, stride, grid, target, expert_config=CONFIG)
        x = torch.randn(grid.total_voxels, channels, device="cuda", dtype=dtype)
        w = (torch.randn(channels, channels, 3, 3, 3, device="cuda") / (channels * 27) ** 0.5).to(dtype)
        b = torch.randn(channels, device="cuda", dtype=dtype)
        gy = torch.randn(target.total_voxels, channels, device="cuda", dtype=dtype)

        def run():
            xx, ww, bb = (t.clone().requires_grad_(True) for t in (x, w, b))
            y = plan.execute(xx, ww, bb)
            y.backward(gy)
            return y.detach(), xx.grad, ww.grad, bb.grad

        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            with _overrides(**overrides):
                runs = [run() for _ in range(5)]
        finally:
            torch.use_deterministic_algorithms(previous)
        self.assertNotIn("gather_scatter", plan._backend.fallback)
        for name, i in (("forward", 0), ("input gradient", 1), ("weight gradient", 2), ("bias gradient", 3)):
            for r in runs[1:]:
                self.assertTrue(torch.equal(r[i], runs[0][i]), name)

    def test_misaligned_views_run(self):
        # cp.async needs 16-byte-aligned operands; a contiguous view that starts mid-buffer is
        # copied into fresh storage instead of faulting.
        grid = _random_grid(2000, 10)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        n = grid.total_voxels
        buf = torch.randn(n * 16 + 1, device="cuda", dtype=torch.float16)
        x = buf[1:].view(n, 16)
        self.assertNotEqual(x.data_ptr() % 16, 0)
        wbuf = torch.randn(32 * 16 * 27 + 1, device="cuda", dtype=torch.float16) * 0.05
        w = wbuf[1:].view(32, 16, 3, 3, 3)
        xg, wg = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        with _overrides(splits=1):  # split-K accumulates with atomics, so is not bitwise repeatable
            y = plan.execute(x, w)
            torch.testing.assert_close(y, plan.execute(x.clone(), w.clone()), rtol=0, atol=0)
        yg = plan.execute(xg, wg)
        yg.backward(torch.ones_like(yg))
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(xg.grad).all() and torch.isfinite(wg.grad).all())

    @parameterized.expand([("masked_igemm", "masked_igemm"), ("gather_scatter", "default")])
    def test_rejects_channel_mismatch(self, _, backend):
        grid = _random_grid(500, 8)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config={"backend": backend})
        w = torch.randn(32, 16, 3, 3, 3, device="cuda", dtype=torch.float16)
        for channels in (12, 20):
            x = torch.randn(grid.total_voxels, channels, device="cuda", dtype=torch.float16)
            with self.assertRaisesRegex(ValueError, "the weights expect 16"):
                plan.execute(x, w)

    def test_test_overrides_reject_zero_counts(self):
        with self.assertRaisesRegex(RuntimeError, "splits must be at least 1"):
            _fvdb_cpp._masked_igemm_set_test_overrides(splits=0)
        with self.assertRaisesRegex(RuntimeError, "weight_row_splits must be at least 1"):
            _fvdb_cpp._masked_igemm_set_test_overrides(weight_row_splits=0)
        _fvdb_cpp._masked_igemm_set_test_overrides()

    def test_rejects_kernel_beyond_shared_memory(self):
        grid = _random_grid(200, 6)
        with self.assertRaisesRegex(RuntimeError, "bytes of shared memory per block"):
            _build_topology(grid, grid, (101, 101, 1))

    def test_anisotropic_kernel(self):
        self._check(_random_grid(2000, 10), 64, 64, (3, 5, 1), torch.float16)

    def test_batch(self):
        self._check(_random_grid(1500, 10, batch=3), 64, 128, 3, torch.float16)

    def test_sparse_rows(self):
        # Mostly isolated voxels: many rows and tiles use few or no taps.
        self._check(_random_grid(500, 200), 64, 64, 3, torch.float16)

    def test_single_voxel(self):
        grid = GridBatch.from_ijk(JaggedTensor([torch.zeros(1, 3, dtype=torch.int32, device="cuda")]))
        self._check(grid, 64, 64, 3, torch.float16)

    @parameterized.expand([("global", False), ("local", True)])
    def test_row_order(self, _, local):
        # The local order regroups rows in windows of 65536, so the grid spans several windows.
        grid = _random_grid(200000, 40)
        self.assertGreater(grid.total_voxels, 2 * 65536)
        with _overrides(local_row_order=local):
            self._check(grid, 64, 64, 3, torch.float16)

    @parameterized.expand([(1,), (5,), (1000,)])
    def test_weight_gradient_row_split(self, splits):
        # Row splitting is chosen only when features are several times L2; force it on small grids.
        with _overrides(weight_row_splits=splits):
            self._check(_shell_grid(48), 64, 128, 3, torch.float16)
            self._check(_random_grid(3000, 12, batch=2), 128, 64, 3, torch.float16)

    def test_neighbor_map(self):
        grid = _random_grid(800, 6)
        topo = _build_topology(grid, grid, 3)
        self.assertIsNone(topo.reverse_map, "a submanifold map is its own reverse")
        ijk = grid.ijk.jdata.cpu()
        lookup = {tuple(v.tolist()): i for i, v in enumerate(ijk)}
        nbr = topo.neighbor_map.cpu()
        taps = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        for o in range(0, ijk.shape[0], 7):
            for k, (a, b, c) in enumerate(taps):
                expect = lookup.get((int(ijk[o, 0]) + a, int(ijk[o, 1]) + b, int(ijk[o, 2]) + c), -1)
                self.assertEqual(int(nbr[o, k]), expect)
        self.assertTrue(
            torch.equal(topo.sorted_rows.sort().values.cpu(), torch.arange(ijk.shape[0], dtype=torch.int32))
        )

    @parameterized.expand(
        [
            ("sparse", (3, 3, 3), 1),
            ("dense", (3, 3, 3), 1),
            ("shell", (3, 3, 3), 1),
            ("batch", (3, 3, 3), 1),
            ("view", (3, 3, 3), 1),
            ("sparse", (3, 5, 1), 1),
            ("dense", (17, 1, 1), 1),
            ("sparse", (19, 1, 1), 1),
            ("sparse", (3, 3, 3), 2),
            ("batch", (2, 2, 2), 2),
            ("shell", (3, 3, 3), 2),
            ("sparse", (5, 5, 5), 1),
            ("batch", (7, 7, 7), 1),
            ("sparse", (5, 5, 5), 2),
        ]
    )
    def test_vbm_iteration_matches_leaf_scan(self, kind, kernel_size, stride):
        # The neighbor-map builders walk the row grid through the VoxelBlockManager or the leaf
        # scan; each row is written by one thread either way, so the topologies match exactly.
        if kind == "sparse":
            grid = _random_grid(4000, 40)
        elif kind == "dense":
            grid = GridBatch.from_dense(1, [20, 19, 18], device="cuda")
        elif kind == "shell":
            grid = _shell_grid(64)
        elif kind == "batch":
            grid = _random_grid(3000, 16, batch=3)
        else:
            grid = _random_grid(2000, 12, batch=4)[1:3]
        output = grid if stride == 1 else grid.conv_grid(kernel_size, stride)
        with _overrides(vbm_iteration=False):
            ref = _build_topology(grid, output, kernel_size, stride)
        with _overrides(vbm_iteration=True):
            topo = _build_topology(grid, output, kernel_size, stride)
        names = ["neighbor_map", "row_masks", "sorted_rows", "pair_in", "pair_out", "pair_offsets"]
        if math.prod(kernel_size) > 32:
            names.append("neighbor_offsets")
        for name in names:
            self.assertTrue(torch.equal(getattr(topo, name), getattr(ref, name)), name)
        self.assertEqual(topo.reverse_map is None, ref.reverse_map is None)
        if ref.reverse_map is not None:
            self.assertTrue(torch.equal(topo.reverse_map, ref.reverse_map), "reverse_map")

    # ---- Kernels of more than 32 taps (multi-word tap masks, chunked neighbor staging) ----

    @parameterized.expand(
        [
            ("k5", 5, 1, 64, 64, torch.float16),
            ("k7", 7, 1, 32, 64, torch.float16),
            ("k33_boundary", (11, 3, 1), 1, 64, 32, torch.float16),
            ("k443_even", (4, 4, 3), 1, 64, 64, torch.float16),
            ("k5_bf16", 5, 1, 64, 128, torch.bfloat16),
            ("k5_tf32", 5, 1, 64, 96, torch.float32),
            ("k5_s2", 5, 2, 64, 64, torch.float16),
            ("k7_s2", 7, 2, 32, 32, torch.float16),
            ("k5_narrow", 5, 1, 3, 20, torch.float16),
        ]
    )
    def test_large_kernel(self, _, kernel_size, stride, cin, cout, dtype):
        grid = _random_grid(2500, 10, batch=2)
        plan = ConvolutionPlan.from_grid_batch(kernel_size, stride, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(kernel_size, stride, grid)
        tol = {torch.float16: 4e-3, torch.bfloat16: 2e-2, torch.float32: 5e-3}[dtype]
        self._compare_plans(plan, ref_plan, cin, cout, dtype, tol)

    @parameterized.expand([("factory", True), ("from_plan", False)])
    def test_large_kernel_transpose(self, _, factory):
        fine = _random_grid(2500, 10, batch=2)
        coarse = fine.conv_grid(5, 2)
        if factory:
            plan = ConvolutionPlan.from_grid_batch_transposed(5, 2, coarse, fine, expert_config=CONFIG)
            ref_plan = ConvolutionPlan.from_grid_batch_transposed(5, 2, coarse, fine)
        else:
            plan = ConvolutionPlan.from_plan_transposed(
                ConvolutionPlan.from_grid_batch(5, 2, fine, coarse, expert_config=CONFIG)
            )
            ref_plan = ConvolutionPlan.from_plan_transposed(ConvolutionPlan.from_grid_batch(5, 2, fine, coarse))
        self._compare_plans(plan, ref_plan, 64, 32, torch.float16, 4e-3)

    @parameterized.expand([("split2", 2), ("split5", 5)])
    def test_large_kernel_split_k(self, _, splits):
        # Split-K starts a CTA's tap walk partway through a tap group.
        grid = _random_grid(1500, 8)
        with _overrides(splits=splits):
            plan = ConvolutionPlan.from_grid_batch(5, 1, grid, expert_config=CONFIG)
            ref_plan = ConvolutionPlan.from_grid_batch(5, 1, grid)
            self._compare_plans(plan, ref_plan, 96, 64, torch.float16, 4e-3)

    @staticmethod
    def _dense_from_compact(nbr, offsets, masks, k):
        """Expand a compact map (present neighbors in tap order per row) to dense [N, K]."""
        n, words = masks.shape
        bits = ((masks.long().unsqueeze(-1) >> torch.arange(32)) & 1).bool().view(n, words * 32)[:, :k]
        dense = torch.full((n, k), -1, dtype=torch.int32)
        dense[bits] = nbr
        assert torch.equal(bits.sum(1), (offsets[1:] - offsets[:-1]).long())
        return dense

    @parameterized.expand([("k5", 5), ("k7", 7), ("k33", (11, 3, 1))])
    def test_large_kernel_compact_map(self, _, kernel_size):
        grid = _random_grid(1500, 8)
        topo = _build_topology(grid, grid, kernel_size)
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        k = math.prod(ks)
        words = (k + 31) // 32
        masks = topo.row_masks.cpu()
        offsets = topo.neighbor_offsets.cpu()
        self.assertEqual(tuple(masks.shape), (grid.total_voxels, words))
        self.assertEqual(int(offsets[-1]), topo.neighbor_map.numel())
        dense = self._dense_from_compact(topo.neighbor_map.cpu(), offsets, masks, k)

        # Every row's neighbors match a direct coordinate lookup, taps in k0-slowest order.
        ijk = grid.ijk.jdata.cpu()
        lookup = {tuple(v.tolist()): i for i, v in enumerate(ijk)}
        pad = [(s - 1) // 2 for s in ks]
        taps = [(a - pad[0], b - pad[1], c - pad[2]) for a in range(ks[0]) for b in range(ks[1]) for c in range(ks[2])]
        for o in range(0, ijk.shape[0], 23):
            expect = [lookup.get((int(ijk[o, 0]) + a, int(ijk[o, 1]) + b, int(ijk[o, 2]) + c), -1) for a, b, c in taps]
            self.assertEqual(dense[o].tolist(), expect)

        # Rows run in non-decreasing Gray rank of the whole mask, tap K-1 most significant.
        def rank(row: int) -> int:
            m = sum((int(masks[row, w]) & 0xFFFFFFFF) << (32 * w) for w in range(words))
            r = 0
            while m:
                r ^= m
                m >>= 1
            return r

        ranks = [rank(int(o)) for o in topo.sorted_rows.cpu()]
        self.assertEqual(ranks, sorted(ranks))

    def test_large_kernel_compact_reverse_map(self):
        # A strided map stores its reverse compactly: input row i lists, in tap order, every output
        # row o that reads it, so (i, k, o) is a reverse entry exactly when (o, k, i) is a forward one.
        fine = _random_grid(2000, 9)
        coarse = fine.conv_grid(5, 2)
        topo = _build_topology(fine, coarse, 5, 2)
        fwd = self._dense_from_compact(topo.neighbor_map.cpu(), topo.neighbor_offsets.cpu(), topo.row_masks.cpu(), 125)
        o, k = (fwd >= 0).nonzero(as_tuple=True)
        by_input: dict[int, list[tuple[int, int]]] = {}
        for i, kk, oo in zip(fwd[o, k].tolist(), k.tolist(), o.tolist()):
            by_input.setdefault(i, []).append((kk, oo))
        rev = topo.reverse_map.cpu()
        offsets = topo.reverse_offsets.cpu()
        self.assertEqual(rev.numel(), o.numel())
        for i in range(fine.total_voxels):
            expect = [oo for _, oo in sorted(by_input.get(i, []))]
            self.assertEqual(rev[offsets[i] : offsets[i + 1]].tolist(), expect)

    @parameterized.expand(
        [((64, 128),), ((128, 64),), ((192, 256),), ((32, 32),), ((96, 64),), ((64, 96),), ((96, 96),)]
    )
    def test_tf32(self, channels):
        with _tf32(True):
            self._check(_random_grid(3000, 12), *channels, 3, torch.float32, tol=5e-3)

    def test_tf32_shell(self):
        with _tf32(True):
            self._check(_shell_grid(48), 128, 128, 3, torch.float32, tol=5e-3)

    @parameterized.expand(
        [
            ("float64", 64, 64, torch.float64),
        ]
    )
    def test_uncovered_inputs_fall_back(self, _, cin, cout, dtype):
        # Inputs the kernels do not cover run on gather-scatter and match it exactly.
        grid = _random_grid(1500, 10)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        x = torch.randn(grid.total_voxels, cin, device="cuda", dtype=dtype)
        w = torch.randn(cout, cin, 3, 3, 3, device="cuda", dtype=dtype)
        with _tf32(False):
            y = plan.execute(grid.jagged_like(x), w).jdata
            yr = ref_plan.execute(grid.jagged_like(x), w).jdata
        torch.testing.assert_close(y, yr)
        self.assertIn("gather_scatter", plan._backend.fallback)

    def test_fallback_warns_once(self):
        grid = _random_grid(1500, 10)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        x = grid.jagged_like(torch.randn(grid.total_voxels, 16, device="cuda", dtype=torch.float64))
        w = torch.randn(16, 16, 3, 3, 3, device="cuda", dtype=torch.float64)
        convolution_plan._WARNED_MASKED_IGEMM_FALLBACKS.clear()
        with self.assertWarnsRegex(UserWarning, r"torch\.float64 inputs; running gather-scatter") as cm:
            plan.execute(x, w)
        self.assertEqual(os.path.basename(cm.filename), os.path.basename(__file__))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan.execute(x, w)
            plan.execute(grid.jagged_like(x.jdata.float()), w.float())  # covered: runs natively, no warning
        self.assertEqual([str(c.message) for c in caught], [])

    @parameterized.expand([(True,), (False,)])
    def test_module_bias_and_grad_flags(self, input_requires_grad):
        torch.manual_seed(0)
        grid = _random_grid(2000, 10)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        conv = fvdb.nn.SparseConv3d(64, 128, 3, bias=True).cuda()
        torch.nn.init.normal_(conv.bias)
        ref = fvdb.nn.SparseConv3d(64, 128, 3, bias=True).cuda()
        ref.load_state_dict(conv.state_dict())
        conv = conv.half()

        x = torch.randn(grid.total_voxels, 64, device="cuda")
        xh = x.half().requires_grad_(input_requires_grad)
        xr = xh.detach().float().requires_grad_(input_requires_grad)
        y = conv(grid.jagged_like(xh), plan).jdata
        yr = ref(grid.jagged_like(xr), ref_plan).jdata
        gy = torch.randn_like(yr)
        y.backward(gy.half())
        yr.backward(gy.half().float())

        self.assertLess(_rel_err(y, yr), 4e-3)
        self.assertLess(_rel_err(conv.bias.grad, ref.bias.grad), 4e-3)
        self.assertLess(_rel_err(conv.weight.grad, ref.weight.grad), 4e-3)
        if input_requires_grad:
            self.assertLess(_rel_err(xh.grad, xr.grad), 4e-3)
        else:
            self.assertIsNone(xh.grad)

    @parameterized.expand(
        [
            ("submanifold", 3, 1, False, 64, 128),
            ("strided", 2, 2, False, 64, 64),
            ("transposed", 2, 2, True, 96, 32),
            ("narrow", 3, 1, False, 3, 40),
            ("wide", 5, 1, False, 64, 64),
            ("wide_transposed", 5, 2, True, 32, 64),
        ]
    )
    def test_strict_fp32(self, _, kernel_size, stride, transposed, cin, cout):
        # float32 with TF32 off runs natively as 3xTF32, as accurate as IEEE float32 and far below TF32.
        torch.manual_seed(0)
        fine = _random_grid(3000, 12)
        coarse = fine if stride == 1 else fine.conv_grid(kernel_size, stride)
        if transposed:
            make = lambda cfg: ConvolutionPlan.from_grid_batch_transposed(
                kernel_size, stride, coarse, fine, expert_config=cfg
            )
        else:
            make = lambda cfg: ConvolutionPlan.from_grid_batch(kernel_size, stride, fine, coarse, expert_config=cfg)
        plan, ref_plan = make(CONFIG), make({})
        source = plan.source_grid_batch
        ks = (kernel_size,) * 3
        x = torch.randn(source.total_voxels, cin, device="cuda", dtype=torch.float64)
        w = torch.randn(cout, cin, *ks, device="cuda", dtype=torch.float64) / (cin * math.prod(ks)) ** 0.5

        def run(plan, dtype):
            xx = x.detach().clone().to(dtype).requires_grad_(True)
            ww = w.detach().clone().to(dtype).requires_grad_(True)
            y = plan.execute(source.jagged_like(xx), ww).jdata
            y.backward(torch.ones_like(y))
            return y.detach(), xx.grad, ww.grad

        with _tf32(False):
            ref = run(ref_plan, torch.float64)
            strict = run(plan, torch.float32)
        self.assertNotIn("gather_scatter", plan._backend.fallback)
        with _tf32(True):
            tf32 = run(plan, torch.float32)
        for name, a, b, r in zip(("forward", "input gradient", "weight gradient"), strict, tf32, ref):
            err_strict, err_tf32 = _rel_err(a.double(), r), _rel_err(b.double(), r)
            self.assertLess(err_strict, 2e-6, name)
            self.assertLess(err_strict * 20, err_tf32, f"{name}: 3xTF32 should be far below TF32")

    @parameterized.expand([("narrow", 3), ("wide", 5)])
    def test_strict_fp32_non_finite(self, _, kernel_size):
        # 3xTF32 splits each operand into a TF32 big part and a residual. Rounding FLT_MAX to TF32
        # must not overflow to inf (values that close to it keep TF32 precision), and inf * 0 in a
        # cross term must not turn an infinite product into NaN.
        torch.manual_seed(0)
        grid = _random_grid(500, 6)
        plan = ConvolutionPlan.from_grid_batch(kernel_size, 1, grid, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(kernel_size, 1, grid, grid)
        cin = cout = 32
        # Positive weights small enough that FLT_MAX * w stays finite.
        w = torch.rand(cout, cin, *(kernel_size,) * 3, device="cuda") * 2.0**-100 + 2.0**-110

        def run(x):
            with _tf32(False):
                return plan.execute(grid.jagged_like(x), w).jdata, ref_plan.execute(grid.jagged_like(x), w).jdata

        x = torch.randn(grid.total_voxels, cin, device="cuda")
        x[0, 0] = torch.finfo(torch.float32).max  # rounding up to TF32 would overflow
        y, ref = run(x)
        self.assertTrue(torch.isfinite(ref).all())
        self.assertTrue(torch.isfinite(y).all())
        near_max = ref.abs() > 1.0  # outputs of the FLT_MAX feature; the rest are ~2**-95
        self.assertTrue(near_max.any())
        self.assertTrue(((y[near_max] - ref[near_max]).abs() <= 1e-3 * ref[near_max].abs()).all())
        self.assertLess(_rel_err(y[~near_max], ref[~near_max]), 1e-5)

        x = torch.randn(grid.total_voxels, cin, device="cuda")
        x[1, 3] = float("inf")
        x[2, 5] = -float("inf")
        y, ref = run(x)
        self.assertNotIn("gather_scatter", plan._backend.fallback)
        self.assertTrue(torch.isinf(ref).any())
        self.assertTrue(torch.equal(torch.isnan(y), torch.isnan(ref)), "NaN positions")
        self.assertTrue(torch.equal(torch.isinf(y), torch.isinf(ref)), "inf positions")
        infinite, finite = torch.isinf(ref), torch.isfinite(ref)
        self.assertTrue(torch.equal(y[infinite], ref[infinite]), "signs of infinities")
        self.assertLess(_rel_err(y[finite], ref[finite]), 1e-5)

    def test_tf32_legacy_api(self):
        # The other TF32 tests use fp32_precision; this one enables TF32 through allow_tf32.
        matmul = torch.backends.cuda.matmul
        previous = matmul.allow_tf32
        matmul.allow_tf32 = True
        try:
            grid = _random_grid(2000, 10)
            plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
            x = torch.randn(grid.total_voxels, 64, device="cuda")
            plan.execute(grid.jagged_like(x), torch.randn(64, 64, 3, 3, 3, device="cuda"))
            self.assertNotIn("gather_scatter", plan._backend.fallback)
        finally:
            # Restore through the same API; resetting fp32_precision would leave the flags mixed.
            matmul.allow_tf32 = previous

    @parameterized.expand(
        [
            ("half_features", torch.float16, torch.float32, True),
            ("half_weights", torch.float32, torch.float16, True),
            ("half_features_strict", torch.float16, torch.float32, False),
        ]
    )
    def test_mixed_dtypes(self, _, feature_dtype, weight_dtype, tf32):
        # Mixed inputs promote like gather-scatter and run natively, as 3xTF32 when TF32 is off.
        torch.manual_seed(0)
        grid = _random_grid(2000, 10)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid)
        x = torch.randn(grid.total_voxels, 64, device="cuda", dtype=feature_dtype, requires_grad=True)
        w = (torch.randn(64, 64, 3, 3, 3, device="cuda") / 40).to(weight_dtype).requires_grad_(True)
        xr = x.detach().float().requires_grad_(True)
        wr = w.detach().float().requires_grad_(True)
        with _tf32(tf32):
            y = plan.execute(grid.jagged_like(x), w).jdata
            gy = torch.randn_like(y)
            y.backward(gy)
        with _tf32(False):
            yr = ref_plan.execute(grid.jagged_like(xr), wr).jdata
            yr.backward(gy)

        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(x.grad.dtype, feature_dtype)
        self.assertEqual(w.grad.dtype, weight_dtype)
        self.assertNotIn("gather_scatter", plan._backend.fallback)
        self.assertLess(_rel_err(y, yr), 5e-3)
        self.assertLess(_rel_err(x.grad, xr.grad), 5e-3)
        self.assertLess(_rel_err(w.grad, wr.grad), 5e-3)

    def test_rejects_mismatched_inputs(self):
        grid = _random_grid(500, 8)
        topo = _build_topology(grid, grid, 3)
        x = torch.randn(grid.total_voxels, 64, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 64, 3, 3, 3, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "same device"):
            _fvdb_cpp.masked_igemm_conv(x, w.cpu(), None, topo)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            _fvdb_cpp.masked_igemm_conv(x, w.float(), None, topo)
        with self.assertRaisesRegex(RuntimeError, "bias"):
            _fvdb_cpp.masked_igemm_conv(x, w, torch.zeros(64, device="cuda"), topo)

    @parameterized.expand(
        [
            ("half", 300000, 128, torch.float16),
            ("bf16", 70000, 256, torch.bfloat16),
            ("float", 100000, 64, torch.float32),
            ("small", 100, 64, torch.float16),
            ("wide", 1000, 4096, torch.float16),
            ("empty", 0, 64, torch.float16),
        ]
    )
    def test_bias_gradient(self, _, rows, cols, dtype):
        x = torch.randn(rows, cols, device="cuda").to(dtype)
        with _tf32(True):  # float32 must not take the TF32 matmul path
            y = convolution_plan._bias_gradient(x)
        ref = x.double().sum(0)
        self.assertEqual(y.dtype, dtype)
        self.assertLess(
            ((y.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item(),
            1e-2 if dtype != torch.float32 else 1e-5,
        )

    def test_supports(self):
        cuda = torch.device("cuda", torch.cuda.current_device())
        self.assertTrue(_fvdb_cpp.masked_igemm_supports(64, 32, torch.float16, cuda))
        self.assertFalse(_fvdb_cpp.masked_igemm_supports(64, 32, torch.float16, torch.device("cpu")))
        self.assertFalse(_fvdb_cpp.masked_igemm_supports(0, 64, torch.float16, cuda))
        self.assertFalse(_fvdb_cpp.masked_igemm_supports(64, 0, torch.float16, cuda))
        self.assertTrue(_fvdb_cpp.masked_igemm_supports(48, 8, torch.float16, cuda))
        self.assertFalse(_fvdb_cpp.masked_igemm_supports(12, 64, torch.float16, cuda))
        self.assertFalse(_fvdb_cpp.masked_igemm_supports(64, 4, torch.float16, cuda))
        # A device index past the last GPU is unsupported, not an internal error.
        missing = torch.device("cuda", torch.cuda.device_count())
        self.assertFalse(_fvdb_cpp.masked_igemm_supports(64, 64, torch.float16, missing))

    def _compare_plans(self, plan, ref_plan, cin, cout, dtype, tol, *, native=True):
        """Forward, input, weight and bias gradients of ``plan`` against float32 gather-scatter."""
        torch.manual_seed(0)
        source = plan.source_grid_batch
        ks = tuple(int(k) for k in plan.kernel_size.tolist())
        x = torch.randn(source.total_voxels, cin, device="cuda").to(dtype).requires_grad_(True)
        w = (torch.randn(cout, cin, *ks, device="cuda") / (cin * math.prod(ks)) ** 0.5).to(dtype).requires_grad_(True)
        b = torch.randn(cout, device="cuda").to(dtype).requires_grad_(True)
        xr = x.detach().float().requires_grad_(True)
        wr = w.detach().float().requires_grad_(True)
        br = b.detach().float().requires_grad_(True)
        with _tf32(True):
            y = plan.execute(source.jagged_like(x), w, b).jdata
            gy = torch.randn_like(y)
            y.backward(gy)
        with _tf32(False):
            yr = ref_plan.execute(source.jagged_like(xr), wr, br).jdata
            yr.backward(gy.float())

        self.assertIsInstance(plan._backend, _MaskedIGemmBackend)
        self.assertEqual(y.shape, yr.shape)
        self.assertEqual("gather_scatter" not in plan._backend.fallback, native)
        self.assertLess(_rel_err(y, yr), tol, "forward")
        self.assertLess(_rel_err(x.grad, xr.grad), tol, "input gradient")
        self.assertLess(_rel_err(w.grad, wr.grad), tol, "weight gradient")
        self.assertLess(_rel_err(b.grad, br.grad), tol, "bias gradient")

    @parameterized.expand(
        [
            ("k2_s2", 2, 2, 64, 128, torch.float16),
            ("k3_s2", 3, 2, 128, 64, torch.float16),
            ("k2_s2_bf16", 2, 2, 64, 96, torch.bfloat16),
            ("k2_s2_tf32", 2, 2, 32, 64, torch.float32),
            ("k3_s122", 3, (1, 2, 2), 64, 64, torch.float16),
            ("k2_s1", 2, 1, 64, 64, torch.float16),
            ("k3_s1_dilated", 3, 1, 32, 64, torch.float16),
            ("narrow", 2, 2, 16, 40, torch.float16),
            ("padded", 2, 2, 3, 20, torch.float16),
            ("fallback", 2, 2, 64, 64, torch.float64),
        ]
    )
    def test_strided(self, _, kernel_size, stride, cin, cout, dtype):
        grid = _random_grid(3000, 12, batch=2)
        plan = ConvolutionPlan.from_grid_batch(kernel_size, stride, grid, expert_config=CONFIG)
        ref_plan = ConvolutionPlan.from_grid_batch(kernel_size, stride, grid)
        self.assertIsNotNone(plan._backend.topology.reverse_map)
        tol = {torch.float16: 4e-3, torch.bfloat16: 2e-2, torch.float32: 5e-3, torch.float64: 1e-5}[dtype]
        self._compare_plans(plan, ref_plan, cin, cout, dtype, tol, native=dtype != torch.float64)

    @parameterized.expand(
        [
            ("factory_k2_s2", 2, 2, 128, 64, torch.float16, True),
            ("factory_k3_s2", 3, 2, 64, 64, torch.float16, True),
            ("from_plan_k2_s2", 2, 2, 64, 128, torch.float16, False),
            ("from_plan_k3_s2_tf32", 3, 2, 64, 32, torch.float32, False),
            ("factory_padded", 2, 2, 20, 4, torch.float16, True),
            ("factory_fallback", 2, 2, 64, 64, torch.float64, True),
        ]
    )
    def test_strided_transpose(self, _, kernel_size, stride, cin, cout, dtype, factory):
        fine = _random_grid(3000, 12, batch=2)
        coarse = fine.conv_grid(kernel_size, stride)
        if factory:
            plan = ConvolutionPlan.from_grid_batch_transposed(kernel_size, stride, coarse, fine, expert_config=CONFIG)
            ref_plan = ConvolutionPlan.from_grid_batch_transposed(kernel_size, stride, coarse, fine)
        else:
            plan = ConvolutionPlan.from_plan_transposed(
                ConvolutionPlan.from_grid_batch(kernel_size, stride, fine, coarse, expert_config=CONFIG)
            )
            ref_plan = ConvolutionPlan.from_plan_transposed(
                ConvolutionPlan.from_grid_batch(kernel_size, stride, fine, coarse)
            )
        tol = {torch.float16: 4e-3, torch.float32: 5e-3, torch.float64: 1e-5}[dtype]
        self._compare_plans(plan, ref_plan, cin, cout, dtype, tol, native=dtype != torch.float64)

    def test_reverse_map(self):
        fine = _random_grid(2000, 10, batch=2)
        topo = _build_topology(fine, fine.conv_grid(3, 2), 3, 2)
        nbr, rev = topo.neighbor_map.long(), topo.reverse_map.long()
        self.assertEqual(tuple(rev.shape), (fine.total_voxels, 27))
        self.assertEqual(topo.feature_total_voxels, fine.total_voxels)
        self.assertEqual(int((nbr >= 0).sum()), int((rev >= 0).sum()))
        o, k = torch.nonzero(nbr >= 0, as_tuple=True)
        self.assertTrue(torch.equal(rev[nbr[o, k], k], o))

    @parameterized.expand([("forward", False), ("transposed", True)])
    def test_strided_coverage_report(self, _, transposed):
        fine = _random_grid(1500, 10)
        coarse = fine.conv_grid(2, 2)
        if transposed:
            report = ConvolutionPlan.from_grid_batch_transposed(
                2, 2, coarse, fine, expert_config=CONFIG
            ).coverage_report
            expected = ConvolutionPlan.from_grid_batch_transposed(2, 2, coarse, fine).coverage_report
        else:
            report = ConvolutionPlan.from_grid_batch(2, 2, fine, coarse, expert_config=CONFIG).coverage_report
            expected = ConvolutionPlan.from_grid_batch(2, 2, fine, coarse).coverage_report
        self.assertEqual(report, expected)

    @parameterized.expand([("forward", False), ("transposed", True)])
    def test_strided_strict_coverage(self, _, transposed):
        # Dropping coarse voxels leaves fine voxels with no coarse neighbor while every kept coarse
        # voxel still has fine inputs, so only the transposed direction has zero-degree outputs.
        fine = _random_grid(1500, 10)
        full = fine.conv_grid(2, 2)
        coarse = GridBatch.from_ijk(
            JaggedTensor([full.ijk.jdata[::2].contiguous()]), voxel_sizes=full.voxel_sizes, origins=full.origins
        )
        for config in (CONFIG, {}):
            if transposed:
                with self.assertRaisesRegex(ValueError, "zero-degree output rows"):
                    ConvolutionPlan.from_grid_batch_transposed(
                        2, 2, coarse, fine, expert_config=config, strict_output_coverage=True
                    )
            else:
                ConvolutionPlan.from_grid_batch(2, 2, fine, coarse, expert_config=config, strict_output_coverage=True)

    def test_module_and_transpose(self):
        grid = _random_grid(1000, 8)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        conv = fvdb.nn.SparseConv3d(64, 64, 3, bias=True).cuda().half()
        x = grid.jagged_like(torch.randn(grid.total_voxels, 64, device="cuda", dtype=torch.float16))
        y = conv(x, plan)
        ref = conv(x, ConvolutionPlan.from_grid_batch(3, 1, grid, grid))
        self.assertLess(_rel_err(y.jdata, ref.jdata), 4e-3)

    @parameterized.expand(
        [
            ("native", 64, 128, torch.float16, False),
            ("padded", 4, 20, torch.float16, False),
            ("fallback", 64, 64, torch.float64, False),
            ("factory_native", 64, 128, torch.float16, True),
            ("factory_tf32", 64, 96, torch.float32, True),
        ]
    )
    def test_transpose(self, _, cin, cout, dtype, factory):
        # Transposed masked plans, from either constructor, match gather-scatter's transpose.
        torch.manual_seed(0)
        grid = _random_grid(2000, 10)
        if factory:
            plan = ConvolutionPlan.from_grid_batch_transposed((3, 1, 5), 1, grid, grid, expert_config=CONFIG)
            ref_plan = ConvolutionPlan.from_grid_batch_transposed((3, 1, 5), 1, grid, grid)
        else:
            plan = ConvolutionPlan.from_plan_transposed(
                ConvolutionPlan.from_grid_batch((3, 1, 5), 1, grid, grid, expert_config=CONFIG)
            )
            ref_plan = ConvolutionPlan.from_plan_transposed(ConvolutionPlan.from_grid_batch((3, 1, 5), 1, grid, grid))
        self.assertIsInstance(plan._backend, _MaskedIGemmBackend)

        x = torch.randn(grid.total_voxels, cin, device="cuda", dtype=dtype, requires_grad=True)
        w = (torch.randn(cout, cin, 3, 1, 5, device="cuda") / 40).to(dtype).requires_grad_(True)
        b = torch.randn(cout, device="cuda", dtype=dtype, requires_grad=True)
        xr = x.detach().float().requires_grad_(True)
        wr = w.detach().float().requires_grad_(True)
        br = b.detach().float().requires_grad_(True)
        with _tf32(True):
            y = plan.execute(grid.jagged_like(x), w, b).jdata
            gy = torch.randn_like(y)
            y.backward(gy)
        with _tf32(False):
            yr = ref_plan.execute(grid.jagged_like(xr), wr, br).jdata
            yr.backward(gy.float())

        tol = {torch.float16: 4e-3, torch.float32: 5e-3, torch.float64: 1e-5}[dtype]
        self.assertEqual("gather_scatter" in plan._backend.fallback, dtype == torch.float64)
        self.assertLess(_rel_err(y, yr), tol)
        self.assertLess(_rel_err(x.grad, xr.grad), tol)
        self.assertLess(_rel_err(w.grad, wr.grad), tol)
        self.assertLess(_rel_err(b.grad, br.grad), tol)

    def test_kernel_size_one(self):
        # 1x1x1 masked plans are identity matmuls, which accept SparseConv3d's 2-D weights.
        grid = _random_grid(1000, 8)
        forward = ConvolutionPlan.from_grid_batch(1, 1, grid, grid, expert_config=CONFIG)
        transposed = ConvolutionPlan.from_grid_batch_transposed(1, 1, grid, grid, expert_config=CONFIG)
        self.assertIsInstance(forward._backend, _MatmulBackend)
        self.assertIsInstance(transposed._backend, _MatmulBackend)
        conv = fvdb.nn.SparseConv3d(64, 32, 1, bias=True).cuda()
        x = grid.jagged_like(torch.randn(grid.total_voxels, 64, device="cuda"))
        ref = conv(x, ConvolutionPlan.from_grid_batch(1, 1, grid, grid))
        torch.testing.assert_close(conv(x, forward).jdata, ref.jdata)

    def test_wide_bias_runs_natively(self):
        # A float32 bias on a float16 conv keeps the float16 kernel and is added afterwards.
        torch.manual_seed(0)
        grid = _random_grid(1500, 10)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        x = torch.randn(grid.total_voxels, 64, device="cuda", dtype=torch.float16)
        w = (torch.randn(64, 64, 3, 3, 3, device="cuda") / 40).half()
        b = torch.randn(64, device="cuda")
        with _tf32(False):
            y = plan.execute(x, w, b)
        self.assertEqual(y.dtype, torch.float32)
        self.assertNotIn("gather_scatter", plan._backend.fallback)
        self.assertLess(_rel_err(y, plan.execute(x, w).float() + b), 1e-3)

    @parameterized.expand([("gather_scatter", 3, {}), ("matmul", 1, {}), ("masked_igemm", 3, CONFIG)])
    def test_rejects_bias_shape(self, _, kernel_size, config):
        grid = _random_grid(500, 8)
        plan = ConvolutionPlan.from_grid_batch(kernel_size, 1, grid, grid, expert_config=config)
        x = torch.randn(grid.total_voxels, 64, device="cuda", dtype=torch.float16)
        w = torch.randn(32, 64, kernel_size, kernel_size, kernel_size, device="cuda", dtype=torch.float16)
        for bias in (torch.zeros(1, 32, device="cuda"), torch.zeros((), device="cuda")):
            with self.assertRaisesRegex(ValueError, "bias must have shape"):
                plan.execute(x, w, bias.half())

    @parameterized.expand([("gather_scatter", 3, {}), ("matmul", 1, {}), ("masked_igemm", 3, CONFIG)])
    def test_execute_bias(self, _, kernel_size, config):
        torch.manual_seed(0)
        grid = _random_grid(1000, 8)
        plan = ConvolutionPlan.from_grid_batch(kernel_size, 1, grid, grid, expert_config=config)
        x = torch.randn(grid.total_voxels, 64, device="cuda", dtype=torch.float16)
        w = (torch.randn(32, 64, kernel_size, kernel_size, kernel_size, device="cuda") / 20).half()
        b = torch.randn(32, device="cuda", dtype=torch.float16)
        expected = plan.execute(x, w).float() + b.float()
        self.assertLess(_rel_err(plan.execute(x, w, b), expected), 4e-3)
        self.assertLess(_rel_err(plan.execute(grid.jagged_like(x), w, b).jdata, expected), 4e-3)

    def test_coverage_report(self):
        grid = _random_grid(1500, 10)
        report = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG).coverage_report
        expected = ConvolutionPlan.from_grid_batch(3, 1, grid, grid).coverage_report
        self.assertEqual(report, expected)


class TestMaskedIGemmCpuFallback(unittest.TestCase):
    """masked_igemm plans on CPU grids run gather-scatter, with a one-time warning."""

    def test_cpu_grid_runs_gather_scatter(self):
        gen = torch.Generator(device="cpu").manual_seed(0)
        grid = GridBatch.from_ijk(
            JaggedTensor([torch.randint(-8, 8, (400, 3), generator=gen, dtype=torch.int32) for _ in range(2)])
        )
        x = torch.randn(grid.total_voxels, 8, generator=gen)
        w = torch.randn(16, 8, 3, 3, 3, generator=gen)
        convolution_plan._WARNED_MASKED_IGEMM_FALLBACKS.clear()
        with self.assertWarnsRegex(UserWarning, r"cpu grids; running gather-scatter") as cm:
            plan = ConvolutionPlan.from_grid_batch(3, 2, grid, expert_config=CONFIG)
        self.assertEqual(os.path.basename(cm.filename), os.path.basename(__file__))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transposed = ConvolutionPlan.from_grid_batch_transposed(
                3, 2, plan.target_grid_batch, grid, expert_config=CONFIG
            )
        self.assertEqual([str(c.message) for c in caught], [])
        ref = ConvolutionPlan.from_grid_batch(3, 2, grid)
        ref_t = ConvolutionPlan.from_grid_batch_transposed(3, 2, ref.target_grid_batch, grid)

        def run(p, q, x, w):
            x = x.clone().requires_grad_(True)
            w = w.clone().requires_grad_(True)
            y = p.execute(grid.jagged_like(x), w).jdata
            z = q.execute(p.target_grid_batch.jagged_like(y), w.transpose(0, 1)).jdata
            z.square().sum().backward()
            return y, z, x.grad, w.grad

        got, want = run(plan, transposed, x, w), run(ref, ref_t, x, w)
        for g, r in zip(got[:3], want[:3]):
            torch.testing.assert_close(g, r, rtol=0, atol=0)
        # Separately built CPU gather-scatter plans order each tap's pairs differently, so weight
        # gradients agree only to float rounding even between two default plans.
        self.assertLess(((got[3] - want[3]).norm() / want[3].norm()).item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
