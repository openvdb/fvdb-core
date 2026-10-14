# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the masked implicit GEMM (masked_igemm) submanifold convolution backend.

The backend runs submanifold convolution only: target grid == source grid, stride 1, odd kernel
sizes with volume <= 32, CUDA SM80+, float16/bfloat16 features, channel counts divisible by 64.
Forward, input gradient and weight gradient are checked against the default gather-scatter
backend evaluated in float32.
"""

import contextlib
import os
import unittest

import torch
from fvdb.convolution_plan import _MaskedIGemmBackend
from fvdb.functional._dispatch import _get_grid_data
from parameterized import parameterized

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor, _fvdb_cpp

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
def _tf32(enabled: bool):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


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
        ]
    )
    def test_random_grid(self, cin, cout, dtype):
        self._check(_random_grid(3000, 12), cin, cout, 3, dtype)

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_shell(self, dtype):
        self._check(_shell_grid(48), 128, 128, 3, dtype)

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

    @parameterized.expand([("global",), ("local",)])
    def test_row_order(self, order):
        # The local order regroups rows in windows of 65536, so the grid spans several windows.
        grid = _random_grid(200000, 40)
        self.assertGreater(grid.total_voxels, 2 * 65536)
        os.environ["FVDB_MIGEMM_ORDER"] = order
        try:
            self._check(grid, 64, 64, 3, torch.float16)
        finally:
            del os.environ["FVDB_MIGEMM_ORDER"]

    def test_neighbor_map(self):
        grid = _random_grid(800, 6)
        topo = _fvdb_cpp.migemm_build_topology(_get_grid_data(grid), torch.tensor([3, 3, 3], dtype=torch.int32))
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
            ("sparse", (3, 3, 3)),
            ("dense", (3, 3, 3)),
            ("shell", (3, 3, 3)),
            ("batch", (3, 3, 3)),
            ("view", (3, 3, 3)),
            ("sparse", (3, 5, 1)),
            ("dense", (17, 1, 1)),
            ("sparse", (19, 1, 1)),
        ]
    )
    def test_staged_builder_matches_scan(self, kind, kernel_size):
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
        ks = torch.tensor(kernel_size, dtype=torch.int32)
        data = _get_grid_data(grid)
        os.environ["FVDB_MIGEMM_BUILDER"] = "scan"
        try:
            ref = _fvdb_cpp.migemm_build_topology(data, ks)
        finally:
            del os.environ["FVDB_MIGEMM_BUILDER"]
        os.environ["FVDB_MIGEMM_BUILDER"] = "staged"
        try:
            topo = _fvdb_cpp.migemm_build_topology(data, ks)
        finally:
            del os.environ["FVDB_MIGEMM_BUILDER"]
        for name in ("neighbor_map", "row_masks", "pair_in", "pair_out", "pair_offsets"):
            self.assertTrue(torch.equal(getattr(topo, name), getattr(ref, name)), name)

    def test_rejects_unsupported(self):
        grid = _random_grid(500, 8)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(3, 2, grid, expert_config=CONFIG)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(3, 1, grid, grid.conv_grid(3, 1), expert_config=CONFIG)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(4, 1, grid, grid, expert_config=CONFIG)

    @parameterized.expand([((64, 128),), ((128, 64),), ((192, 256),)])
    def test_tf32(self, channels):
        with _tf32(True):
            self._check(_random_grid(3000, 12), *channels, 3, torch.float32, tol=5e-3)

    def test_tf32_shell(self):
        with _tf32(True):
            self._check(_shell_grid(48), 128, 128, 3, torch.float32, tol=5e-3)

    @parameterized.expand(
        [
            ("strict_fp32", 64, 64, torch.float32),
            ("float64", 64, 64, torch.float64),
            ("channels", 48, 64, torch.float16),
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

    def test_module_and_transpose(self):
        grid = _random_grid(1000, 8)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        conv = fvdb.nn.SparseConv3d(64, 64, 3, bias=True).cuda().half()
        x = grid.jagged_like(torch.randn(grid.total_voxels, 64, device="cuda", dtype=torch.float16))
        y = conv(x, plan)
        ref = conv(x, ConvolutionPlan.from_grid_batch(3, 1, grid, grid))
        self.assertLess(_rel_err(y.jdata, ref.jdata), 4e-3)

        transposed = ConvolutionPlan.from_plan_transposed(plan)
        self.assertEqual(transposed.execute(y, conv.weight.transpose(0, 1).contiguous()).jdata.shape[1], 64)


if __name__ == "__main__":
    unittest.main()
