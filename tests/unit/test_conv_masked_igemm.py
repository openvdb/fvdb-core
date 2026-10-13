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

        xr = xl.detach().float().requires_grad_(True)
        wr = wl.detach().float().requires_grad_(True)
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

    def test_rejects_unsupported(self):
        grid = _random_grid(500, 8)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(3, 2, grid, expert_config=CONFIG)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(3, 1, grid, grid.conv_grid(3, 1), expert_config=CONFIG)
        with self.assertRaises(ValueError):
            ConvolutionPlan.from_grid_batch(4, 1, grid, grid, expert_config=CONFIG)
        plan = ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=CONFIG)
        with self.assertRaises(ValueError):
            plan.execute(
                grid.jagged_like(torch.randn(grid.total_voxels, 64, device="cuda")),
                torch.randn(64, 64, 3, 3, 3, device="cuda"),
            )
        with self.assertRaises(ValueError):
            plan.execute(
                grid.jagged_like(torch.randn(grid.total_voxels, 48, device="cuda", dtype=torch.float16)),
                torch.randn(64, 48, 3, 3, 3, device="cuda", dtype=torch.float16),
            )

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
