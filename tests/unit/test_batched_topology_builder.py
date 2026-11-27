# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Equivalence tests for the batched topology builder (issues #755, #785).

On CUDA, ``refined_grid`` / ``coarsened_grid`` (and through them ``conv_grid`` /
``conv_transpose_grid`` for K == S) build all batch members together, in one or more
TopologyBuilder runs per pass, instead of one NanoVDB build + merge per member. These tests pin the
batched results against:

- ``from_ijk`` (PointsToGrid) grids built from independently computed expected coordinates,
  compared **elementwise** (``torch.equal`` on ``ijk.jdata``), which pins the canonical NanoVDB
  node ordering (root tiles by offset-shifted key, then x-major upper/lower child offsets), and
- the CPU implementations of the same ops, compared as per-member coordinate sets.

Emphasis is on the cases the batched pipeline must get right and a per-member loop got for free:
batches with empty members (first/middle/last/all), members of wildly unequal sizes, coordinates
straddling root-tile boundaries (+-4096) and negative octants, grids spread over many root tiles,
multi-pass factors (4 = two chained passes), and batches split into several builder runs by the
tile cap.
"""

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
from parameterized import parameterized

import fvdb
from fvdb import GridBatch, JaggedTensor


def _build(ijks, device):
    jt = JaggedTensor([t.to(device=device, dtype=torch.int32) for t in ijks])
    return GridBatch.from_ijk(jt, voxel_sizes=1.0, origins=0.0)


def _expected_refine_ijk(ijk: torch.Tensor, factor: int) -> torch.Tensor:
    """All fine coordinates of ``ijk`` subdivided by ``factor`` (unique by construction)."""
    if ijk.numel() == 0:
        return ijk.reshape(0, 3)
    offsets = torch.stack(
        torch.meshgrid(torch.arange(factor), torch.arange(factor), torch.arange(factor), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    fine = ijk.to(torch.int64)[:, None, :] * factor + offsets[None, :, :].to(ijk.device)
    return fine.reshape(-1, 3).to(torch.int32)


def _expected_coarsen_ijk(ijk: torch.Tensor, factor: int) -> torch.Tensor:
    """Unique coarse coordinates floor(ijk / factor)."""
    if ijk.numel() == 0:
        return ijk.reshape(0, 3)
    coarse = torch.div(ijk.to(torch.int64), factor, rounding_mode="floor")
    return torch.unique(coarse, dim=0).to(torch.int32)


def _ijk_sets(grid: GridBatch):
    return [set(map(tuple, t.cpu().to(torch.int64).tolist())) for t in grid.ijk.unbind()]


def _run_with_caching_allocator(script: str) -> subprocess.CompletedProcess:
    """Runs `script` in a fresh Python process with PyTorch's caching allocator on.

    The tests that measure device memory need it: CI sets PYTORCH_NO_CUDA_MEMORY_CACHING=1, under
    which torch.cuda.memory_allocated() and max_memory_allocated() report 0 and
    set_per_process_memory_fraction() is not enforced. A fresh process also starts with an empty
    cache, so free blocks cached by earlier tests cannot serve allocations a cap should refuse.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_NO_CUDA_MEMORY_CACHING"}
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


# Run by test_far_flung_voxels_do_not_size_upper_readback_by_bbox; see _run_with_caching_allocator.
_FAR_FLUNG_PEAK_SCRIPT = """
import torch
from fvdb import GridBatch, JaggedTensor

ijk = torch.tensor([[-1_000_000] * 3, [1_000_000] * 3], dtype=torch.int32, device="cuda")
grid = GridBatch.from_ijk(JaggedTensor([ijk]), voxel_sizes=1.0, origins=0.0)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
baseline = torch.cuda.memory_allocated()
assert baseline > 0, "memory stats are not being recorded"
grid.refined_grid(2)
grid.coarsened_grid(2)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() - baseline
assert peak < 256 * 1024 * 1024, f"peak device allocation {peak} bytes"
print("ok")
"""


# Run by test_failed_run_releases_builder_scratch; see _run_with_caching_allocator.
_FAILED_RUN_SCRIPT = """
import torch
from fvdb import GridBatch, JaggedTensor

ijk = torch.tensor([[k * 2048 + 10, 10, 10] for k in range(128)], dtype=torch.int32, device="cuda")
grid = GridBatch.from_ijk(JaggedTensor([ijk]), voxel_sizes=1.0, origins=0.0)
device = torch.device("cuda", torch.cuda.current_device())
torch.cuda.synchronize()
reserved = torch.cuda.memory_reserved(device)
baseline = torch.cuda.memory_allocated(device)
assert baseline > 0, "memory stats are not being recorded"
total = torch.cuda.get_device_properties(device).total_memory
torch.cuda.set_per_process_memory_fraction((reserved + 512 * 1024 * 1024) / total, device)
try:
    grid.refined_grid(2)
    raise AssertionError("refined_grid did not run out of memory under the cap")
except torch.OutOfMemoryError:
    pass
finally:
    torch.cuda.set_per_process_memory_fraction(1.0, device)
torch.cuda.synchronize()
held = torch.cuda.memory_allocated(device) - baseline
assert held <= 0, f"{held} bytes still allocated after the failed run"
refined = grid.refined_grid(2)
assert refined.total_voxels == 8 * 128, refined.total_voxels
print("ok")
"""


# Coordinate batches exercising the tricky regimes of the batched builder.
def _tricky_batches():
    torch.manual_seed(42)
    return {
        "mixed_sizes": [
            torch.randint(-8, 8, (200, 3), dtype=torch.int32),
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            torch.randint(20, 60, (500, 3), dtype=torch.int32),
        ],
        "empty_members": [
            torch.empty((0, 3), dtype=torch.int32),
            torch.randint(-10, 10, (100, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.tensor([[5, 5, 5], [5, 5, 6]], dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
        ],
        "all_empty": [
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
        ],
        # Straddles the +-4096 root-tile boundaries and negative octants, so each grid has
        # several root tiles and their order depends on the offset-shifted tile key.
        "tile_boundaries": [
            torch.tensor(
                [
                    [-4097, -4097, -4097],
                    [-4096, -4096, -4096],
                    [-2049, 0, 0],
                    [-1, -1, -1],
                    [0, 0, 0],
                    [4095, 4095, 4095],
                    [4096, 4096, 4096],
                    [4096, -4097, 0],
                ],
                dtype=torch.int32,
            ),
            torch.randint(-4200, -3900, (300, 3), dtype=torch.int32),
            torch.randint(3900, 4200, (300, 3), dtype=torch.int32),
        ],
        "single_grid": [
            torch.randint(-16, 16, (400, 3), dtype=torch.int32),
        ],
        # Small clusters scattered over many root tiles per grid, so each grid's speculative root
        # holds dozens of tiles and most of them stay empty.
        "wide_bbox": [
            torch.cat(
                [
                    torch.randint(-12000, 12000, (1, 3), dtype=torch.int32)
                    + torch.randint(0, 6, (40, 3), dtype=torch.int32)
                    for _ in range(12)
                ]
            )
            for _ in range(3)
        ],
        "larger_batch": [torch.randint(-32, 32, (50 + 37 * i, 3), dtype=torch.int32) for i in range(16)],
    }


# Point batches for from_nearest_voxels_to_points. World coordinates are chosen so that, with the
# voxel size / origin below, the index-space coordinates cover the same tricky regimes as
# _tricky_batches(): mixed member sizes, empty members (first/middle/last/all), coordinates
# straddling the +-4096 root-tile boundaries with negative components (points exactly on voxel
# boundaries included, where floor and the +1 pad land in different root tiles), single members
# (non-empty, one point, empty), and a 16-member batch.
_NEAREST_VOXEL_SIZE = 0.5  # power of two: p * (1 / voxel_size) is exact in float32
_NEAREST_ORIGIN = 0.25  # origin / voxel_size = 0.5 is exact in float32
# Keys of _nearest_voxel_point_batches(), listed so test parametrization does not build the batches
# (and reseed torch's global RNG) at import time; a test pins the two against each other.
_NEAREST_BATCH_NAMES = (
    "mixed_sizes",
    "empty_members",
    "all_empty",
    "tile_boundaries",
    "single_grid",
    "single_point",
    "single_empty",
    "larger_batch",
)


def _nearest_voxel_point_batches():
    torch.manual_seed(11)
    vs = _NEAREST_VOXEL_SIZE
    # World position of index-space value `i` (voxel boundary), so floor lands exactly on `i`.
    world = lambda i: i * vs + _NEAREST_ORIGIN  # noqa: E731
    return {
        "mixed_sizes": [
            torch.rand(2000, 3) * 8 - 4,
            torch.tensor([[0.3, -0.7, 1.1]]),
            torch.rand(50, 3) * 0.1 + 5.0,
            torch.rand(3000, 3) * 40 - 20,
        ],
        "empty_members": [
            torch.empty(0, 3),
            torch.rand(500, 3) * 6 - 3,
            torch.empty(0, 3),
            torch.tensor([[world(5), world(5), world(5)], [world(5), world(5), world(6)]]),
            torch.empty(0, 3),
        ],
        "all_empty": [torch.empty(0, 3), torch.empty(0, 3)],
        "tile_boundaries": [
            torch.tensor(
                [
                    [world(-4097), world(-4097), world(-4097)],
                    [world(-4096), world(-4096), world(-4096)],
                    [world(-4097) + 0.1, world(-4097) + 0.2, world(-4097) + 0.3],
                    [world(-2049), 0.0, 0.0],
                    [world(-1), world(-1), world(-1)],
                    [world(-1) + 0.49, world(-1) + 0.49, world(-1) + 0.49],
                    [0.0, 0.0, 0.0],
                    [world(4095), world(4095), world(4095)],
                    [world(4095) + 0.3, world(4095) + 0.3, world(4095) + 0.3],
                    [world(4096), world(4096), world(4096)],
                    [world(4096), world(-4097), world(0)],
                    [world(-8193), world(0), world(-4097)],
                ]
            ),
            torch.rand(600, 3) * (world(-3900) - world(-4200)) + world(-4200),
            torch.rand(600, 3) * (world(4200) - world(3900)) + world(3900),
            torch.cat(
                [
                    torch.rand(300, 3) * 20 - world(4110),
                    torch.rand(300, 3) * 20 + world(4080),
                ]
            ),
        ],
        "single_grid": [torch.rand(1500, 3) * 16 - 8],
        "single_point": [torch.tensor([[world(-4097) + 0.1, world(4095) + 0.4, world(-1) + 0.3]])],
        "single_empty": [torch.empty(0, 3)],
        "larger_batch": [torch.rand(50 + 137 * i, 3) * 32 - 16 for i in range(16)],
    }


def _expected_nearest_voxels_ijk(points: torch.Tensor, voxel_size: float, origin: float) -> torch.Tensor:
    """Unique floor(transform(p)) + {0,1}^3 -- the 8 voxels nearest to each point -- computed like
    the op's VoxelCoordTransform (p * (1 / voxel_size) + (-origin / voxel_size), then floor)."""
    if points.numel() == 0:
        return torch.empty((0, 3), dtype=torch.int32)
    scale = torch.tensor(1.0 / voxel_size, dtype=torch.float32)
    translate = torch.tensor(-origin / voxel_size, dtype=torch.float32)
    ijk0 = torch.floor(points.to(torch.float32) * scale + translate).to(torch.int64)
    offsets = torch.stack(torch.meshgrid(*[torch.arange(2)] * 3, indexing="ij"), dim=-1).reshape(-1, 3)
    ijk = (ijk0[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return torch.unique(ijk, dim=0).to(torch.int32)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for the batched builder")
class TestBatchedTopologyBuilder(unittest.TestCase):
    def _check_against_expected(self, result: GridBatch, expected_ijks, msg: str):
        """Elementwise-pin `result` against a from_ijk build of the expected coordinates."""
        expected = _build(expected_ijks, result.device)
        self.assertEqual(result.grid_count, expected.grid_count, msg)
        self.assertTrue(torch.equal(result.num_voxels, expected.num_voxels), msg)
        self.assertTrue(
            torch.equal(result.ijk.jdata, expected.ijk.jdata),
            f"{msg}: voxel enumeration (canonical node order) differs from PointsToGrid",
        )
        for b in range(result.grid_count):
            self.assertTrue(
                torch.equal(result.bbox_at(b).cpu(), expected.bbox_at(b).cpu()),
                f"{msg}: bbox of member {b}",
            )

    def _check_against_cpu(self, result: GridBatch, cpu_result: GridBatch, msg: str):
        self.assertEqual(result.grid_count, cpu_result.grid_count, msg)
        self.assertTrue(torch.equal(result.num_voxels.cpu(), cpu_result.num_voxels), msg)
        self.assertEqual(_ijk_sets(result), _ijk_sets(cpu_result), msg)

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_refined_grid_matches_expected_and_cpu(self, name):
        coords = _tricky_batches()[name]
        for factor in (2, 4):
            grid = _build(coords, "cuda")
            result = grid.refined_grid(factor)
            expected = [_expected_refine_ijk(c, factor) for c in coords]
            self._check_against_expected(result, expected, f"{name} refine x{factor}")
            cpu_result = _build(coords, "cpu").refined_grid(factor)
            self._check_against_cpu(result, cpu_result, f"{name} refine x{factor} vs CPU")

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_coarsened_grid_matches_expected_and_cpu(self, name):
        coords = _tricky_batches()[name]
        for factor in (2, 4):
            grid = _build(coords, "cuda")
            result = grid.coarsened_grid(factor)
            expected = [_expected_coarsen_ijk(c, factor) for c in coords]
            self._check_against_expected(result, expected, f"{name} coarsen x{factor}")
            cpu_result = _build(coords, "cpu").coarsened_grid(factor)
            self._check_against_cpu(result, cpu_result, f"{name} coarsen x{factor} vs CPU")

    def test_conv_grid_k2s2_multi_grid_matches_per_member(self):
        coords = _tricky_batches()["mixed_sizes"]
        full = _build(coords, "cuda")
        conv = full.conv_grid(kernel_size=2, stride=2)
        convt = full.conv_transpose_grid(kernel_size=2, stride=2)
        for b, c in enumerate(coords):
            single = _build([c], "cuda")
            conv_single = single.conv_grid(kernel_size=2, stride=2)
            convt_single = single.conv_transpose_grid(kernel_size=2, stride=2)
            self.assertTrue(
                torch.equal(conv.ijk.unbind()[b], conv_single.ijk.jdata),
                f"conv_grid k2s2 member {b}",
            )
            self.assertTrue(
                torch.equal(convt.ijk.unbind()[b], convt_single.ijk.jdata),
                f"conv_transpose_grid k2s2 member {b}",
            )

    @parameterized.expand([(name,) for name in _tricky_batches().keys()])
    def test_conv_stride1_and_k3s2_match_cpu(self, name):
        # Stride-1 uniform K routes through batched box-dilate passes (odd K: [-1,1]^3 per pass;
        # even K: one-sided {-1,0}^3 / {0,1}^3 passes), and k3s2 transpose through refine + one
        # negative pad pass. Pin against the CPU implementation (coordinate sets) and against a
        # from_ijk build of the CPU coordinates (elementwise: canonical node order).
        coords = _tricky_batches()[name]
        cuda_grid = _build(coords, "cuda")
        cpu_grid = _build(coords, "cpu")
        cases = (
            [("conv_grid", k, 1) for k in (2, 3, 4, 5)]
            + [("conv_transpose_grid", k, 1) for k in (2, 3, 4, 5)]
            + [("conv_transpose_grid", 3, 2)]
        )
        for op, k, s in cases:
            msg = f"{name} {op} k{k}s{s}"
            result = getattr(cuda_grid, op)(kernel_size=k, stride=s)
            reference = getattr(cpu_grid, op)(kernel_size=k, stride=s)
            self._check_against_cpu(result, reference, msg)
            expected = _build(list(reference.ijk.unbind()), "cuda")
            self.assertTrue(
                torch.equal(result.ijk.jdata, expected.ijk.jdata),
                f"{msg}: voxel enumeration (canonical node order) differs from PointsToGrid",
            )

    def test_masked_refine_multi_grid(self):
        # Masked subdivision routes through pruneGrid then the batched refine.
        coords = _tricky_batches()["mixed_sizes"]
        grid = _build(coords, "cuda")
        mask = JaggedTensor([((t.sum(-1) % 2) == 0) for t in grid.ijk.unbind()])
        result = grid.refined_grid(2, mask=mask)
        cpu_grid = _build(coords, "cpu")
        cpu_mask = JaggedTensor([m.cpu() for m in mask.unbind()])
        cpu_result = cpu_grid.refined_grid(2, mask=cpu_mask)
        self._check_against_cpu(result, cpu_result, "masked refine x2 vs CPU")

    @parameterized.expand([("tile_boundaries",), ("wide_bbox",), ("empty_members",)])
    def test_chunked_build_matches_unchunked(self, name):
        # FVDB_TOPOLOGY_MAX_TILES=1 gives every grid its own builder run (a grid needing more tiles
        # runs alone), so each pass produces several chunks, chained passes read their sources
        # across chunks, and the last pass assembles the chunks into one storage.
        coords = _tricky_batches()[name]
        grid = _build(coords, "cuda")
        ops = {
            "refine x4": lambda g: g.refined_grid(4),
            "coarsen x4": lambda g: g.coarsened_grid(4),
            "conv_grid k3s1": lambda g: g.conv_grid(kernel_size=3, stride=1),
            "conv_grid k5s1": lambda g: g.conv_grid(kernel_size=5, stride=1),
            "conv_transpose_grid k3s2": lambda g: g.conv_transpose_grid(kernel_size=3, stride=2),
        }
        for op_name, op in ops.items():
            msg = f"{name} {op_name}"
            unchunked = op(grid)
            with mock.patch.dict(os.environ, {"FVDB_TOPOLOGY_MAX_TILES": "1"}):
                chunked = op(grid)
            self.assertEqual(chunked.grid_count, unchunked.grid_count, msg)
            self.assertTrue(torch.equal(chunked.ijk.jdata, unchunked.ijk.jdata), msg)
            self.assertTrue(torch.equal(chunked.ijk.joffsets, unchunked.ijk.joffsets), msg)
            for b in range(chunked.grid_count):
                self.assertTrue(
                    torch.equal(chunked.bbox_at(b).cpu(), unchunked.bbox_at(b).cpu()),
                    f"{msg}: bbox of member {b}",
                )

    @parameterized.expand([("1e3",), ("64k",), ("abc",), ("-5",), ("0",), ("99999999999999999999",)])
    def test_invalid_max_tiles_env_raises(self, value):
        grid = _build(_tricky_batches()["single_grid"], "cuda")
        with mock.patch.dict(os.environ, {"FVDB_TOPOLOGY_MAX_TILES": value}):
            with self.assertRaisesRegex(RuntimeError, "FVDB_TOPOLOGY_MAX_TILES must be a positive integer"):
                grid.refined_grid(2)

    def test_far_flung_voxels_do_not_size_upper_readback_by_bbox(self):
        # Two voxels at +-1e6 span about 489^3 root tiles of bbox but hold only two upper nodes.
        # Bounding the upper-node readback by bbox volume alone asked for about 2.8 GB here.
        # The peak is measured in a fresh process (see _run_with_caching_allocator); the outputs
        # are checked here.
        result = _run_with_caching_allocator(_FAR_FLUNG_PEAK_SCRIPT)
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("ok", result.stdout)
        coords = [torch.tensor([[-1_000_000] * 3, [1_000_000] * 3], dtype=torch.int32)]
        grid = _build(coords, "cuda")
        refined = grid.refined_grid(2)
        coarsened = grid.coarsened_grid(2)
        self._check_against_expected(refined, [_expected_refine_ijk(c, 2) for c in coords], "far-flung refine x2")
        self._check_against_expected(coarsened, [_expected_coarsen_ijk(c, 2) for c in coords], "far-flung coarsen x2")

    def test_failed_run_releases_builder_scratch(self):
        # One voxel pair per source upper node, 4096 apart in x, so refining needs 128 speculative
        # root tiles and about 2 GiB of densified lower masks. Cap the allocator well below that:
        # the run fails after allocating its roots and upper masks, and must not leave them held
        # by the cached builder. The next run must still work. The cap needs the caching allocator
        # and an empty cache, so this runs in a fresh process (see _run_with_caching_allocator).
        result = _run_with_caching_allocator(_FAILED_RUN_SCRIPT)
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("ok", result.stdout)

    def test_refine_coarsen_roundtrip_grid_count(self):
        # Serialization-visible metadata: grid_count/address arithmetic must be consistent for a
        # multi-grid handle produced by the batched builder.
        coords = _tricky_batches()["empty_members"]
        grid = _build(coords, "cuda")
        fine = grid.refined_grid(2)
        back = fine.coarsened_grid(2)
        self.assertEqual(fine.grid_count, grid.grid_count)
        self.assertEqual(back.grid_count, grid.grid_count)
        self.assertEqual(_ijk_sets(back), _ijk_sets(grid))
        self.assertTrue(torch.equal(back.num_voxels, grid.num_voxels))

    # -- from_dense: build one member, replicate the buffer B times (issue #775 item 4) --------

    def _check_from_dense_matches_per_member(self, num_grids, dims, ijk_min, mask, msg):
        """Multi-member from_dense equals B independent single-member builds, elementwise."""
        batched = GridBatch.from_dense(num_grids, dims, ijk_min, 1.0, 0.0, mask=mask, device="cuda")
        single = GridBatch.from_dense(1, dims, ijk_min, 1.0, 0.0, mask=mask, device="cuda")
        self.assertEqual(batched.grid_count, num_grids, msg)
        self.assertEqual(single.grid_count, 1, msg)
        self.assertTrue(torch.equal(batched.num_voxels, single.num_voxels.repeat(num_grids)), msg)
        self.assertEqual(batched.total_voxels, num_grids * single.total_voxels, msg)
        self.assertTrue(
            torch.equal(batched.ijk.jdata, single.ijk.jdata.repeat(num_grids, 1)),
            f"{msg}: voxel enumeration of replicated members differs from a single-member build",
        )
        self.assertTrue(torch.equal(batched.ijk.joffsets.cpu(), torch.arange(num_grids + 1) * single.total_voxels), msg)
        for b in range(num_grids):
            self.assertTrue(
                torch.equal(batched.bbox_at(b).cpu(), single.bbox_at(0).cpu()), f"{msg}: bbox of member {b}"
            )
            self.assertTrue(torch.equal(batched.ijk[b].jdata, single.ijk.jdata), f"{msg}: ijk of member {b}")
        # Member b of the batch is addressable as its own grid (header mGridIndex / mGridCount and
        # the handle's per-grid offset table must agree).
        idx = torch.tensor([num_grids - 1, 0], device="cuda")
        sliced = batched[idx]
        self.assertEqual(sliced.grid_count, 2, msg)
        self.assertTrue(torch.equal(sliced.ijk.jdata, single.ijk.jdata.repeat(2, 1)), msg)
        # And the whole thing matches the CPU implementation as coordinate sets.
        cpu_mask = None if mask is None else mask.cpu()
        cpu = GridBatch.from_dense(num_grids, dims, ijk_min, 1.0, 0.0, mask=cpu_mask, device="cpu")
        self._check_against_cpu(batched, cpu, f"{msg}: vs CPU")
        return batched, single

    @parameterized.expand([(b,) for b in (1, 3, 16)])
    def test_from_dense_unmasked_matches_per_member(self, num_grids):
        # Non-cubic box, negative / tile-straddling ijk_min (crosses the 8-voxel leaf boundary and
        # a 128-voxel upper-node boundary), so leaf origins are not aligned to the box corner.
        for dims, ijk_min in (([8, 8, 8], [0, 0, 0]), ([5, 12, 7], [-3, 125, 4]), ([17, 4, 33], [-4100, 0, 4090])):
            batched, single = self._check_from_dense_matches_per_member(
                num_grids, dims, ijk_min, None, f"from_dense B={num_grids} dims={dims} ijk_min={ijk_min}"
            )
            self.assertEqual(single.total_voxels, dims[0] * dims[1] * dims[2])
            lo = torch.tensor(ijk_min, dtype=torch.int32)
            hi = lo + torch.tensor(dims, dtype=torch.int32) - 1
            for b in range(num_grids):
                self.assertTrue(torch.equal(batched.bbox_at(b).cpu(), torch.stack([lo, hi])))

    def test_from_dense_replication_spans_header_blocks(self):
        # 257 members: the header-stamp kernel runs two 256-thread blocks, and the last doubling
        # copy is partial (one grid after the 256 already filled).
        self._check_from_dense_matches_per_member(257, [5, 12, 7], [-3, 125, 4], None, "from_dense B=257")

    @parameterized.expand([(b,) for b in (1, 3, 16)])
    def test_from_dense_masked_matches_per_member(self, num_grids):
        torch.manual_seed(775)
        dims = [9, 6, 14]
        ijk_min = [-2, 7, -13]
        mask = torch.rand(*dims, device="cuda") < 0.3
        batched, single = self._check_from_dense_matches_per_member(
            num_grids, dims, ijk_min, mask, f"masked from_dense B={num_grids}"
        )
        self.assertEqual(single.total_voxels, int(mask.sum().item()))
        # Selected coordinates are exactly the mask's true cells, shifted by ijk_min.
        expected = torch.nonzero(mask).to(torch.int32) + torch.tensor(ijk_min, dtype=torch.int32, device="cuda")
        self.assertEqual(set(map(tuple, expected.tolist())), _ijk_sets(single)[0])

    @parameterized.expand([(b,) for b in (1, 3, 16)])
    def test_from_dense_mask_selects_nothing(self, num_grids):
        dims = [4, 5, 6]
        mask = torch.zeros(*dims, dtype=torch.bool, device="cuda")
        batched, single = self._check_from_dense_matches_per_member(
            num_grids, dims, [1, 2, 3], mask, f"all-false mask from_dense B={num_grids}"
        )
        self.assertEqual(batched.total_voxels, 0)
        self.assertEqual(batched.ijk.jdata.shape, (0, 3))
        self.assertTrue(torch.equal(batched.num_voxels.cpu(), torch.zeros(num_grids, dtype=torch.int64)))
        # Empty members are still valid grids: topology ops on them work and stay empty.
        self.assertEqual(batched.dual_grid().total_voxels, 0)
        self.assertEqual(batched.dual_grid().grid_count, num_grids)

    def test_from_dense_replicated_members_are_independent_grids(self):
        # Ops that address each member through its own header (voxel lookup, coarsening, and
        # concatenation with another batch) must see B identical but separate grids.
        num_grids = 5
        grid = GridBatch.from_dense(num_grids, [10, 6, 3], [1, -9, 2], 1.0, 0.0, device="cuda")
        pts = torch.tensor([[1, -9, 2], [10, -4, 4], [11, -4, 4]], dtype=torch.int32, device="cuda")
        idx = grid.ijk_to_index(JaggedTensor([pts] * num_grids), cumulative=True)
        for b in range(num_grids):
            self.assertEqual(idx[b].jdata.tolist(), [b * 180, b * 180 + 179, -1])
        coarse = grid.coarsened_grid(2)
        coarse_single = GridBatch.from_dense(1, [10, 6, 3], [1, -9, 2], 1.0, 0.0, device="cuda").coarsened_grid(2)
        self.assertTrue(torch.equal(coarse.ijk.jdata, coarse_single.ijk.jdata.repeat(num_grids, 1)))
        other = _build(_tricky_batches()["mixed_sizes"], "cuda")
        cat = GridBatch.from_cat([grid, other])
        self.assertEqual(cat.grid_count, num_grids + other.grid_count)
        self.assertTrue(torch.equal(cat.ijk.jdata, torch.cat([grid.ijk.jdata, other.ijk.jdata])))

    def test_nearest_batch_names_match_batches(self):
        self.assertEqual(_NEAREST_BATCH_NAMES, tuple(_nearest_voxel_point_batches()))

    @parameterized.expand([(name,) for name in _NEAREST_BATCH_NAMES])
    def test_from_nearest_voxels_to_points_matches_expected_and_cpu(self, name):
        # CUDA: the floored-ijk base batch is padded by {0,1}^3 in one batched BoxDilate(0, +1)
        # pass instead of one PadGrid per member.
        pts = _nearest_voxel_point_batches()[name]
        vs, og = _NEAREST_VOXEL_SIZE, _NEAREST_ORIGIN
        result = GridBatch.from_nearest_voxels_to_points(
            JaggedTensor([p.cuda() for p in pts]), voxel_sizes=vs, origins=og
        )
        expected = [_expected_nearest_voxels_ijk(p, vs, og) for p in pts]
        for p, e in zip(pts, expected):
            # Every point contributes 8 candidates, so a non-empty member has at least 8 voxels.
            self.assertEqual(e.shape[0] >= 8, p.shape[0] > 0, f"{name}: expected-set sanity")
        self._check_against_expected(result, expected, f"from_nearest_voxels_to_points[{name}]")
        for b, p in enumerate(pts):
            single = GridBatch.from_nearest_voxels_to_points(JaggedTensor([p.cuda()]), voxel_sizes=vs, origins=og)
            self.assertTrue(
                torch.equal(result.ijk[b].jdata, single.ijk.jdata),
                f"from_nearest_voxels_to_points[{name}]: member {b} differs from its single-member build",
            )
        cpu = GridBatch.from_nearest_voxels_to_points(JaggedTensor(pts), voxel_sizes=vs, origins=og)
        self._check_against_cpu(result, cpu, f"from_nearest_voxels_to_points[{name}] vs CPU")

    def test_from_nearest_voxels_to_points_per_member_transforms(self):
        # Per-member voxel sizes / origins: each member's base voxels use its own transform, and the
        # output keeps each member's transform.
        torch.manual_seed(5)
        pts = [torch.rand(700, 3) * 10 - 5, torch.empty(0, 3), torch.rand(300, 3) * 3]
        sizes = [0.5, 1.0, 0.25]
        origins = [0.0, 0.25, -0.125]
        kwargs = dict(voxel_sizes=[[v] * 3 for v in sizes], origins=[[o] * 3 for o in origins])
        result = GridBatch.from_nearest_voxels_to_points(JaggedTensor([p.cuda() for p in pts]), **kwargs)
        expected = [_expected_nearest_voxels_ijk(p, v, o) for p, v, o in zip(pts, sizes, origins)]
        self._check_against_expected(result, expected, "from_nearest_voxels_to_points per-member transforms")
        for b in range(len(pts)):
            self.assertTrue(
                torch.allclose(result.voxel_size_at(b).cpu().double(), torch.tensor([sizes[b]] * 3).double())
            )
            self.assertTrue(torch.allclose(result.origin_at(b).cpu().double(), torch.tensor([origins[b]] * 3).double()))
        cpu = GridBatch.from_nearest_voxels_to_points(JaggedTensor(pts), **kwargs)
        self._check_against_cpu(result, cpu, "from_nearest_voxels_to_points per-member transforms vs CPU")


if __name__ == "__main__":
    unittest.main()
