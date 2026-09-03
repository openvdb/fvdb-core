# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Benchmark generated-topology grid construction (issue #755).

Times the per-call cost of conv_grid / conv_transpose_grid / refined_grid / coarsened_grid and of
ConvolutionPlan construction with generated targets, as a function of batch size. Before the
batched leaf-mask builder these scaled linearly in batch size (~1.5 ms fixed overhead per member);
after, they should be near-flat in B.

Usage:
    python src/benchmarks/convolution/benchmark_conv_grid_build.py [--json results.json] [--gso]

--gso additionally runs the issue's verbatim repro on the GSO shoes dataset (requires the
dataset download used by fvdb.utils.examples.load_gso_shoes).
"""

import argparse
import json
import time

import torch

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor


def make_shell_batch(batch_size: int, resolution: int = 64, device: str = "cuda") -> GridBatch:
    """B roughly-spherical shells of ~`resolution`^2*3 voxels each (surface-like sparsity,
    similar occupancy statistics to meshes voxelized at `resolution`)."""
    ijks = []
    for b in range(batch_size):
        torch.manual_seed(1234 + b)
        n = resolution * resolution * 6
        pts = torch.randn(n, 3, dtype=torch.float64)
        pts = pts / pts.norm(dim=-1, keepdim=True)
        radius = 0.35 + 0.05 * (b % 5) / 5.0
        ijk = ((pts * radius + 0.5) * resolution).floor().to(torch.int32)
        ijks.append(torch.unique(ijk, dim=0))
    jt = JaggedTensor([t.to(device) for t in ijks])
    return GridBatch.from_ijk(jt, voxel_sizes=1.0 / resolution, origins=0.0)


def time_op(fn, warmup: int = 3, iters: int = 20) -> float:
    """Median wall time of fn() in milliseconds, CUDA-event timed."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def bench_batch(grid: GridBatch) -> dict:
    ops = {
        "conv_transpose_grid k2s2": lambda: grid.conv_transpose_grid(kernel_size=2, stride=2),
        "conv_grid k2s2": lambda: grid.conv_grid(kernel_size=2, stride=2),
        "conv_grid k3s1": lambda: grid.conv_grid(kernel_size=3, stride=1),
        "refined_grid x2": lambda: grid.refined_grid(2),
        "coarsened_grid x2": lambda: grid.coarsened_grid(2),
        "plan from_grid_batch k2s2": lambda: ConvolutionPlan.from_grid_batch(2, 2, grid),
        "plan from_grid_batch_transposed k2s2": lambda: ConvolutionPlan.from_grid_batch_transposed(2, 2, grid),
    }
    return {name: time_op(fn) for name, fn in ops.items()}


def bench_pyramid(grid: GridBatch, levels: int = 4) -> float:
    """The generative-training pattern: rebuild the full conv_grid pyramid + per-level plans."""

    def build():
        g = grid
        plans = []
        for _ in range(levels):
            plans.append(ConvolutionPlan.from_grid_batch(3, 1, g))
            plans.append(ConvolutionPlan.from_grid_batch(2, 2, g))
            g = g.conv_grid(kernel_size=2, stride=2)
        return plans

    return time_op(build, warmup=2, iters=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="write results to this JSON file")
    parser.add_argument("--gso", action="store_true", help="also run the issue #755 GSO repro")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 32, 48])
    args = parser.parse_args()

    assert torch.cuda.is_available(), "this benchmark requires CUDA"
    device = torch.cuda.get_device_name()
    print(f"device: {device}")

    results = {"device": device, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "batches": {}}

    for batch_size in args.batch_sizes:
        grid = make_shell_batch(batch_size)
        entry = bench_batch(grid)
        entry["pyramid 4-level (8 plans + 3 conv_grids)"] = bench_pyramid(grid)
        entry["total_voxels"] = int(grid.total_voxels)
        results["batches"][batch_size] = entry
        print(f"\nbatch_size={batch_size} (total voxels {grid.total_voxels}):")
        for name, ms in entry.items():
            if isinstance(ms, float):
                print(f"  {name:45s} {ms:8.3f} ms")

    if args.gso:
        from fvdb.utils.examples import load_gso_shoes

        meshes = load_gso_shoes(limit=16)
        v = JaggedTensor([(m[0] - m[0].amin(0)) / m[0].amax() * 0.96 + 0.02 for m in meshes])
        f = JaggedTensor([m[1].int() for m in meshes])
        g = GridBatch.from_mesh(v, f, voxel_sizes=1 / 64, origins=0.0)
        ms = time_op(lambda: ConvolutionPlan.from_grid_batch_transposed(2, 2, g))
        results["gso_from_grid_batch_transposed_k2s2_ms"] = ms
        print(f"\nGSO shoes B=16: ConvolutionPlan.from_grid_batch_transposed(2,2,g): {ms:.3f} ms")

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
