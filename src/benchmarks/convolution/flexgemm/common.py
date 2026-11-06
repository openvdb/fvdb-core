# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Shared workloads, timing and backend adapters for the fVDB vs FlexGEMM conv benchmarks."""
import math

import torch

import fvdb

try:
    import flex_gemm
    from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d

    HAVE_FLEX_GEMM = True
except ImportError:
    HAVE_FLEX_GEMM = False


# RES, C, L from FlexGEMM's tests/submconv/train_benchmark.py, extended to RES=1024.
SHELL_CONFIGS = [
    (8, 1024, 16),
    (16, 1024, 8),
    (32, 1024, 4),
    (64, 1024, 2),
    (128, 512, 2),
    (256, 256, 2),
    (512, 128, 2),
    (1024, 64, 2),
]

SOLID_CONFIGS = [
    (32, 256, 2),
    (64, 128, 2),
    (128, 64, 2),
]


@torch.no_grad()
def sphere_coords(res: int, ch: int, batch_size: int = 1, dtype=torch.float32, solid: bool = False):
    """Voxelized sphere shell (FlexGEMM's sphere_coords) or solid ball.

    Returns (feats [N, ch], coords [N, 4] int32 as (b, i, j, k), shape).
    Shell output matches FlexGEMM's tests/utils.py voxel for voxel.
    """
    chunks = []
    for i in range(0, res, 256):
        for j in range(0, res, 256):
            for k in range(0, res, 256):
                ijk = torch.stack(
                    torch.meshgrid(
                        torch.arange(i, min(i + 256, res), device="cuda"),
                        torch.arange(j, min(j + 256, res), device="cuda"),
                        torch.arange(k, min(k + 256, res), device="cuda"),
                        indexing="ij",
                    ),
                    dim=-1,
                ).int()
                dist = ((ijk.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
                active = dist <= res / 2
                if not solid:
                    active &= dist >= res / 2 - 1.25
                offset = torch.tensor([i, j, k], device="cuda", dtype=torch.int32)
                chunks.append(torch.nonzero(active).int() + offset)
    ijk = torch.cat(chunks, dim=0)
    bidx = torch.arange(batch_size, device="cuda").repeat_interleave(ijk.shape[0]).int()
    coords = torch.cat([bidx.unsqueeze(-1), ijk.repeat(batch_size, 1)], dim=-1).contiguous()
    feats = torch.randn(coords.shape[0], ch, device="cuda", dtype=dtype)
    return feats, coords, torch.Size([batch_size, ch, res, res, res])


def coords_to_jagged(coords: torch.Tensor, batch_size: int) -> fvdb.JaggedTensor:
    ijk = coords[:, 1:].contiguous()
    counts = torch.bincount(coords[:, 0].long(), minlength=batch_size).tolist()
    return fvdb.JaggedTensor(list(torch.split(ijk, counts)))


def set_precision(mode: str):
    """mode in {fp16, tf32, fp32}; returns the feature dtype."""
    tf32 = mode != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if HAVE_FLEX_GEMM:
        flex_gemm.kernels.triton.spconv.config.allow_tf32 = tf32
    return torch.float16 if mode == "fp16" else torch.float32


def zero_grad(params):
    """In-place zeroing, as FlexGEMM's tests/utils.py does, so grad buffers persist across steps."""
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def time_cuda(fn, iters: int = 20, warmup: int = 2):
    """Mean ms per call and peak GB over the timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**30


# ---------------------------------------------------------------------------------------------
# Training-step adapters. Each returns a zero-arg step closure.
#
# topology="per-iter" rebuilds grid + plan (or neighbor cache) every step, as FlexGEMM's README
# benchmark does. topology="fixed" builds it once, as a training loop on a fixed grid would.
# ---------------------------------------------------------------------------------------------


def fvdb_step(backend: str, feats, coords, batch_size: int, C: int, L: int, topology: str):
    model = torch.nn.ModuleList([fvdb.nn.SparseConv3d(C, C, 3, bias=True).cuda().to(feats.dtype) for _ in range(L)])
    ijk = coords_to_jagged(coords, batch_size)
    expert = {"backend": backend}

    def build():
        grid = fvdb.GridBatch.from_ijk(ijk)
        plan = fvdb.ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=expert, channel_pairs=((C, C),))
        return grid, plan

    fixed = build() if topology == "fixed" else None

    def step():
        zero_grad(model.parameters())
        grid, plan = fixed if fixed is not None else build()
        h = grid.jagged_like(feats)
        for layer in model:
            h = layer(h, plan)
        h = h.jdata
        h.backward(torch.ones_like(h))

    return step


def flex_step(feats, coords, shape, C: int, L: int, topology: str):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    dt = feats.dtype
    params = [
        (
            torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device="cuda", dtype=dt) / math.sqrt(27 * C)),
            torch.nn.Parameter(torch.zeros(C, device="cuda", dtype=dt)),
        )
        for _ in range(L)
    ]

    def build():
        return SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1), True)

    fixed = build() if topology == "fixed" else None

    def step():
        zero_grad([p for wb in params for p in wb])
        nc = fixed if fixed is not None else build()
        h = feats
        for w, b in params:
            # FlexGEMM's backward crashes with bias=None, so a bias is always passed.
            h = sparse_submanifold_conv3d(h, coords, shape, w, b, nc)[0]
        h.backward(torch.ones_like(h))

    return step
