# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Sweep masked_igemm forward tile configs (FVDB_MIGEMM_CFG) per workload.

    python tune_tiles.py --mode fp16
"""
import argparse
import math
import os

import torch
from common import coords_to_jagged, set_precision, sphere_coords, time_cuda

import fvdb

WGRAD_CONFIGS = [
    "128x128x4",
    "128x128x3",
    "128x128x2",
    "128x64x4",
    "128x64x3",
    "64x128x4",
    "64x128x3",
    "64x64x4",
    "64x64x3",
    "64x64x2",
]
CONFIGS = [
    "128x256x3",
    "128x256x2",
    "64x256x3",
    "64x256x2",
    "128x128x4",
    "128x128x3",
    "128x128x2",
    "64x128x4",
    "64x128x3",
    "128x64x4",
    "128x64x3",
    "128x64x2",
    "64x64x4",
    "64x64x3",
]
WORKLOADS = [(8, 1024), (16, 1024), (32, 1024), (64, 1024), (128, 512), (256, 256), (512, 128), (1024, 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "bf16", "tf32"], default="fp16")
    ap.add_argument("--wgrad", action="store_true", help="sweep weight-gradient configs (timing full backward)")
    ap.add_argument(
        "--dgrad", action="store_true", help="time forward plus input gradient, which share FVDB_MIGEMM_CFG"
    )
    args = ap.parse_args()
    dtype = set_precision("tf32" if args.mode == "tf32" else "fp16")
    if args.mode == "bf16":
        dtype = torch.bfloat16

    for RES, C in WORKLOADS:
        feats, coords, _ = sphere_coords(RES, C, 1, dtype)
        grid = fvdb.GridBatch.from_ijk(coords_to_jagged(coords, 1))
        plan = fvdb.ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config={"backend": "masked_igemm"})
        x = grid.jagged_like(feats)
        W = (torch.randn(C, C, 3, 3, 3, device="cuda") / math.sqrt(27 * C)).to(dtype)
        gy = torch.randn_like(feats)
        xg = feats.clone().requires_grad_(True)

        Wg = W.clone().requires_grad_(True)
        y = plan.execute(grid.jagged_like(xg), Wg).jdata

        def bwd():
            torch.autograd.grad(y, (xg, Wg), gy, retain_graph=True)

        def fwd_dgrad():
            yx = plan.execute(grid.jagged_like(xg), W).jdata
            torch.autograd.grad(yx, xg, gy)

        fn = bwd if args.wgrad else fwd_dgrad if args.dgrad else (lambda: plan.execute(x, W))
        var = "FVDB_MIGEMM_WCFG" if args.wgrad else "FVDB_MIGEMM_CFG"
        results = []
        for cfg in WGRAD_CONFIGS if args.wgrad else CONFIGS:
            os.environ[var] = cfg
            try:
                ms, _ = time_cuda(fn, iters=20, warmup=3)
                results.append((ms, cfg))
            except Exception as e:
                results.append((float("inf"), f"{cfg} ({type(e).__name__})"))
        os.environ.pop(var, None)
        default_ms, _ = time_cuda(fn, iters=20, warmup=3)
        results.sort()
        best = ", ".join(f"{c} {t:.3f}" for t, c in results[:5])
        print(f"RES={RES:5} C={C:5}  default {default_ms:.3f} ms | best: {best}", flush=True)
        del feats, coords, grid, plan
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
