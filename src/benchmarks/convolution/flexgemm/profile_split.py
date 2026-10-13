# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Split one conv layer's cost into grid build, plan build, forward and backward.

    python profile_split.py --mode fp16 --backends default,flex_gemm --kernels
"""
import argparse
import json
import math

import torch
from common import (
    HAVE_FLEX_GEMM,
    coords_to_jagged,
    set_precision,
    sphere_coords,
    time_cuda,
)

import fvdb

if HAVE_FLEX_GEMM:
    import flex_gemm
    from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d

DEFAULT_CONFIGS = [(16, 1024), (64, 1024), (256, 256), (1024, 64)]


def fvdb_parts(backend, feats, coords, C, W):
    ijk = coords_to_jagged(coords, 1)
    expert = {"backend": backend}

    def grid_build():
        return fvdb.GridBatch.from_ijk(ijk)

    grid = grid_build()

    def plan_build():
        return fvdb.ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config=expert, channel_pairs=((C, C),))

    plan = plan_build()
    x = grid.jagged_like(feats.clone().requires_grad_(True))
    Wp = W.clone().requires_grad_(True)

    def fwd():
        return plan.execute(x, Wp)

    def fwd_bwd():
        y = plan.execute(x, Wp).jdata
        y.backward(torch.ones_like(y))

    topo = getattr(plan._backend, "topology", None)
    if hasattr(topo, "total_pairs"):
        pairs = int(topo.total_pairs)
    elif hasattr(topo, "neighbor_map"):
        pairs = int((topo.neighbor_map >= 0).sum())
    else:
        pairs = None
    return {"grid": grid_build, "plan": plan_build, "fwd": fwd, "fwd_bwd": fwd_bwd}, pairs


def flex_parts(feats, coords, shape, C, W):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    Wf = W.permute(0, 2, 3, 4, 1).contiguous().requires_grad_(True)
    xf = feats.clone().requires_grad_(True)
    bf = torch.zeros(C, device="cuda", dtype=feats.dtype, requires_grad=True)

    def plan_build():
        return SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1), True)

    nc = plan_build()

    def fwd():
        return sparse_submanifold_conv3d(xf, coords, shape, Wf, bf, nc)[0]

    def fwd_bwd():
        y = sparse_submanifold_conv3d(xf, coords, shape, Wf, bf, nc)[0]
        y.backward(torch.ones_like(y))

    return {"grid": None, "plan": plan_build, "fwd": fwd, "fwd_bwd": fwd_bwd}


def kernel_table(fn, label):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    print(f"\n-- {label} kernels (one fwd+bwd) --")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=70))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "tf32", "fp32"], default="fp16")
    ap.add_argument("--backends", default="default,flex_gemm")
    ap.add_argument("--configs", default=None, help="RES:C,RES:C")
    ap.add_argument("--solid", action="store_true")
    ap.add_argument("--kernels", action="store_true", help="print a per-kernel table for each backend")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = set_precision(args.mode)
    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]

    out = {}
    for RES, C in configs:
        feats, coords, shape = sphere_coords(RES, C, 1, dtype, solid=args.solid)
        N = coords.shape[0]
        W = (torch.randn(C, C, 3, 3, 3, device="cuda") / math.sqrt(27 * C)).to(dtype)
        pairs = None
        rows = {}
        for name in args.backends.split(","):
            try:
                if name == "flex_gemm":
                    parts = flex_parts(feats, coords, shape, C, W)
                else:
                    parts, p = fvdb_parts(name, feats, coords, C, W)
                    pairs = pairs or p
                t = {k: (time_cuda(fn, iters=10, warmup=3)[0] if fn else 0.0) for k, fn in parts.items()}
                t["bwd"] = t["fwd_bwd"] - t["fwd"]
                rows[name] = t
                if args.kernels:
                    kernel_table(parts["fwd_bwd"], name)
            except Exception as e:
                rows[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"}
            torch.cuda.empty_cache()

        flops = 2 * (pairs or 0) * C * C
        print(f"\n== RES={RES} C={C} {args.mode} N={N} pairs={pairs} ({(pairs or 0) / N:.1f}/voxel)")
        print(f"{'':26}{'grid':>9}{'plan':>9}{'fwd':>9}{'bwd':>9}{'fwd TFLOP/s':>13}")
        for name, t in rows.items():
            if "error" in t:
                print(f"{name:26}{t['error']}")
                continue
            tf = flops / t["fwd"] / 1e9 if pairs else float("nan")
            print(f"{name:26}{t['grid']:9.2f}{t['plan']:9.2f}{t['fwd']:9.2f}{t['bwd']:9.2f}{tf:13.1f}")
        out[f"{RES},{C}"] = {"N": N, "pairs": pairs, "ms": rows}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
