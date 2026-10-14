# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""fVDB vs FlexGEMM strided-conv training benchmark.

Times one strided convolution (default 2x2x2, stride 2: a U-Net encoder downsample) forward and
backward, with input, weight and bias gradients. Configs follow FlexGEMM's tests/spconv/bwd.py.

    python bench_strided.py --mode fp16 --topology fixed --json out.json
    python bench_strided.py --mode tf32 --topology per-iter --backends masked_igemm,flex_gemm
"""
import argparse
import json
import math

import torch
from common import HAVE_FLEX_GEMM, coords_to_jagged, set_precision, sphere_coords, time_cuda, zero_grad

import fvdb

if HAVE_FLEX_GEMM:
    import flex_gemm
    from flex_gemm.ops.spconv import SparseConv3dFunction, sparse_conv3d

# RES, C, batch from FlexGEMM's tests/spconv/bwd.py, extended to RES=1024.
STRIDED_CONFIGS = [
    (8, 1024, 256),
    (16, 1024, 64),
    (32, 1024, 16),
    (64, 1024, 4),
    (128, 512, 4),
    (256, 256, 2),
    (512, 128, 1),
    (1024, 64, 1),
]


def fvdb_step(backend, feats, coords, batch, C, k, s, topology, weight, bias):
    ijk = coords_to_jagged(coords, batch)
    expert = {"backend": backend}
    w = torch.nn.Parameter(weight.clone())
    b = torch.nn.Parameter(bias.clone())
    # FlexGEMM rows follow the input coords; fVDB rows follow NanoVDB order.
    grid = fvdb.GridBatch.from_ijk(ijk)
    x = torch.empty_like(feats)
    x[grid.ijk_to_index(ijk, cumulative=True).jdata.long()] = feats
    x.requires_grad_(True)

    def build():
        grid = fvdb.GridBatch.from_ijk(ijk)
        return grid, fvdb.ConvolutionPlan.from_grid_batch(k, s, grid, expert_config=expert, channel_pairs=((C, C),))

    fixed = build() if topology == "fixed" else None

    def forward():
        grid, plan = fixed if fixed is not None else build()
        return plan, plan.execute(grid.jagged_like(x), w, b)

    def step():
        zero_grad([w, b, x])
        out = forward()[1].jdata
        out.backward(torch.ones_like(out))

    return step, forward


def flex_step(feats, coords, shape, k, s, topology, weight, bias):
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    # FlexGEMM weights are (Co, Kw, Kh, Kd, Ci); fVDB's are (Co, Ci, k0, k1, k2) on the same taps.
    w = torch.nn.Parameter(weight.permute(0, 2, 3, 4, 1).contiguous())
    b = torch.nn.Parameter(bias.clone())
    x = feats.detach().clone().requires_grad_(True)
    # fVDB pads floor((k - 1) / 2) before each axis: fine = stride * coarse + tap - pad.
    ks, st, pad, dil = (k,) * 3, (s,) * 3, ((k - 1) // 2,) * 3, (1, 1, 1)

    def build():
        out_coords = SparseConv3dFunction._get_output_coords(coords, shape, ks, st, pad, dil)
        cache = SparseConv3dFunction._compute_neighbor_cache(coords, out_coords, shape, ks, st, pad, dil, True)
        return out_coords, cache

    fixed = build() if topology == "fixed" else None

    def forward():
        out_coords, cache = fixed if fixed is not None else build()
        out = sparse_conv3d(x, coords, shape, w, b, cache, out_coords, st, pad, dil)[0]
        return out_coords, out

    def step():
        zero_grad([w, b, x])
        out = forward()[1]
        out.backward(torch.ones_like(out))

    return step, forward


def _keys(ijk: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    ijk = ijk.long() + (1 << 20)
    return ((batch_idx.long() * (1 << 21) + ijk[:, 0]) * (1 << 21) + ijk[:, 1]) * (1 << 21) + ijk[:, 2]


@torch.no_grad()
def check(fvdb_forward, flex_forward) -> float:
    """Max relative difference between fVDB and FlexGEMM outputs at the output voxels both produce.

    FlexGEMM drops outputs outside the dense volume's bounds; fVDB keeps every reachable voxel.
    """
    plan, out = fvdb_forward()
    target = plan.target_grid_batch
    fv_keys = _keys(target.ijk.jdata, target.ijk.jidx)
    out_coords, flex_out = flex_forward()
    fx_keys = _keys(out_coords[:, 1:], out_coords[:, 0])
    if not torch.isin(fx_keys, fv_keys).all():
        raise RuntimeError("FlexGEMM produced output voxels fVDB did not")
    shared = torch.isin(fv_keys, fx_keys)
    fv_keys, fv = fv_keys[shared], out.jdata[shared].float()
    fv, fx = fv[fv_keys.argsort()], flex_out[fx_keys.argsort()].float()
    return ((fv - fx).abs().max() / fx.abs().max()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "tf32", "fp32"], required=True)
    ap.add_argument("--topology", choices=["per-iter", "fixed"], default="fixed")
    ap.add_argument("--kernel", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--backends", default="masked_igemm,default,flex_gemm")
    ap.add_argument("--configs", default=None, help="subset as RES:C:B,RES:C:B")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = set_precision(args.mode)
    backends = args.backends.split(",")
    if "flex_gemm" in backends and not HAVE_FLEX_GEMM:
        raise SystemExit("flex_gemm is not installed")
    configs = STRIDED_CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]
    k, s = args.kernel, args.stride

    meta = {
        "gpu": torch.cuda.get_device_name(),
        "mode": args.mode,
        "topology": args.topology,
        "kernel": k,
        "stride": s,
    }
    results = {"meta": meta, "rows": {}}
    print(f"# {meta}")
    print(f"{'RES,C,B':>14} {'N':>9}  " + "  ".join(f"{b:>22}" for b in backends) + "   max rel diff")

    for RES, C, B in configs:
        feats, coords, shape = sphere_coords(RES, C, B, dtype)
        torch.manual_seed(0)
        weight = (torch.randn(C, C, k, k, k, device="cuda") / math.sqrt(C * k**3)).to(dtype)
        bias = torch.randn(C, device="cuda").to(dtype)
        row = {"N": int(coords.shape[0])}
        cells, forwards = [], {}
        for name in backends:
            try:
                if name == "flex_gemm":
                    step, fwd = flex_step(feats, coords, shape, k, s, args.topology, weight, bias)
                else:
                    step, fwd = fvdb_step(name, feats, coords, B, C, k, s, args.topology, weight, bias)
                forwards[name] = fwd
                ms, gb = time_cuda(step, iters=args.iters)
                row[name] = {"ms": ms, "peak_gb": gb}
                cells.append(f"{ms:9.2f} ms {gb:6.2f} GB")
            except Exception as e:  # report and keep going, e.g. OOM
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
                row[name] = {"error": msg}
                cells.append(f"{'ERR':>22}")
                print(f"  {name}: {msg}")
            torch.cuda.empty_cache()
        diff = ""
        if "masked_igemm" in forwards and "flex_gemm" in forwards:
            row["max_rel_diff"] = check(forwards["masked_igemm"], forwards["flex_gemm"])
            diff = f"{row['max_rel_diff']:.1e}"
        if "flex_gemm" in row and "ms" in row["flex_gemm"]:
            for name in backends:
                if name != "flex_gemm" and "ms" in row[name]:
                    row[name]["speed_vs_flex"] = row["flex_gemm"]["ms"] / row[name]["ms"]
        results["rows"][f"{RES},{C},{B}"] = row
        print(f"{f'{RES},{C},{B}':>14} {row['N']:>9}  " + "  ".join(cells) + f"   {diff}", flush=True)
        del feats, coords, forwards
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
