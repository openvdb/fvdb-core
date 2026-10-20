# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Check fVDB conv backends against FlexGEMM: forward, input gradient and weight gradient.

Both libraries get identical weights. FlexGEMM rows follow the input coord order and fVDB rows
follow NanoVDB order, so fVDB results are remapped through ijk_to_index before comparing.

    python check.py --backends default
"""
import argparse
import math

import flex_gemm
import torch
from common import coords_to_jagged, sphere_coords
from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d

import fvdb


def rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def check(backend, RES, C, dtype, solid):
    feats, coords, shape = sphere_coords(RES, C, 1, dtype, solid=solid)
    ijk = coords_to_jagged(coords, 1)
    grid = fvdb.GridBatch.from_ijk(ijk)
    idx = grid.ijk_to_index(ijk).jdata.long()  # flex row r -> fVDB row idx[r]

    W = (torch.randn(C, C, 3, 3, 3, device="cuda") / math.sqrt(27 * C)).to(dtype)
    gy = torch.randn_like(feats)

    # FlexGEMM: weight (Co, K0, K1, K2, Ci).
    xf = feats.clone().requires_grad_(True)
    Wf = W.permute(0, 2, 3, 4, 1).contiguous().requires_grad_(True)
    bf = torch.zeros(C, device="cuda", dtype=dtype, requires_grad=True)
    nc = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1), True)
    yf = sparse_submanifold_conv3d(xf, coords, shape, Wf, bf, nc)[0]
    yf.backward(gy)

    # fVDB: weight (Co, Ci, K0, K1, K2), rows in NanoVDB order.
    x_nvdb = torch.empty_like(feats)
    x_nvdb[idx] = feats
    gy_nvdb = torch.empty_like(gy)
    gy_nvdb[idx] = gy
    xv = x_nvdb.clone().requires_grad_(True)
    Wv = W.clone().requires_grad_(True)
    plan = fvdb.ConvolutionPlan.from_grid_batch(
        3, 1, grid, grid, expert_config={"backend": backend}, channel_pairs=((C, C),)
    )
    yv = plan.execute(grid.jagged_like(xv), Wv).jdata
    yv.backward(gy_nvdb)

    errs = {
        "fwd": rel_err(yv[idx], yf),
        "dgrad": rel_err(xv.grad[idx], xf.grad),
        "wgrad": rel_err(Wv.grad.permute(0, 2, 3, 4, 1), Wf.grad),
    }
    return coords.shape[0], errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="default")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--configs", default="16:64,32:64,64:128,128:32", help="RES:C,RES:C")
    ap.add_argument("--solid", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    flex_gemm.kernels.triton.spconv.config.allow_tf32 = False
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    tol = {"fp32": 1e-4, "fp16": 1e-2}
    dtypes = {"fp32": torch.float32, "fp16": torch.float16}

    failed = False
    for backend in args.backends.split(","):
        for dname in args.dtypes.split(","):
            for cfg in args.configs.split(","):
                RES, C = (int(v) for v in cfg.split(":"))
                try:
                    N, errs = check(backend, RES, C, dtypes[dname], args.solid)
                except Exception as e:
                    print(f"{backend:20} {dname} RES={RES:4} C={C:4}  ERR {type(e).__name__}: {str(e)[:80]}")
                    continue
                ok = all(v <= tol[dname] for v in errs.values())
                failed |= not ok
                es = "  ".join(f"{k} {v:.1e}" for k, v in errs.items())
                print(f"{backend:20} {dname} RES={RES:4} C={C:4} N={N:8}  {es}  {'ok' if ok else 'FAIL'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
