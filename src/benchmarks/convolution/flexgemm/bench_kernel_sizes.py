# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""masked_igemm vs the gather-scatter default and pred_gather_igemm across kernel sizes.

One layer in float32 (TF32 unless --tf32 0), batch 1, plan built once. For each workload, kernel
size and stride, times the forward alone, forward + backward, and the plan build, and reports the
forward's relative error against the default backend in float64. Plans are built with no target
grid, so stride-1 plans dilate the input grid. pred_gather_igemm only admits kernel sizes 3, 5, 7
and strides 1, 2.

    python bench_kernel_sizes.py --json out.json
    python bench_kernel_sizes.py --backends masked_igemm --workloads shell:512:32,solid:128:32 --json out.json
"""

import argparse
import json

import torch
from common import coords_to_jagged, sphere_coords, time_cuda

import fvdb

BACKENDS = ("pred_gather_igemm", "masked_igemm", "default")
WORKLOADS = (
    ("shell", 128, 128),
    ("shell", 256, 64),
    ("shell", 256, 128),
    ("shell", 512, 32),
    ("shell", 512, 64),
    ("shell", 1024, 32),
    ("solid", 64, 64),
    ("solid", 128, 32),
)


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def run_backend(backend: str, grid, feats, w, k: int, s: int, ref: torch.Tensor, iters: int) -> dict:
    C = feats.shape[1]

    def build():
        return fvdb.ConvolutionPlan.from_grid_batch(
            k, s, grid, expert_config={"backend": backend}, channel_pairs=((C, C),)
        )

    plan = build()
    err = rel_err(plan.execute(feats, w), ref)
    fwd_ms, _ = time_cuda(lambda: plan.execute(feats, w), iters=iters)
    x = feats.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)

    def step():
        x.grad = None
        wg.grad = None
        y = plan.execute(x, wg)
        y.backward(torch.ones_like(y))

    step_ms, peak = time_cuda(step, iters=iters)
    del plan, x, wg
    torch.cuda.empty_cache()
    build_ms, _ = time_cuda(build, iters=5, warmup=1)
    return {"fwd_ms": fwd_ms, "step_ms": step_ms, "build_ms": build_ms, "err": err, "peak_gb": peak}


def run_case(grid, feats, k: int, s: int, backends, iters: int) -> dict:
    C = feats.shape[1]
    torch.manual_seed(0)
    w = torch.randn(C, C, k, k, k, device="cuda") / (C * k**3) ** 0.5
    out: dict = {"N_in": int(feats.shape[0])}
    try:
        ref_plan = fvdb.ConvolutionPlan.from_grid_batch(k, s, grid, channel_pairs=((C, C),))
        ref = ref_plan.execute(feats.double(), w.double())
    except Exception as e:  # e.g. out of memory in float64; skip this case and keep going
        out["error"] = f"reference: {type(e).__name__}: {str(e)[:120]}"
        return out
    out["N_out"] = int(ref.shape[0])
    del ref_plan
    for b in backends:
        try:
            out[b] = run_backend(b, grid, feats, w, k, s, ref, iters)
        except Exception as e:  # noqa: BLE001
            out[b] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--kernel-sizes", default="3,5,7")
    ap.add_argument("--workloads", default=None, help="kind:res:C,... with kind shell or solid")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    backends = tuple(args.backends.split(","))
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(",")]
    workloads = WORKLOADS
    if args.workloads:
        workloads = tuple((kd, int(r), int(c)) for kd, r, c in (w.split(":") for w in args.workloads.split(",")))
        if bad := [kd for kd, _, _ in workloads if kd not in ("shell", "solid")]:
            ap.error(f"workload kind must be shell or solid; got {bad}")

    rows = {}
    print(f"{'case':>24} {'N_in':>8} | {'backend':>17} {'fwd ms':>8} {'step ms':>8} {'build':>7} {'err':>8}")
    for kind, res, C in workloads:
        _, coords, _ = sphere_coords(res, 1, solid=(kind == "solid"))
        grid = fvdb.GridBatch.from_ijk(coords_to_jagged(coords, 1))
        feats = torch.randn(grid.total_voxels, C, device="cuda")
        for k in kernel_sizes:
            for s in (1, 2):
                key = f"{kind},{res},{C},k{k},s{s}"
                rows[key] = r = run_case(grid, feats, k, s, backends, args.iters)
                if "error" in r:
                    print(f"{key:>24} {r['N_in']:>8} | ERR {r['error']}", flush=True)
                    continue
                for b in backends:
                    v = r[b]
                    if "error" in v:
                        print(f"{key:>24} {r['N_in']:>8} | {b:>17} ERR {v['error']}", flush=True)
                    else:
                        print(
                            f"{key:>24} {r['N_in']:>8} | {b:>17} {v['fwd_ms']:8.3f} {v['step_ms']:8.3f} "
                            f"{v['build_ms']:7.2f} {v['err']:8.1e}",
                            flush=True,
                        )
        del feats, coords, grid
        torch.cuda.empty_cache()
    meta = {"gpu": torch.cuda.get_device_name(), "tf32": bool(args.tf32), "dtype": "float32", "batch": 1}
    with open(args.json, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
