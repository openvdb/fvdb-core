# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""fVDB vs FlexGEMM submanifold-conv training benchmark.

A port of FlexGEMM's tests/submconv/train_benchmark.py to fVDB's ConvolutionPlan API. Each step
runs L submanifold 3x3x3 layers forward, then backward.

    python bench_train.py --mode fp16 --topology per-iter --json out.json
    python bench_train.py --mode tf32 --topology fixed --workload solid --backends default,flex_gemm
"""
import argparse
import json

import torch
from common import (
    HAVE_FLEX_GEMM,
    SHELL_CONFIGS,
    SOLID_CONFIGS,
    flex_step,
    fvdb_step,
    set_precision,
    sphere_coords,
    time_cuda,
)


def make_step(name, feats, coords, shape, batch_size, C, L, topology):
    if name == "flex_gemm":
        return flex_step(feats, coords, shape, C, L, topology)
    return fvdb_step(name, feats, coords, batch_size, C, L, topology)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "tf32", "fp32"], required=True)
    ap.add_argument("--topology", choices=["per-iter", "fixed"], default="per-iter")
    ap.add_argument("--workload", choices=["shell", "solid"], default="shell")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--backends", default="default,flex_gemm", help="fVDB backend names and/or flex_gemm")
    ap.add_argument("--configs", default=None, help="subset as RES:C:L,RES:C:L")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = set_precision(args.mode)
    backends = args.backends.split(",")
    if "flex_gemm" in backends and not HAVE_FLEX_GEMM:
        raise SystemExit("flex_gemm is not installed")

    configs = SHELL_CONFIGS if args.workload == "shell" else SOLID_CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]

    meta = {
        "gpu": torch.cuda.get_device_name(),
        "mode": args.mode,
        "topology": args.topology,
        "workload": args.workload,
        "batch": args.batch,
    }
    results = {"meta": meta, "rows": {}}
    print(f"# {meta}")
    print(f"{'RES,C,L':>14} {'N':>9}  " + "  ".join(f"{b:>22}" for b in backends))

    for RES, C, L in configs:
        feats, coords, shape = sphere_coords(RES, C, args.batch, dtype, solid=args.workload == "solid")
        row = {"N": int(coords.shape[0])}
        cells = []
        for name in backends:
            try:
                step = make_step(name, feats, coords, shape, args.batch, C, L, args.topology)
                ms, gb = time_cuda(step, iters=args.iters)
                row[name] = {"ms": ms, "peak_gb": gb}
                cells.append(f"{ms:9.2f} ms {gb:6.2f} GB")
            except Exception as e:  # report and keep going, e.g. unsupported dtype or OOM
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
                row[name] = {"error": msg}
                cells.append(f"{'ERR':>22}")
                print(f"  {name}: {msg}")
            torch.cuda.empty_cache()
        if "flex_gemm" in row and "ms" in row["flex_gemm"]:
            for name in backends:
                if name != "flex_gemm" and "ms" in row[name]:
                    row[name]["speed_vs_flex"] = row["flex_gemm"]["ms"] / row[name]["ms"]
        results["rows"][f"{RES},{C},{L}"] = row
        print(f"{f'{RES},{C},{L}':>14} {row['N']:>9}  " + "  ".join(cells), flush=True)
        del feats, coords

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
