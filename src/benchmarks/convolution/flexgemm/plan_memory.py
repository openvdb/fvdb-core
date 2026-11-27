# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""GPU memory a ConvolutionPlan holds, and its build peak, for masked_igemm and the default backend.

    python plan_memory.py --kind shell --res 512 --kernel-size 7 --stride 1

Plans are built with no target grid, so stride-1 plans dilate the input grid.
"""

import argparse

import torch
from common import coords_to_jagged, sphere_coords

import fvdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["shell", "solid"], default="shell")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--backends", default="masked_igemm,default")
    args = ap.parse_args()

    _, coords, _ = sphere_coords(args.res, 1, solid=args.kind == "solid")
    grid = fvdb.GridBatch.from_ijk(coords_to_jagged(coords, 1))
    print(f"{args.kind} RES {args.res}, {grid.total_voxels} voxels, kernel {args.kernel_size}, stride {args.stride}")
    for backend in args.backends.split(","):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        plan = fvdb.ConvolutionPlan.from_grid_batch(
            args.kernel_size, args.stride, grid, expert_config={"backend": backend}
        )
        torch.cuda.synchronize()
        held = (torch.cuda.memory_allocated() - base) / 2**30
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        print(f"  {backend:>14}: holds {held:.2f} GiB, build peak {peak:.2f} GiB")
        del plan


if __name__ == "__main__":
    main()
