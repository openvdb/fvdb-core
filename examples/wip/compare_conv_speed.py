# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import time

import numpy as np
import torch
import tqdm
from fvdb.utils.examples import load_dragon_mesh

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor


def benchmark_inplace_conv(grid: GridBatch, in_feature, in_kernel):
    start_time = time.perf_counter()
    out_feature = grid.sparse_conv_halo(in_feature, in_kernel)
    torch.cuda.synchronize()
    return time.perf_counter() - start_time


def benchmark_plan_conv(grid: GridBatch, in_feature, in_kernel):
    start_time = time.perf_counter()
    plan = ConvolutionPlan.from_grid_batch(kernel_size=in_kernel.size(-1), stride=1, source_grid=grid)
    torch.cuda.synchronize()

    plan_time = time.perf_counter()
    out_feature = plan.execute(in_feature, in_kernel)
    torch.cuda.synchronize()

    return plan_time - start_time, time.perf_counter() - plan_time


def main():
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.float32
    kernel_size = 3
    in_channel, out_channel = 128, 64

    vox_size = 0.005
    vox_origin = (0.0, 0.0, 0.0)
    p, _ = load_dragon_mesh(device=device, dtype=dtype)
    jagged_p = JaggedTensor(p)

    index0 = GridBatch.from_points(jagged_p, voxel_sizes=vox_size, origins=vox_origin)

    grid_feats = torch.rand((index0.total_voxels, in_channel), device=device, dtype=dtype) * 0.5 + 0.5
    kernels = (
        torch.rand(out_channel, in_channel, kernel_size, kernel_size, kernel_size, dtype=dtype, device=device) * 0.5
        + 0.5
    )

    torch.cuda.synchronize()

    inplace_time = []
    plan_build_time = []
    conv_time = []

    for iter in tqdm.trange(100):
        inplace = benchmark_inplace_conv(index0, grid_feats, kernels)
        plan_build, conv = benchmark_plan_conv(index0, grid_feats, kernels)
        inplace_time.append(inplace)
        plan_build_time.append(plan_build)
        conv_time.append(conv)

    inplace_time, plan_build_time, conv_time = inplace_time[5:], plan_build_time[5:], conv_time[5:]

    print(f"Num voxels = {index0.num_voxels}, channel = {in_channel} -> {out_channel}, device = {device}")
    print(f"Convolution Inplace {np.mean(inplace_time):.4f} +/- {np.std(inplace_time):.4f}")
    print(f"Plan Build {np.mean(plan_build_time):.4f} +/- {np.std(plan_build_time):.4f}")
    print(f"Plan Convolution {np.mean(conv_time):.4f} +/- {np.std(conv_time):.4f}")


if __name__ == "__main__":
    main()
