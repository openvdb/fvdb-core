# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import random

import fvdb.nn as fvdbnn
import pytest
import torch
from fvdb.convolution_plan import _CUTLASS_SUPPORTED_CHANNELS

import fvdb

torch.backends.cudnn.deterministic = True

random.seed(42)
torch.manual_seed(42)
PTS_CACHE = [torch.empty((10_000, 3), dtype=torch.float32).normal_() for _ in range(100)]


@pytest.mark.parametrize("i_ch", [3, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("o_ch", [3, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("backend", ["default", "cutlass", "me", "halo", "igemm_mode0", "igemm_mode1", "igemm_mode2"])
@pytest.mark.benchmark(
    group="sparse_conv3d",
    warmup=True,
    warmup_iterations=3,
)
def test_forward_conv3d(benchmark, i_ch, o_ch, backend):
    if backend == "cutlass":
        if (i_ch, o_ch) not in _CUTLASS_SUPPORTED_CHANNELS:
            pytest.skip(f"Cutlass backend does not support channel pair {i_ch, o_ch}")

    device = torch.device("cuda", torch.cuda.current_device())
    pts = random.choice(PTS_CACHE).to(device=device) * 4

    coords = torch.floor(pts / 0.01).to(torch.int32)
    grid = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor(coords), device=device)

    feature = torch.empty(grid.total_voxels, i_ch, dtype=torch.float32, device=device).random_()
    feature_jt = fvdb.JaggedTensor([feature])

    # Create a convolution plan with the specified backend
    plan = fvdb.ConvolutionPlan.from_grid_batch(
        kernel_size=3, stride=1, source_grid=grid, channel_pairs=((i_ch, o_ch),), expert_config={"backend": backend}
    )
    model = fvdbnn.SparseConv3d(in_channels=i_ch, out_channels=o_ch).to(device)

    model.eval()

    def run_model():
        return model(feature_jt, plan)

    benchmark.pedantic(run_model, iterations=10, rounds=20)
