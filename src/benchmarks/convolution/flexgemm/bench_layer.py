# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Stem and head layers: one submanifold 3x3x3 conv with few channels, forward + backward.

Compares masked_igemm, fVDB's default backend and FlexGEMM on channel counts that are not
multiples of 8, which masked_igemm runs zero-padded. Stem layers take no input gradient.

    python bench_layer.py out.json
"""
import json
import math
import sys

import flex_gemm
import torch
from common import coords_to_jagged, set_precision, sphere_coords, time_cuda
from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d

import fvdb

dtype = set_precision("fp16")
flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
rows = {}
for RES in (256, 512, 1024):
    _, coords, shape = sphere_coords(RES, 1, 1, dtype)
    ijk = coords_to_jagged(coords, 1)
    grid = fvdb.GridBatch.from_ijk(ijk)
    cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1), True)
    for ci, co, input_grad in ((1, 32, False), (3, 32, False), (4, 64, False), (32, 3, True)):
        x = torch.randn(coords.shape[0], ci, device="cuda", dtype=dtype).requires_grad_(input_grad)
        res = {}
        for name in ("masked_igemm", "default"):
            plan = fvdb.ConvolutionPlan.from_grid_batch(3, 1, grid, grid, expert_config={"backend": name})
            conv = fvdb.nn.SparseConv3d(ci, co, 3, bias=True).cuda().to(dtype)

            def step(conv=conv, plan=plan):
                conv.weight.grad = conv.bias.grad = x.grad = None
                y = conv(grid.jagged_like(x), plan).jdata
                y.backward(torch.ones_like(y))

            res[name] = time_cuda(step, iters=30)[0]
            if name == "masked_igemm":
                assert "gather_scatter" not in plan._backend.fallback
        w = torch.nn.Parameter(torch.randn(co, 3, 3, 3, ci, device="cuda", dtype=dtype) / math.sqrt(27 * ci))
        b = torch.nn.Parameter(torch.zeros(co, device="cuda", dtype=dtype))

        def fstep():
            w.grad = b.grad = x.grad = None
            y = sparse_submanifold_conv3d(x, coords, shape, w, b, cache)[0]
            y.backward(torch.ones_like(y))

        res["flex_gemm"] = time_cuda(fstep, iters=30)[0]
        rows[f"{RES},{ci},{co}"] = res
        print(
            f"RES {RES:4d} {ci:2d}->{co:2d}  masked {res['masked_igemm']:6.2f}  default {res['default']:6.2f}  flex {res['flex_gemm']:6.2f} ms"
            f"  -> {res['flex_gemm']/res['masked_igemm']:.2f}x vs flex, {res['default']/res['masked_igemm']:.1f}x vs default",
            flush=True,
        )
json.dump(
    {"meta": {"gpu": torch.cuda.get_device_name(), "mode": "fp16", "layer": "one submanifold 3x3x3"}, "rows": rows},
    open(sys.argv[1], "w"),
    indent=1,
)
