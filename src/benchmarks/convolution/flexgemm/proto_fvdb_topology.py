# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Prototype: fVDB topology driving FlexGEMM's Triton masked implicit GEMM kernels.

Builds FlexGEMM's [N, 27] neighbor map from fVDB's gather-scatter rulebook, with rows in NanoVDB
leaf order, then times FlexGEMM's kernels under different tile orderings:

    gray      FlexGEMM's global Gray-code sort
    leaf      identity, rows stay in NanoVDB leaf order
    leafgray  Gray-code sort within each 8^3 leaf
    stock     FlexGEMM end to end (its own hash map and coord order), for reference

    python proto_fvdb_topology.py --mode fp16
"""
import argparse
import json
import math

import flex_gemm
import torch
from common import coords_to_jagged, set_precision, sphere_coords, time_cuda
from flex_gemm import kernels as fg_kernels
from flex_gemm.ops.spconv import SubMConv3dFunction, sparse_submanifold_conv3d
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dNeighborCache

import fvdb

MISSING = 0xFFFFFFFF
CONFIGS = [(16, 1024, 8), (64, 1024, 2), (256, 256, 2), (1024, 64, 2)]


def fvdb_neighbor_map(grid: fvdb.GridBatch, C: int) -> torch.Tensor:
    """[N, 27] uint32 neighbor map from fVDB's rulebook. Row i is output voxel i (leaf order)."""
    plan = fvdb.ConvolutionPlan.from_grid_batch(3, 1, grid, grid, channel_pairs=((C, C),))
    topo = plan._backend.topology
    N, K = topo.output_total_voxels, topo.kernel_volume
    counts = (topo.offsets[1:] - topo.offsets[:-1]).to("cuda")
    tap = torch.repeat_interleave(torch.arange(K, device="cuda"), counts)
    nmap = torch.full((N * K,), -1, dtype=torch.int64, device="cuda")
    nmap[topo.scatter_indices.long() * K + tap] = topo.gather_indices.long()
    return nmap.view(N, K).to(torch.uint32)


def gray_to_binary(gray: torch.Tensor, V: int) -> torch.Tensor:
    binary = gray.clone()
    for v in range(1, V):
        binary ^= gray >> v
    return binary


def leaf_ids(grid: fvdb.GridBatch) -> torch.Tensor:
    """Per-voxel leaf ordinal; voxels are leaf-contiguous in NanoVDB order."""
    leaf = grid.ijk.jdata >> 3
    change = torch.ones(leaf.shape[0], dtype=torch.int64, device="cuda")
    change[1:] = (leaf[1:] != leaf[:-1]).any(dim=1).long()
    return torch.cumsum(change, 0) - 1


def make_cache(nmap: torch.Tensor, order: str, grid: fvdb.GridBatch) -> SubMConv3dNeighborCache:
    gray, sorted_idx, vsi, vso, vseg = fg_kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(nmap)
    N, V = nmap.shape
    if order == "leaf":
        sorted_idx = torch.arange(N, device="cuda", dtype=torch.int64)
    elif order == "leafgray":
        key = (leaf_ids(grid) << 32) | gray_to_binary(gray, V).long()
        sorted_idx = torch.argsort(key)
    elif order != "gray":
        raise ValueError(order)
    return SubMConv3dNeighborCache(
        neighbor_map=nmap,
        gray_code=gray,
        sorted_idx=sorted_idx,
        valid_signal_i=vsi,
        valid_signal_o=vso,
        valid_signal_seg=vseg,
    )


def tap_efficiency(cache: SubMConv3dNeighborCache, block: int) -> float:
    """Useful (row, tap) products over computed ones for tiles of `block` sorted rows."""
    gray = cache.gray_code[cache.sorted_idx].long()
    N = gray.shape[0]
    pad = (-N) % block
    g = torch.cat([gray, torch.zeros(pad, dtype=gray.dtype, device=gray.device)]).view(-1, block)
    union = torch.zeros(g.shape[0], dtype=torch.int64, device=g.device)
    for v in range(27):
        union |= (g & (1 << v)).amax(dim=1)
    useful = sum(int(((gray >> v) & 1).sum()) for v in range(27))
    computed = sum(int(((union >> v) & 1).sum()) for v in range(27)) * block
    return useful / computed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "tf32", "fp32"], default="fp16")
    ap.add_argument("--orders", default="stock,gray,leaf,leafgray")
    ap.add_argument("--configs", default=None, help="RES:C:L,...")
    ap.add_argument("--solid", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = set_precision(args.mode)
    flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]

    out = {}
    for RES, C, L in configs:
        feats, coords, shape = sphere_coords(RES, C, 1, dtype, solid=args.solid)
        ijk = coords_to_jagged(coords, 1)
        grid = fvdb.GridBatch.from_ijk(ijk)
        # fVDB rows: NanoVDB order. Coords are only used by FlexGEMM when it builds its own cache.
        nvdb_coords = torch.cat([torch.zeros_like(grid.ijk.jdata[:, :1]), grid.ijk.jdata], 1).int().contiguous()
        idx = grid.ijk_to_index(ijk).jdata.long()
        x_nvdb = torch.empty_like(feats)
        x_nvdb[idx] = feats

        params = [
            (
                torch.nn.Parameter(torch.randn(C, 3, 3, 3, C, device="cuda", dtype=dtype) / math.sqrt(27 * C)),
                torch.nn.Parameter(torch.zeros(C, device="cuda", dtype=dtype)),
            )
            for _ in range(L)
        ]

        def run(x, crd, nc):
            def step():
                for w, b in params:
                    if w.grad is not None:
                        w.grad.zero_()
                        b.grad.zero_()
                h = x
                for w, b in params:
                    h = sparse_submanifold_conv3d(h, crd, shape, w, b, nc)[0]
                h.backward(torch.ones_like(h))

            return step

        row = {"N": int(coords.shape[0])}
        ref = None
        for order in args.orders.split(","):
            if order == "stock":
                topo_fn = lambda: SubMConv3dFunction._compute_neighbor_cache(coords, shape, (3, 3, 3), (1, 1, 1), True)
                x, crd = feats, coords
            else:
                topo_fn = lambda o=order: make_cache(fvdb_neighbor_map(fvdb.GridBatch.from_ijk(ijk), C), o, grid)
                x, crd = x_nvdb, nvdb_coords
            nc = topo_fn()
            topo_ms, _ = time_cuda(topo_fn, iters=10, warmup=2)
            step_ms, gb = time_cuda(run(x, crd, nc), iters=20, warmup=2)

            # Numerical agreement of one forward pass against the first ordering.
            y = sparse_submanifold_conv3d(x, crd, shape, params[0][0], params[0][1], nc)[0]
            y = y[idx] if order != "stock" else y
            err = 0.0 if ref is None else ((y - ref).abs().max() / ref.abs().max()).item()
            ref = y if ref is None else ref

            eff = tap_efficiency(nc, 64)
            row[order] = {"topo_ms": topo_ms, "step_ms": step_ms, "peak_gb": gb, "tap_eff64": eff, "rel_err": err}
            print(
                f"RES={RES:5} C={C:5} L={L:2} {order:9} topo {topo_ms:7.2f} ms  step {step_ms:8.2f} ms  "
                f"peak {gb:5.2f} GB  tap-eff(64) {eff:5.3f}  err {err:.1e}",
                flush=True,
            )
            torch.cuda.empty_cache()
        out[f"{RES},{C},{L}"] = row

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
