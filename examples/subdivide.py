# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import time

import fvdb.viz as fviz
import polyscope as ps
import torch
from fvdb.utils.examples import load_dragon_mesh

from fvdb import Grid


def main():
    logging.basicConfig(level=logging.INFO)
    logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.float32

    vox_size = 0.01
    vox_origin = (0.0, 0.0, 0.0)
    p, n = load_dragon_mesh(device=device, dtype=dtype)

    index0 = Grid.from_points(p, vox_size, vox_origin)
    grids = [index0]

    logging.info("Splatting into grid...")
    start = time.time()
    nsplat = index0.splat_trilinear(p, n)
    if device == "cuda":
        torch.cuda.synchronize()
    logging.info(f"Done in {time.time() - start}s!")

    logging.info("Building refined grids")
    start = time.time()
    for i in range(2):
        subdiv_factor = i + 1
        mask = torch.rand(grids[i].num_voxels, device=device) > 0.5
        grids.append(grids[-1].refined_grid(subdiv_factor, mask))
        assert mask.sum().item() * subdiv_factor**3 == grids[-1].num_voxels
    if device == "cuda":
        torch.cuda.synchronize()
    logging.info(f"Done in {time.time() - start}s!")

    p, n = p.cpu(), n.cpu()

    ps.init()
    ps.register_point_cloud("points", p, radius=0.00075)

    for i, index in enumerate(grids):
        dual_index = index.dual_grid()
        gp = index.ijk
        gd = dual_index.ijk
        dual_v, dual_e = fviz.grid_edge_network(index)

        dual_v = dual_v.cpu()
        dual_e = dual_e.cpu()
        gp = index.grid_to_world(gp.to(dtype)).cpu()
        gd = dual_index.grid_to_world(gd.to(dtype)).cpu()
        gp, gd = gp.cpu(), gd.cpu()

        ps.register_curve_network(f"grid edges {i}", dual_v.cpu(), dual_e.cpu(), enabled=True, radius=0.0005)
        ps.register_point_cloud(f"vox corners {i}", gd, radius=0.0005 * (i + 1))
        if i == 0:
            grid_pts = ps.register_point_cloud("vox centers", gp, radius=0.0005)
            grid_pts.add_vector_quantity("splatted normals", nsplat.cpu(), enabled=True, length=0.05, radius=0.001)
    ps.show()


if __name__ == "__main__":
    main()
