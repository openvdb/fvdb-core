# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import fvdb.viz as fviz
import numpy as np
import polyscope as ps
import torch
from fvdb.utils.examples import load_dragon_mesh

from fvdb import Grid


def main():
    device = "cuda"

    vox_size = 0.0075
    vox_origin = (0, 0, 0)
    N = 1

    [p] = load_dragon_mesh(mode="v", skip_every=N, device=torch.device(device))

    index = Grid.from_points(p, vox_size, vox_origin)

    primal_voxels = index.ijk

    nhood = index.neighbor_indexes(primal_voxels, 1, 0)

    ps.init()
    for _ in range(10):
        randvox = np.random.randint(nhood.shape[0])

        voxijk = primal_voxels[randvox]
        nbrs = primal_voxels[nhood[randvox][nhood[randvox] >= 0]]
        print(nhood[randvox])
        nhood_ijk = torch.cat([voxijk.unsqueeze(0), nbrs], dim=0)

        vp, ve = fviz.grid_edge_network(index)

        vi, vei = fviz.grid_edge_network(Grid.from_ijk(nhood_ijk, voxel_size=vox_size, origin=vox_origin))

        ps.register_curve_network("vox", vp.cpu().numpy(), ve.cpu().numpy(), radius=0.0025)
        ps.register_curve_network("nhd", vi.cpu().numpy(), vei.cpu().numpy(), radius=0.005)
        ps.show()


if __name__ == "__main__":
    main()
