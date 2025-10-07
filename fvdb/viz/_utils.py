# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import torch

from .._Cpp import JaggedTensor
from .._grid import Grid
from .._grid_batch import GridBatch


def grid_edge_network(grid: Grid) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a set of line segments representing the edges of the active voxels in the grid.

    Args:
        grid (Grid): The grid to extract edges from.

    Returns:
        edge_vertices (torch.Tensor): A tensor of shape (N, 3) representing the vertices of the edges.
        edge_indices (torch.Tensor): A tensor of shape (M, 2) representing the indices of the
            vertices that form each edge. _i.e_ edge_indices[j] = [v0, v1] means that the j-th edge
            connects vertices at positions edge_vertices[v0] and edge_vertices[v1].
    """
    gv, ge = grid._impl.viz_edge_network
    return gv.jdata, ge.jdata


def gridbatch_edge_network(grid: GridBatch) -> tuple[JaggedTensor, JaggedTensor]:
    """
    Return a set of line segments representing the edges of the active voxels in the grid batch.

    Args:
        grid (GridBatch): The grid batch to extract edges from with B grids.

    Returns:
        edge_vertices (JaggedTensor): A jagged tensor of shape (B, N_b, 3) representing the vertices of the edges.
        edge_indices (JaggedTensor): A jagged tensor of shape (B, M_b, 2) representing the indices of the
            vertices that form each edge. _i.e_ edge_indices[b][j] = [v0, v1] means that the j-th edge in the b-th grid
            connects vertices at positions edge_vertices[b][v0] and edge_vertices[b][v1].
    """
    gv, ge = grid._impl.viz_edge_network
    return gv, ge
