# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import torch

from .._fvdb_cpp import JaggedTensor
from ..grid import Grid
from ..grid_batch import GridBatch


def grid_edge_network(grid: Grid) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a set of line segments representing the edges of the active voxels in the grid.
    This can be useful for visualizing a :class:`~fvdb.Grid` as a wireframe.

    The line segments are represented by an ``(N, 3)`` tensor of vertices and an ``(M, 2)`` tensor of indices
    into the vertex tensor. such that each edge is defined by a pair of vertex indices, where
    ``edge_indices[j] = [v0, v1]`` means that the j-th edge connects vertices at positions
    ``edge_vertices[v0]`` and ``edge_vertices[v1]``.

    Example usage:

    .. code-block:: python

        import fvdb

        # Create a grid from points
        grid = fvdb.Grid.from_points(...)

        # Get the edge network of the grid, defining line segments for each edge of the active voxels
        edge_vertices, edge_indices = fvdb.viz.grid_edge_network(grid)

        # Get the start and end position of each edge
        v0 = edge_vertices[edge_indices[:, 0]] # Start position
        v1 = edge_vertices[edge_indices[:, 1]] # End position

    Args:
        grid (Grid): The :class:`~fvdb.Grid` to extract edges from.

    Returns:
        edge_vertices (torch.Tensor): A tensor of shape ``(N, 3)`` representing the vertices of the edges.
        edge_indices (torch.Tensor): A tensor of shape ``(M, 2)`` representing the indices of the
            vertices that form each edge. *i.e.*  ``edge_indices[j] = [v0, v1]`` means that the j-th edge
            connects vertices at positions ``edge_vertices[v0]`` and ``edge_vertices[v1]``.
    """
    gv, ge = grid._impl.viz_edge_network
    return gv.jdata, ge.jdata


def gridbatch_edge_network(grid: GridBatch) -> tuple[JaggedTensor, JaggedTensor]:
    """
    Return a set of line segments representing the edges of the active voxels in the grid batch.
    This can be useful for visualizing a :class:`~fvdb.GridBatch` as a wireframe.

    The line segments are represented by a jagged tensor of vertices and a jagged tensor of indices
    into the vertex tensor. such that each edge is defined by a pair of vertex indices, where
    ``edge_indices[b][j] = [v0, v1]`` means that the j-th edge in the b-th grid connects vertices at positions
    ``edge_vertices[b][v0]`` and ``edge_vertices[b][v1]``.

    Example usage:

    .. code-block:: python

        import fvdb

        # Create a grid batch from multiple grids
        grid_batch = fvdb.GridBatch.from_grids([...])

        # Get the edge network of the grid batch, defining line segments for each edge of the active voxels
        edge_vertices, edge_indices = fvdb.viz.gridbatch_edge_network(grid_batch)

        # Iterate over each grid in the batch, and get the start and end position of each edge
        for b in range(len(grid_batch)):
            # Get the start and end position of each edge in the b-th grid
            v0 = edge_vertices[b][edge_indices[b][:, 0]] # Start position
            v1 = edge_vertices[b][edge_indices[b][:, 1]] # End position

            # ... do something with v0 and v1 ...


    Args:
        grid (GridBatch): The :class:`~fvdb.GridBatch` to extract edges from with B grids.

    Returns:
        edge_vertices (JaggedTensor): A jagged tensor of shape ``(B, N_b, 3)`` representing the vertices of the edges.
        edge_indices (JaggedTensor): A jagged tensor of shape ``(B, M_b, 2)`` representing the indices of the
            vertices that form each edge. *i.e.*  ``edge_indices[b][j] = [v0, v1]`` means that the j-th edge in the b-th grid
            connects vertices at positions ``edge_vertices[b][v0]`` and ``edge_vertices[b][v1]``.
    """
    gv, ge = grid._impl.viz_edge_network
    return gv, ge
