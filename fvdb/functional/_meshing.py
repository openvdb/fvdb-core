# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for meshing and TSDF integration on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


@overload
def marching_cubes(
    grid: Grid, field: torch.Tensor, level: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def marching_cubes(
    grid: GridBatch, field: JaggedTensor, level: float = 0.0
) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]: ...


def marching_cubes(
    grid: Grid | GridBatch,
    field: torch.Tensor | JaggedTensor,
    level: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
    """
    Extract isosurface meshes using the marching cubes algorithm.

    Generates triangle meshes representing the isosurface at ``level`` from a
    scalar field defined on the voxels.

    Args:
        grid: The grid structure.
        field: Scalar field values at each voxel.
        level: Isovalue at which to extract the surface. Default ``0.0``.

    Returns:
        A tuple ``(vertices, face_indices, vertex_normals)``.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_f = JaggedTensor(field)
        v, f, n = grid._impl.marching_cubes(jt_f._impl, level)
        return v.jdata, f.jdata, n.jdata
    v, f, n = grid._impl.marching_cubes(field._impl, level)
    return JaggedTensor(impl=v), JaggedTensor(impl=f), JaggedTensor(impl=n)


@overload
def integrate_tsdf(
    grid: Grid,
    truncation_distance: float,
    projection_matrix: torch.Tensor,
    cam_to_world_matrix: torch.Tensor,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    depth_image: torch.Tensor,
    weight_image: torch.Tensor | None = None,
) -> tuple[Grid, torch.Tensor, torch.Tensor]: ...


@overload
def integrate_tsdf(
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: JaggedTensor,
    weights: JaggedTensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[GridBatch, JaggedTensor, JaggedTensor]: ...


def integrate_tsdf(
    grid,
    truncation_distance,
    projection_matrix_or_matrices,
    cam_to_world_matrix_or_matrices,
    tsdf,
    weights,
    depth_image_or_images,
    weight_image_or_images=None,
):
    """
    Integrate depth images into a TSDF volume.

    Updates TSDF values and weights by integrating depth observations from camera
    viewpoints. For :class:`~fvdb.Grid`, inputs are single-view tensors; for
    :class:`~fvdb.GridBatch`, they are batched.

    Args:
        grid: The grid structure.
        truncation_distance: Maximum TSDF truncation distance.
        projection_matrix_or_matrices: Camera projection matrix(es).
        cam_to_world_matrix_or_matrices: Camera-to-world transform(s).
        tsdf: Current TSDF values.
        weights: Current integration weights.
        depth_image_or_images: Depth image(s).
        weight_image_or_images: Optional per-pixel weight image(s).

    Returns:
        A tuple ``(updated_grid, updated_tsdf, updated_weights)``.
    """
    from ..grid import Grid

    if isinstance(grid, Grid):
        jt_tsdf = JaggedTensor(tsdf)
        jt_w = JaggedTensor(weights)
        rg, rt, rw = grid._impl.integrate_tsdf(
            truncation_distance,
            projection_matrix_or_matrices.unsqueeze(0),
            cam_to_world_matrix_or_matrices.unsqueeze(0),
            jt_tsdf._impl,
            jt_w._impl,
            depth_image_or_images.unsqueeze(0),
            weight_image_or_images.unsqueeze(0) if weight_image_or_images is not None else None,
        )
        return Grid(impl=rg), rt.jdata, rw.jdata

    from ..grid_batch import GridBatch as GB

    rg, rt, rw = grid._impl.integrate_tsdf(
        truncation_distance,
        projection_matrix_or_matrices,
        cam_to_world_matrix_or_matrices,
        tsdf._impl,
        weights._impl,
        depth_image_or_images,
        weight_image_or_images,
    )
    return GB(impl=rg), JaggedTensor(impl=rt), JaggedTensor(impl=rw)
