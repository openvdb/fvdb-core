# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for meshing and TSDF integration on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ._dispatch import _prepare_args

if TYPE_CHECKING:
    from ..grid_batch import GridBatch


@overload
def marching_cubes(
    grid: GridBatch, field: torch.Tensor, level: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def marching_cubes(
    grid: GridBatch, field: JaggedTensor, level: float = 0.0
) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]: ...


def marching_cubes(
    grid: GridBatch,
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
    grid_data, (field,), unwrap = _prepare_args(grid, field)
    result = _fvdb_cpp.marching_cubes(grid_data, field._impl, level)
    return unwrap(result[0].jdata), unwrap(result[1].jdata), unwrap(result[2].jdata)


@overload
def integrate_tsdf(
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[GridBatch, torch.Tensor, torch.Tensor]: ...


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
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor | JaggedTensor,
    weights: torch.Tensor | JaggedTensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[GridBatch, torch.Tensor, torch.Tensor] | tuple[GridBatch, JaggedTensor, JaggedTensor]:
    """
    Integrate depth images into a TSDF volume.

    Updates TSDF values and weights by integrating depth observations from camera
    viewpoints.

    Args:
        grid: The grid structure.
        truncation_distance: Maximum TSDF truncation distance.
        projection_matrices: Camera projection matrices, shape ``(B, 4, 4)``.
        cam_to_world_matrices: Camera-to-world transforms, shape ``(B, 4, 4)``.
        tsdf: Current TSDF values.
        weights: Current integration weights.
        depth_images: Depth images, shape ``(B, H, W)``.
        weight_images: Optional per-pixel weight images.

    Returns:
        A tuple ``(updated_grid, updated_tsdf, updated_weights)``.
    """
    from ..grid_batch import GridBatch as GB

    grid_data, (tsdf, weights), unwrap = _prepare_args(grid, tsdf, weights)
    rg, rt, rw = _fvdb_cpp.integrate_tsdf(
        grid_data,
        truncation_distance,
        projection_matrices,
        cam_to_world_matrices,
        tsdf._impl,
        weights._impl,
        depth_images,
        weight_images,
    )
    return GB(data=rg), unwrap(rt.jdata), unwrap(rw.jdata)
