# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for meshing and TSDF integration on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ._dispatch import _get_grid_data, _prepare_args

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
    is_flat = isinstance(field, torch.Tensor)
    grid_data, (field_jt,), unwrap = _prepare_args(grid, field)
    assert field_jt is not None
    result = _fvdb_cpp.marching_cubes(grid_data, field_jt._impl, level)
    if is_flat:
        return result[0].jdata, result[1].jdata, result[2].jdata
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1]), JaggedTensor(impl=result[2])


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

    is_flat = isinstance(tsdf, torch.Tensor)
    grid_data, (tsdf_jt, weights_jt), unwrap = _prepare_args(grid, tsdf, weights)
    assert tsdf_jt is not None and weights_jt is not None
    rg, rt, rw = _fvdb_cpp.integrate_tsdf(
        grid_data,
        truncation_distance,
        projection_matrices,
        cam_to_world_matrices,
        tsdf_jt._impl,
        weights_jt._impl,
        depth_images,
        weight_images,
    )
    new_grid = GB(data=rg)
    if is_flat:
        return new_grid, rt.jdata, rw.jdata
    return new_grid, JaggedTensor(impl=rt), JaggedTensor(impl=rw)


@overload
def integrate_tsdf_with_features(
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    depth_images: torch.Tensor,
    feature_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[GridBatch, torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def integrate_tsdf_with_features(
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: JaggedTensor,
    features: JaggedTensor,
    weights: JaggedTensor,
    depth_images: torch.Tensor,
    feature_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[GridBatch, JaggedTensor, JaggedTensor, JaggedTensor]: ...


def integrate_tsdf_with_features(
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor | JaggedTensor,
    features: torch.Tensor | JaggedTensor,
    weights: torch.Tensor | JaggedTensor,
    depth_images: torch.Tensor,
    feature_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> (
    tuple[GridBatch, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[GridBatch, JaggedTensor, JaggedTensor, JaggedTensor]
):
    """
    Integrate depth and feature images into a TSDF volume with features.

    Similar to :func:`integrate_tsdf` but also integrates feature observations
    (e.g., color) along with the depth information.

    Args:
        grid: The grid structure.
        truncation_distance: Maximum TSDF truncation distance.
        projection_matrices: Camera projection matrices, shape ``(B, 3, 3)``.
        cam_to_world_matrices: Camera-to-world transforms, shape ``(B, 4, 4)``.
        tsdf: Current TSDF values.
        features: Current feature values at each voxel.
        weights: Current integration weights.
        depth_images: Depth images, shape ``(B, H, W)``.
        feature_images: Feature images (e.g., RGB), shape ``(B, H, W, C)``.
        weight_images: Optional per-pixel weight images.

    Returns:
        A tuple ``(updated_grid, updated_tsdf, updated_weights, updated_features)``.
    """
    from ..grid_batch import GridBatch as GB

    is_flat = isinstance(tsdf, torch.Tensor)
    grid_data, (tsdf_jt, features_jt, weights_jt), unwrap = _prepare_args(grid, tsdf, features, weights)
    assert tsdf_jt is not None and features_jt is not None and weights_jt is not None
    rg, rt, rw, rf = _fvdb_cpp.integrate_tsdf_with_features(
        grid_data,
        truncation_distance,
        projection_matrices,
        cam_to_world_matrices,
        tsdf_jt._impl,
        features_jt._impl,
        weights_jt._impl,
        depth_images,
        feature_images,
        weight_images,
    )
    new_grid = GB(data=rg)
    if is_flat:
        return new_grid, rt.jdata, rw.jdata, rf.jdata
    return new_grid, JaggedTensor(impl=rt), JaggedTensor(impl=rw), JaggedTensor(impl=rf)
