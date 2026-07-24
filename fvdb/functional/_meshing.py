# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for meshing and TSDF integration on sparse grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Batch API  (GridBatch + JaggedTensor)
# ---------------------------------------------------------------------------


def marching_cubes_batch(
    grid: GridBatch,
    field: JaggedTensor,
    level: float = 0.0,
) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
    """Extract isosurface meshes using marching cubes on a grid batch.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        field (JaggedTensor): Per-voxel scalar field values.
        level (float): Isovalue at which to extract the surface. Default ``0.0``.

    Returns:
        vertices (JaggedTensor): Mesh vertex positions, shape ``(B, -1, 3)``.
        faces (JaggedTensor): Triangle face indices.
        vertex_edge_keys (JaggedTensor): Edge keys used to deduplicate mesh
            vertices, dtype ``torch.int64``, shape ``(B, -1, 3)``. Each row
            ``[batch_idx, voxel_a, voxel_b]`` identifies the grid edge on which
            the vertex was interpolated, where ``voxel_a`` and ``voxel_b`` are
            flat voxel indices of the two endpoint voxels (``voxel_a >=
            voxel_b``). This can be used to map mesh vertices back to the grid
            edges and voxels they originated from.

    .. seealso:: :func:`marching_cubes_single`
    """
    grid_data = grid.data
    result = _fvdb_cpp.marching_cubes(grid_data, field._impl, level)
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1]), JaggedTensor(impl=result[2])


def marching_cubes_single(
    grid: Grid,
    field: torch.Tensor,
    level: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract isosurface mesh using marching cubes on a single grid.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        field (torch.Tensor): Per-voxel scalar field values.
        level (float): Isovalue at which to extract the surface. Default ``0.0``.

    Returns:
        vertices (torch.Tensor): Mesh vertex positions, shape ``(N, 3)``.
        faces (torch.Tensor): Triangle face indices.
        vertex_edge_keys (torch.Tensor): Edge keys used to deduplicate mesh
            vertices, dtype ``torch.int64``, shape ``(N, 3)``. Each row
            ``[batch_idx, voxel_a, voxel_b]`` identifies the grid edge on which
            the vertex was interpolated, where ``voxel_a`` and ``voxel_b`` are
            flat voxel indices of the two endpoint voxels (``voxel_a >=
            voxel_b``). This can be used to map mesh vertices back to the grid
            edges and voxels they originated from. ``batch_idx`` is always
            ``0`` for a single grid.

    .. seealso:: :func:`marching_cubes_batch`
    """
    grid_data = grid.data
    field_jt = JaggedTensor(field)
    result = _fvdb_cpp.marching_cubes(grid_data, field_jt._impl, level)
    return result[0].jdata, result[1].jdata, result[2].jdata


def dual_contour_batch(
    grid: GridBatch,
    field: JaggedTensor,
    iso: float = 0.0,
    reduce: int = 1,
    adaptivity: float = 0.0,
) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
    """Extract a dual-contouring (DC + QEF) mesh from an SDF on a grid batch.

    Places one vertex per surface cell by minimizing a quadratic error function from the cell's edge
    zero-crossings and SDF-gradient normals (sharp-feature preserving), with optional cluster-collapse
    decimation. Requires a narrow-band SDF with a >= ~3-voxel band for a watertight result (e.g. from
    :func:`retopologize_sdf_batch`).

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        field (JaggedTensor): Per-voxel signed distance values.
        iso (float): Isovalue at which to extract the surface. Default ``0.0``.
        reduce (int): Uniform ``F x F x F`` cluster-collapse decimation block size. ``reduce=1`` is full detail only when ``adaptivity == 0``; when ``adaptivity > 0`` it instead sets the coarse-block size for the adaptive collapse (default ``8`` when ``reduce <= 1``).
        adaptivity (float): Curvature-adaptive decimation in ``[0, 1.5]`` (``0`` = off). When ``> 0``, flat blocks are collapsed while features keep full detail, so the mesh is simplified even at ``reduce=1``.

    Returns:
        vertices (JaggedTensor): Mesh vertex positions, shape ``(B, -1, 3)``.
        faces (JaggedTensor): Triangle face indices, shape ``(B, -1, 3)``.
        normals (JaggedTensor): Per-vertex (normalized SDF gradient) normals, shape ``(B, -1, 3)``.

    .. seealso:: :func:`dual_contour_single`, :func:`marching_cubes_batch`
    """
    grid_data = grid.data
    result = _fvdb_cpp.dual_contour(grid_data, field._impl, iso, reduce, adaptivity)
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1]), JaggedTensor(impl=result[2])


def dual_contour_single(
    grid: Grid,
    field: torch.Tensor,
    iso: float = 0.0,
    reduce: int = 1,
    adaptivity: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract a dual-contouring (DC + QEF) mesh from an SDF on a single grid.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        field (torch.Tensor): Per-voxel signed distance values.
        iso (float): Isovalue at which to extract the surface. Default ``0.0``.
        reduce (int): Uniform ``F x F x F`` cluster-collapse decimation block size. ``reduce=1`` is full detail only when ``adaptivity == 0``; when ``adaptivity > 0`` it instead sets the coarse-block size for the adaptive collapse (default ``8`` when ``reduce <= 1``).
        adaptivity (float): Curvature-adaptive decimation in ``[0, 1.5]`` (``0`` = off). When ``> 0``, flat blocks are collapsed while features keep full detail, so the mesh is simplified even at ``reduce=1``.

    Returns:
        vertices (torch.Tensor): Vertex positions, shape ``(V, 3)``.
        faces (torch.Tensor): Triangle face indices, shape ``(T, 3)``.
        normals (torch.Tensor): Per-vertex (normalized SDF gradient) normals, shape ``(V, 3)``.

    .. seealso:: :func:`dual_contour_batch`, :func:`marching_cubes_single`
    """
    grid_data = grid.data
    field_jt = JaggedTensor(field)
    result = _fvdb_cpp.dual_contour(grid_data, field_jt._impl, iso, reduce, adaptivity)
    return result[0].jdata, result[1].jdata, result[2].jdata


def integrate_tsdf_batch(
    grid: GridBatch,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: JaggedTensor,
    weights: JaggedTensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[GridBatch, JaggedTensor, JaggedTensor]:
    """Integrate depth images into a TSDF volume for a grid batch.

    Args:
        grid (GridBatch): The grid batch defining the TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        projection_matrices (torch.Tensor): Camera projection matrices.
        cam_to_world_matrices (torch.Tensor): Camera-to-world transform matrices.
        tsdf (JaggedTensor): Current TSDF values.
        weights (JaggedTensor): Current integration weights.
        depth_images (torch.Tensor): Depth images to integrate.
        weight_images (torch.Tensor | None): Optional per-pixel weight images.

    Returns:
        updated_grid (GridBatch): The updated grid batch.
        updated_tsdf (JaggedTensor): Updated TSDF values.
        updated_weights (JaggedTensor): Updated integration weights.

    .. seealso:: :func:`integrate_tsdf_single`
    """
    from ..grid_batch import GridBatch as GB

    grid_data = grid.data
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
    return GB(data=rg), JaggedTensor(impl=rt), JaggedTensor(impl=rw)


def integrate_tsdf_single(
    grid: Grid,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """Integrate depth images into a TSDF volume for a single grid.

    Args:
        grid (Grid): The single grid defining the TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        projection_matrices (torch.Tensor): Camera projection matrices.
        cam_to_world_matrices (torch.Tensor): Camera-to-world transform matrices.
        tsdf (torch.Tensor): Current TSDF values.
        weights (torch.Tensor): Current integration weights.
        depth_images (torch.Tensor): Depth images to integrate.
        weight_images (torch.Tensor | None): Optional per-pixel weight images.

    Returns:
        updated_grid (Grid): The updated grid.
        updated_tsdf (torch.Tensor): Updated TSDF values.
        updated_weights (torch.Tensor): Updated integration weights.

    .. seealso:: :func:`integrate_tsdf_batch`
    """
    from ..grid import Grid as G

    grid_data = grid.data
    tsdf_jt = JaggedTensor(tsdf)
    weights_jt = JaggedTensor(weights)
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
    return G(data=rg), rt.jdata, rw.jdata


def integrate_tsdf_with_features_batch(
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
) -> tuple[GridBatch, JaggedTensor, JaggedTensor, JaggedTensor]:
    """Integrate depth and feature images into a TSDF volume with features for a grid batch.

    Args:
        grid (GridBatch): The grid batch defining the TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        projection_matrices (torch.Tensor): Camera projection matrices.
        cam_to_world_matrices (torch.Tensor): Camera-to-world transform matrices.
        tsdf (JaggedTensor): Current TSDF values.
        features (JaggedTensor): Current per-voxel features.
        weights (JaggedTensor): Current integration weights.
        depth_images (torch.Tensor): Depth images to integrate.
        feature_images (torch.Tensor): Feature images to integrate.
        weight_images (torch.Tensor | None): Optional per-pixel weight images.

    Returns:
        updated_grid (GridBatch): The updated grid batch.
        updated_tsdf (JaggedTensor): Updated TSDF values.
        updated_weights (JaggedTensor): Updated integration weights.
        updated_features (JaggedTensor): Updated per-voxel features.

    .. seealso:: :func:`integrate_tsdf_with_features_single`
    """
    from ..grid_batch import GridBatch as GB

    grid_data = grid.data
    rg, rt, rw, rf = _fvdb_cpp.integrate_tsdf_with_features(
        grid_data,
        truncation_distance,
        projection_matrices,
        cam_to_world_matrices,
        tsdf._impl,
        features._impl,
        weights._impl,
        depth_images,
        feature_images,
        weight_images,
    )
    return GB(data=rg), JaggedTensor(impl=rt), JaggedTensor(impl=rw), JaggedTensor(impl=rf)


def integrate_tsdf_with_features_single(
    grid: Grid,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    depth_images: torch.Tensor,
    feature_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[Grid, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate depth and feature images into a TSDF volume with features for a single grid.

    Args:
        grid (Grid): The single grid defining the TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        projection_matrices (torch.Tensor): Camera projection matrices.
        cam_to_world_matrices (torch.Tensor): Camera-to-world transform matrices.
        tsdf (torch.Tensor): Current TSDF values.
        features (torch.Tensor): Current per-voxel features.
        weights (torch.Tensor): Current integration weights.
        depth_images (torch.Tensor): Depth images to integrate.
        feature_images (torch.Tensor): Feature images to integrate.
        weight_images (torch.Tensor | None): Optional per-pixel weight images.

    Returns:
        updated_grid (Grid): The updated grid.
        updated_tsdf (torch.Tensor): Updated TSDF values.
        updated_weights (torch.Tensor): Updated integration weights.
        updated_features (torch.Tensor): Updated per-voxel features.

    .. seealso:: :func:`integrate_tsdf_with_features_batch`
    """
    from ..grid import Grid as G

    grid_data = grid.data
    tsdf_jt = JaggedTensor(tsdf)
    features_jt = JaggedTensor(features)
    weights_jt = JaggedTensor(weights)
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
    return G(data=rg), rt.jdata, rw.jdata, rf.jdata
