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
        normals (JaggedTensor): Per-vertex normals.

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
        normals (torch.Tensor): Per-vertex normals.

    .. seealso:: :func:`marching_cubes_batch`
    """
    grid_data = grid.data
    field_jt = JaggedTensor(field)
    result = _fvdb_cpp.marching_cubes(grid_data, field_jt._impl, level)
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


def integrate_tsdf_frames_single(
    grid: Grid,
    truncation_distance: float,
    projection_matrices: torch.Tensor,
    cam_to_world_matrices: torch.Tensor,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    depth_images: torch.Tensor,
    weight_images: torch.Tensor | None = None,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """Integrate N depth frames into a single :class:`Grid` with one-shot topology.

    Semantically equivalent to calling :func:`integrate_tsdf_single` N
    times in sequence (verified bit-identically by
    ``test_integrate_tsdf_frames_matches_sequential``), but builds the
    union topology over all N frames ONCE up-front — avoiding the
    per-frame ``buildPointTruncationShell + mergeGrids`` cost that
    dominates per-frame wall-clock on small scenes.

    This is the fvdb analog of Open3D's lazy block-hashed allocation:
    "all frames known up-front, topology built once, fusion runs at
    fixed topology". For bulk / offline reality-capture reconstruction
    this is typically 3-5x faster than a per-frame loop.

    The N dimension is carried on ``depth_images.size(0)``. All per-frame
    tensors (``projection_matrices``, ``cam_to_world_matrices``,
    ``depth_images``, ``weight_images`` if given) must share that
    leading dimension.

    Args:
        grid (Grid): Single-scene grid with initial TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        projection_matrices (torch.Tensor): ``[N, 3, 3]`` per-frame intrinsics.
        cam_to_world_matrices (torch.Tensor): ``[N, 4, 4]`` per-frame poses.
        tsdf (torch.Tensor): Current TSDF values on ``grid``.
        weights (torch.Tensor): Current integration weights on ``grid``.
        depth_images (torch.Tensor): ``[N, H, W]`` or ``[N, H, W, 1]`` depth.
        weight_images (torch.Tensor | None): Optional ``[N, H, W]`` per-pixel weights.

    Returns:
        updated_grid (Grid): Union of ``grid`` and the truncation shell of all N frames.
        updated_tsdf (torch.Tensor): TSDF after integrating all N frames.
        updated_weights (torch.Tensor): Weights after integrating all N frames.

    .. seealso:: :func:`integrate_tsdf_frames_with_features_single`
    """
    from ..grid import Grid as G

    grid_data = grid.data
    tsdf_jt = JaggedTensor(tsdf)
    weights_jt = JaggedTensor(weights)
    rg, rt, rw = _fvdb_cpp.integrate_tsdf_batch(
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


def integrate_tsdf_frames_with_features_single(
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
    """N-frame batched integration with per-voxel features (e.g. RGB) for a :class:`Grid`.

    See :func:`integrate_tsdf_frames_single` for the core semantics.
    Feature dtype must match ``tsdf.dtype`` or be ``uint8``.
    """
    from ..grid import Grid as G

    grid_data = grid.data
    tsdf_jt = JaggedTensor(tsdf)
    weights_jt = JaggedTensor(weights)
    features_jt = JaggedTensor(features)
    rg, rt, rw, rf = _fvdb_cpp.integrate_tsdf_batch_with_features(
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


def integrate_tsdf_from_points_batch(
    grid: GridBatch,
    truncation_distance: float,
    points: JaggedTensor,
    sensor_origins: torch.Tensor,
    tsdf: JaggedTensor,
    weights: JaggedTensor,
    carve_free_space: bool = True,
) -> tuple[GridBatch, JaggedTensor, JaggedTensor]:
    """Integrate LiDAR / point-cloud sweeps into a TSDF volume for a grid batch.

    Each point is treated as a ray from ``sensor_origins[b]`` to the point
    endpoint; active voxels along the ray within the truncation band (and
    optionally the free-space band) are updated via weighted average. No
    range-image proxy is used — this is a native sparse ray-walk.

    Args:
        grid (GridBatch): The grid batch defining the TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        points (JaggedTensor): Per-batch LiDAR points, shape ``[B, N_i, 3]``.
        sensor_origins (torch.Tensor): ``[B, 3]`` per-batch sensor origin.
        tsdf (JaggedTensor): Current TSDF values.
        weights (JaggedTensor): Current integration weights.
        carve_free_space (bool): If ``True``, voxels observed as free space
            (in front of the endpoint, outside the truncation band) are
            written ``tsdf = +1``. Matches VDBFusion / nvblox default.

    Returns:
        updated_grid (GridBatch): The updated grid batch (union of input
            topology and the new point truncation shell).
        updated_tsdf (JaggedTensor): Updated TSDF values.
        updated_weights (JaggedTensor): Updated integration weights.

    .. seealso:: :func:`integrate_tsdf_from_points_single`
    """
    from ..grid_batch import GridBatch as GB

    grid_data = grid.data
    rg, rt, rw = _fvdb_cpp.integrate_tsdf_from_points(
        grid_data,
        truncation_distance,
        points._impl,
        sensor_origins,
        tsdf._impl,
        weights._impl,
        carve_free_space,
    )
    return GB(data=rg), JaggedTensor(impl=rt), JaggedTensor(impl=rw)


def integrate_tsdf_from_points_single(
    grid: Grid,
    truncation_distance: float,
    points: torch.Tensor,
    sensor_origin: torch.Tensor,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    carve_free_space: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """Integrate a single LiDAR / point-cloud sweep into a TSDF volume.

    See :func:`integrate_tsdf_from_points_batch` for semantics.

    Args:
        grid (Grid): The single grid defining the TSDF topology.
        truncation_distance (float): TSDF truncation distance.
        points (torch.Tensor): ``[N, 3]`` world-space point cloud.
        sensor_origin (torch.Tensor): ``[3]`` world-space sensor origin.
        tsdf (torch.Tensor): Current TSDF values.
        weights (torch.Tensor): Current integration weights.
        carve_free_space (bool): If ``True``, voxels observed as free
            space are written ``tsdf = +1``.

    Returns:
        updated_grid (Grid): The updated grid.
        updated_tsdf (torch.Tensor): Updated TSDF values.
        updated_weights (torch.Tensor): Updated integration weights.

    .. seealso:: :func:`integrate_tsdf_from_points_batch`
    """
    from ..grid import Grid as G

    grid_data = grid.data
    points_jt = JaggedTensor(points)
    tsdf_jt = JaggedTensor(tsdf)
    weights_jt = JaggedTensor(weights)
    rg, rt, rw = _fvdb_cpp.integrate_tsdf_from_points(
        grid_data,
        truncation_distance,
        points_jt._impl,
        sensor_origin.unsqueeze(0) if sensor_origin.dim() == 1 else sensor_origin,
        tsdf_jt._impl,
        weights_jt._impl,
        carve_free_space,
    )
    return G(data=rg), rt.jdata, rw.jdata


def integrate_tsdf_from_points_frames_single(
    grid: Grid,
    truncation_distance: float,
    points_per_frame: list[torch.Tensor],
    sensor_origins: torch.Tensor,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    carve_free_space: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """Integrate N LiDAR sweeps into a persistent TSDF volume in one C++ call.

    Semantically equivalent to:

    .. code-block:: python

        for i in range(N):
            grid, tsdf, weights = grid.integrate_tsdf_from_points(
                truncation_distance, points_per_frame[i],
                sensor_origins[i], tsdf, weights,
                carve_free_space=carve_free_space,
            )
        return grid, tsdf, weights

    but runs the whole loop inside C++ to eliminate the per-frame
    Python <-> C++ dispatch + JaggedTensor-rewrap overhead. Measured
    2-3x speedup on Mai City seq00 (700 frames @ 20 cm voxels,
    ~130 K pts/sweep) vs the Python-for-loop baseline.

    The output is bit-identical to the sequential reference:
    `test_integrate_tsdf_from_points_frames_matches_sequential`
    pins this with ``atol=rtol=0``.

    Args:
        grid (Grid): Initial grid (may be empty / seed).
        truncation_distance (float): TSDF truncation distance.
        points_per_frame (list[torch.Tensor]): Length-N list; each
            entry is ``[N_i, 3]`` world-frame points.
        sensor_origins (torch.Tensor): ``[N, 3]`` per-frame sensor
            origins in world frame.
        tsdf (torch.Tensor): ``[num_voxels]`` current TSDF values.
        weights (torch.Tensor): ``[num_voxels]`` current integration
            weights.
        carve_free_space (bool): Same semantics as the single-frame
            ``integrate_tsdf_from_points``.

    Returns:
        (updated_grid, updated_tsdf, updated_weights).
    """
    from ..grid import Grid as G

    grid_data = grid.data
    tsdf_jt = JaggedTensor(tsdf)
    weights_jt = JaggedTensor(weights)
    rg, rt, rw = _fvdb_cpp.integrate_tsdf_from_points_frames(
        grid_data,
        truncation_distance,
        list(points_per_frame),
        sensor_origins,
        tsdf_jt._impl,
        weights_jt._impl,
        carve_free_space,
    )
    return G(data=rg), rt.jdata, rw.jdata


def integrate_tsdf_from_points_with_features_batch(
    grid: GridBatch,
    truncation_distance: float,
    points: JaggedTensor,
    sensor_origins: torch.Tensor,
    tsdf: JaggedTensor,
    features: JaggedTensor,
    weights: JaggedTensor,
    point_features: JaggedTensor,
    carve_free_space: bool = True,
) -> tuple[GridBatch, JaggedTensor, JaggedTensor, JaggedTensor]:
    """Integrate point clouds with per-point features into a TSDF volume for a grid batch.

    Features are blended into per-voxel features with the same weighted-
    average formula used by :func:`integrate_tsdf_with_features_batch`.
    Feature dtype must match ``tsdf.dtype`` or be ``uint8`` (for RGB).

    .. seealso:: :func:`integrate_tsdf_from_points_with_features_single`
    """
    from ..grid_batch import GridBatch as GB

    grid_data = grid.data
    rg, rt, rw, rf = _fvdb_cpp.integrate_tsdf_from_points_with_features(
        grid_data,
        truncation_distance,
        points._impl,
        sensor_origins,
        tsdf._impl,
        features._impl,
        weights._impl,
        point_features._impl,
        carve_free_space,
    )
    return GB(data=rg), JaggedTensor(impl=rt), JaggedTensor(impl=rw), JaggedTensor(impl=rf)


def integrate_occupancy_from_points_single(
    grid: Grid,
    truncation_distance: float,
    points: torch.Tensor,
    sensor_origin: torch.Tensor,
    log_odds: torch.Tensor,
    log_odds_hit: float = 0.85,
    log_odds_miss: float = -0.40,
    log_odds_min: float = -4.0,
    log_odds_max: float = 4.0,
) -> tuple[Grid, torch.Tensor]:
    """Integrate a single LiDAR / point-cloud sweep into a Bayesian
    log-odds occupancy volume.

    Sister primitive to :func:`integrate_tsdf_from_points_single`:
    same shell allocator, same HDDA ray-walk, but writes log-odds
    increments (``+log_odds_hit`` for near-endpoint voxels,
    ``log_odds_miss`` for sensor-side voxels in the walk band) and
    clamps the accumulated value to ``[log_odds_min, log_odds_max]``.

    The stored sidecar IS the log-odds. To recover probability on
    the host: ``p = torch.sigmoid(log_odds)``.

    Defaults match nvblox's `ProjectiveIntegratorType.OCCUPANCY`
    defaults (hit +0.85, miss -0.40, clamp [-4, +4]).

    Args:
        grid: Input grid (topology grows via the point-shell union).
        truncation_distance: Width of the hit band around each point
            endpoint, and the shell-allocator dilation distance.
        points: ``[N, 3]`` world-frame point cloud.
        sensor_origin: ``[3]`` or ``[1, 3]`` world-frame sensor origin.
        log_odds: ``[num_voxels]`` current log-odds sidecar.
        log_odds_hit: Increment per hit observation.
        log_odds_miss: Increment per miss observation (negative).
        log_odds_min: Lower clamp bound.
        log_odds_max: Upper clamp bound.

    Returns:
        updated_grid: Union of ``grid`` and the new point shell.
        updated_log_odds: Log-odds sidecar on the updated grid.
    """
    from ..grid import Grid as G

    grid_data = grid.data
    points_jt = JaggedTensor(points)
    log_odds_jt = JaggedTensor(log_odds)
    rg, rlo = _fvdb_cpp.integrate_occupancy_from_points(
        grid_data,
        float(truncation_distance),
        points_jt._impl,
        sensor_origin.unsqueeze(0) if sensor_origin.dim() == 1 else sensor_origin,
        log_odds_jt._impl,
        float(log_odds_hit),
        float(log_odds_miss),
        float(log_odds_min),
        float(log_odds_max),
    )
    return G(data=rg), rlo.jdata


def integrate_occupancy_from_points_frames_single(
    grid: Grid,
    truncation_distance: float,
    points_per_frame: list[torch.Tensor],
    sensor_origins: torch.Tensor,
    log_odds: torch.Tensor,
    log_odds_hit: float = 0.85,
    log_odds_miss: float = -0.40,
    log_odds_min: float = -4.0,
    log_odds_max: float = 4.0,
) -> tuple[Grid, torch.Tensor]:
    """Integrate N LiDAR sweeps into a persistent log-odds occupancy
    volume in one C++ call.

    Semantically equivalent to calling
    :func:`integrate_occupancy_from_points_single` N times in
    sequence, but amortises the per-frame Python <-> C++ dispatch
    overhead. Mirrors the `integrate_tsdf_from_points_frames`
    batched API one-for-one.

    See :func:`integrate_occupancy_from_points_single` for argument
    semantics and default values.
    """
    from ..grid import Grid as G

    grid_data = grid.data
    log_odds_jt = JaggedTensor(log_odds)
    rg, rlo = _fvdb_cpp.integrate_occupancy_from_points_frames(
        grid_data,
        float(truncation_distance),
        list(points_per_frame),
        sensor_origins,
        log_odds_jt._impl,
        float(log_odds_hit),
        float(log_odds_miss),
        float(log_odds_min),
        float(log_odds_max),
    )
    return G(data=rg), rlo.jdata


def compute_esdf_single(
    grid: Grid,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    truncation_distance: float,
    max_distance: float,
    weight_threshold: float = 1.0e-6,
    prune_unreached: bool = False,
    use_vbm: bool = True,
) -> tuple[Grid, torch.Tensor]:
    """Compute a Euclidean Signed Distance Field (ESDF) from an integrated TSDF.

    Extends the narrow-band signed distances stored in ``tsdf`` outward
    (and inward) across a wider support band, producing per-voxel world-
    unit signed distances with ``|d| <= max_distance``. The returned
    :class:`Grid` is the input topology dilated by
    ``ceil(max_distance / voxel_size) + 1`` voxels (unless
    ``prune_unreached=True``, in which case the unreached frontier is
    dropped).

    This is the **second application** of the nanoVDB topology-op
    vocabulary in this campaign (the first being depth/LiDAR TSDF). The
    algorithm composes three primitives:

    * :meth:`Grid.dilated_grid` — allocates the ESDF support band.
    * A custom VBM-stencil kernel (26-neighbour monotone min) — does
      the wavefront propagation.
    * :meth:`Grid.pruned_grid` — optional, drops unreached voxels.

    Scope: float32 CUDA + single grid only.

    TSDF convention: the ``tsdf`` tensor is assumed to follow fvdb's
    ``integrate_tsdf`` convention of ``tsdf = clip(d_world / T, -1, +1)``
    where ``T = truncation_distance``. The returned ESDF is in world
    units (i.e., the same units as ``truncation_distance`` and
    ``max_distance``).

    Args:
        grid: Input TSDF grid topology.
        tsdf: ``[num_voxels]`` fp32 normalized TSDF in ``[-1, +1]``.
        weights: ``[num_voxels]`` fp32 integration weights.
        truncation_distance: TSDF truncation margin in world units.
        max_distance: ESDF support radius in world units.
        weight_threshold: Voxels with ``weights <= threshold`` are not
            used as wavefront sources. Default ``1e-6``.
        prune_unreached: If ``True``, drop voxels the wavefront never
            reached (still at distance ``max_distance`` sentinel).
            Default ``False``: return the full dilated support with
            unreached voxels clamped to ``max_distance``.
        use_vbm: Use :class:`VoxelBlockManager`-based sweep kernel (the
            default) versus per-leaf-slot iteration (ablation). Output
            is bit-identical.

    Returns:
        esdf_grid: New :class:`Grid` for the ESDF support band.
        esdf: ``[esdf_grid.num_voxels]`` fp32 world-unit signed distance.
    """
    from ..grid import Grid as G

    grid_data = grid.data
    out_grid, out_esdf = _fvdb_cpp.compute_esdf(
        grid_data,
        tsdf,
        weights,
        float(truncation_distance),
        float(max_distance),
        float(weight_threshold),
        bool(prune_unreached),
        bool(use_vbm),
    )
    return G(data=out_grid), out_esdf


def compute_esdf_incremental_single(
    grid: Grid,
    tsdf: torch.Tensor,
    weights: torch.Tensor,
    prev_esdf_grid: Grid,
    prev_esdf: torch.Tensor,
    truncation_distance: float,
    max_distance: float,
    weight_threshold: float = 1.0e-6,
    prune_unreached: bool = False,
    use_vbm: bool = True,
    dirty_mask: torch.Tensor | None = None,
) -> tuple[Grid, torch.Tensor]:
    """Monotone-incremental ESDF: warm-start from a previous ESDF.

    Same algorithm as :func:`compute_esdf_single` but takes a
    ``(prev_esdf_grid, prev_esdf)`` pair that was returned from a
    previous call (either this function or :func:`compute_esdf_single`).
    The resulting grid is the merge of ``dilate(grid, K) ∪ prev_esdf_grid``,
    so voxels that were in the previous support band but not in the
    current support band are preserved. Previous ESDF values are
    injected into the new sidecar before the wavefront sweep, giving
    the monotone-min kernel a warm start.

    **Monotone-only assumption**: this function is correct when distances
    can only decrease between frames (new surfaces added; existing
    surfaces refined but not removed). For scenes with dynamic objects
    or noise-resolved phantom surfaces, call :func:`compute_esdf_single`
    periodically as a global correction pass.

    When ``prev_esdf_grid`` is empty (e.g. first frame of a session),
    this falls through to :func:`compute_esdf_single` semantics.

    Args:
        grid: Current TSDF grid.
        tsdf: ``[num_voxels]`` fp32 normalized TSDF in ``[-1, +1]``.
        weights: ``[num_voxels]`` fp32 integration weights.
        prev_esdf_grid: Previous frame's ESDF :class:`Grid`.
        prev_esdf: Previous frame's ``[prev_esdf_grid.num_voxels]`` fp32
            signed distance sidecar.
        truncation_distance: TSDF truncation margin (world units).
        max_distance: ESDF support radius (world units).
        weight_threshold: Voxels with ``weights <= weight_threshold``
            are not used as wavefront sources.
        prune_unreached: If ``True``, drop voxels the wavefront never
            reached.
        use_vbm: Use :class:`VoxelBlockManager`-based sweep kernel.

    Args (continued):
        dirty_mask (torch.Tensor | None): Optional ``[grid.num_voxels]``
            bool tensor marking which voxels' TSDF changed this frame.
            When provided, only dirty voxels seed the ESDF wavefront;
            the rest inherit the previous frame's values unchanged.
            This is the mechanism that makes ``compute_esdf_incremental``
            scale with the dirty-region size rather than with the full
            grid (matching nvblox's block-dirty-tracking behaviour).
            When ``dirty_mask.any() == False`` AND ``prev_esdf_grid``
            is non-empty, the call short-circuits in Python and
            returns ``(prev_esdf_grid, prev_esdf)`` directly without
            entering C++ -- this is the "static TSDF cache hit" path
            that matches nvblox's ~50 us warm-reuse cost.
            Produce the mask via
            :func:`fvdb.functional.dirty_mask_from_sidecars_single`
            (``(new_grid, new_weights, old_grid, old_weights)``) or
            with any user-authored predicate. Default ``None`` =
            full-recompute (original semantics).

    Returns:
        esdf_grid: New :class:`Grid` (merge of dilated support +
            previous ESDF support).
        esdf: ``[esdf_grid.num_voxels]`` fp32 signed distance.
    """
    from ..grid import Grid as G

    # Python-level short-circuit: if the caller provided a dirty mask
    # that is entirely false AND we have a previous ESDF state, we
    # know the monotone-min result is unchanged and return immediately.
    # Costs one host-side `.any().item()` sync (~30 us) and never
    # enters C++. This is the "cache hit" equivalent of nvblox's
    # dirty-block short-circuit -- but expressed at the Python layer
    # against a user-held tensor, not hidden allocator state.
    if dirty_mask is not None and prev_esdf_grid.num_voxels > 0:
        if not dirty_mask.any().item():
            return prev_esdf_grid, prev_esdf

    grid_data = grid.data
    prev_grid_data = prev_esdf_grid.data
    # C++ accepts `dirty_mask` as a possibly-undefined tensor; pass an
    # empty tensor to signal "no dirty mask" (pybind then sees an
    # undefined Tensor which the C++ side interprets via `.defined()`).
    if dirty_mask is None:
        dm_arg = torch.empty(0, device=tsdf.device, dtype=torch.bool)
    else:
        dm_arg = dirty_mask
    out_grid, out_esdf = _fvdb_cpp.compute_esdf_incremental(
        grid_data,
        tsdf,
        weights,
        prev_grid_data,
        prev_esdf,
        float(truncation_distance),
        float(max_distance),
        float(weight_threshold),
        bool(prune_unreached),
        bool(use_vbm),
        dm_arg,
    )
    return G(data=out_grid), out_esdf


def integrate_tsdf_from_points_with_features_single(
    grid: Grid,
    truncation_distance: float,
    points: torch.Tensor,
    sensor_origin: torch.Tensor,
    tsdf: torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    point_features: torch.Tensor,
    carve_free_space: bool = True,
) -> tuple[Grid, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a single point cloud with per-point features into a TSDF volume.

    .. seealso:: :func:`integrate_tsdf_from_points_with_features_batch`
    """
    from ..grid import Grid as G

    grid_data = grid.data
    points_jt = JaggedTensor(points)
    tsdf_jt = JaggedTensor(tsdf)
    features_jt = JaggedTensor(features)
    weights_jt = JaggedTensor(weights)
    point_features_jt = JaggedTensor(point_features)
    rg, rt, rw, rf = _fvdb_cpp.integrate_tsdf_from_points_with_features(
        grid_data,
        truncation_distance,
        points_jt._impl,
        sensor_origin.unsqueeze(0) if sensor_origin.dim() == 1 else sensor_origin,
        tsdf_jt._impl,
        features_jt._impl,
        weights_jt._impl,
        point_features_jt._impl,
        carve_free_space,
    )
    return G(data=rg), rt.jdata, rw.jdata, rf.jdata
