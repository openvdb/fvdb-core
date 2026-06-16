# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for signed-distance-field (SDF) re-initialization and narrow-band retopologization."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp
from ..enums import SmoothingMode
from ..jagged_tensor import JaggedTensor
from ._dense import inject_batch, inject_single

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


def _to_cpp_smoothing(smoothing: SmoothingMode) -> "_fvdb_cpp.SmoothingMode":
    """Convert a public :class:`fvdb.SmoothingMode` to the bound C++ enum (matched by member name)."""
    return getattr(_fvdb_cpp.SmoothingMode, smoothing.name)


# ---------------------------------------------------------------------------
#  reinitialize_sdf  (fixed-topology redistance + de-staircase)
# ---------------------------------------------------------------------------


def reinitialize_sdf_batch(
    grid: GridBatch,
    field: JaggedTensor,
    band: int = 3,
    smooth: int = 0,
    order: int = 3,
    smoothing: SmoothingMode = SmoothingMode.MEAN_CURVATURE,
    redistance_iters: int = -1,
) -> JaggedTensor:
    """Re-initialize a signed per-voxel field into an SDF on the same grid batch.

    Redistances ``field`` to satisfy ``|grad phi| = 1`` (TVD-RK Godunov upwind eikonal solve with a
    frozen Peng sign), then optionally de-staircases it with curvature-based smoothing. The grid
    topology is unchanged: the returned field has the same per-voxel ordering as ``field``.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        field (JaggedTensor): Per-voxel signed field values.
        band (int): Narrow-band half-width in voxels. The field is clamped to ``[-band*vx, band*vx]``.
        smooth (int): Number of smoothing passes (``0`` disables smoothing).
        order (int): TVD-RK order, one of ``1`` (Euler), ``2`` (Heun), or ``3`` (Shu-Osher).
        smoothing (SmoothingMode): Which Laplacian flow each smoothing pass applies --
            :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` (default) or
            :attr:`~fvdb.SmoothingMode.TAUBIN` (volume-preserving). Only used when ``smooth > 0``.
        redistance_iters (int): Number of redistancing sweeps. ``<= 0`` uses the default
            ``max(6, round(2.5*band) + 2)``.

    Returns:
        sdf (JaggedTensor): The re-initialized SDF, same per-voxel ordering as ``field``.

    .. seealso:: :func:`reinitialize_sdf_single`, :func:`retopologize_sdf_batch`
    """
    result = _fvdb_cpp.reinitialize_sdf(
        grid.data, field._impl, band, redistance_iters, order, smooth, _to_cpp_smoothing(smoothing)
    )
    return JaggedTensor(impl=result)


def reinitialize_sdf_single(
    grid: Grid,
    field: torch.Tensor,
    band: int = 3,
    smooth: int = 0,
    order: int = 3,
    smoothing: SmoothingMode = SmoothingMode.MEAN_CURVATURE,
    redistance_iters: int = -1,
) -> torch.Tensor:
    """Re-initialize a signed per-voxel field into an SDF on a single grid.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        field (torch.Tensor): Per-voxel signed field values, shape ``(num_voxels,)``.
        band (int): Narrow-band half-width in voxels.
        smooth (int): Number of smoothing passes (``0`` disables smoothing).
        order (int): TVD-RK order, one of ``1``, ``2``, or ``3``.
        smoothing (SmoothingMode): Which Laplacian flow each smoothing pass applies --
            :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` (default) or
            :attr:`~fvdb.SmoothingMode.TAUBIN` (volume-preserving). Only used when ``smooth > 0``.
        redistance_iters (int): Number of redistancing sweeps. ``<= 0`` uses the default.

    Returns:
        sdf (torch.Tensor): The re-initialized SDF, shape ``(num_voxels,)``.

    .. seealso:: :func:`reinitialize_sdf_batch`, :func:`retopologize_sdf_single`
    """
    field_jt = JaggedTensor(field)
    result = _fvdb_cpp.reinitialize_sdf(
        grid.data, field_jt._impl, band, redistance_iters, order, smooth, _to_cpp_smoothing(smoothing)
    )
    return JaggedTensor(impl=result).jdata


# ---------------------------------------------------------------------------
#  retopologize_sdf  (reinitialize + narrow-band prune)
# ---------------------------------------------------------------------------


def retopologize_sdf_batch(
    grid: GridBatch,
    field: JaggedTensor,
    band: int = 3,
    smooth: int = 0,
    order: int = 3,
    smoothing: SmoothingMode = SmoothingMode.MEAN_CURVATURE,
    redistance_iters: int = -1,
    pad: bool = True,
    prune: bool = True,
) -> tuple[GridBatch, JaggedTensor]:
    """Retopologize a signed field into a clean narrow-band SDF on a (possibly pruned) grid batch.

    If ``pad`` is ``True`` the grid is first dilated by ``band`` voxels (so the eikonal solve has room
    to propagate a full-width band), then :func:`reinitialize_sdf_batch` is run, and finally, if
    ``prune`` is ``True``, the grid is pruned to the voxels strictly inside the band
    (``|phi| < band*vx*0.999``). The prune reuses :meth:`GridBatch.pruned_grid`; the resulting field
    is selected in the grid's canonical voxel order so it stays aligned with the pruned grid.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        field (JaggedTensor): Per-voxel signed field values.
        band (int): Narrow-band half-width in voxels.
        smooth (int): Number of smoothing passes (``0`` disables smoothing).
        order (int): TVD-RK order, one of ``1``, ``2``, or ``3``.
        smoothing (SmoothingMode): Which Laplacian flow each smoothing pass applies --
            :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` (default) or
            :attr:`~fvdb.SmoothingMode.TAUBIN` (volume-preserving). Only used when ``smooth > 0``.
        redistance_iters (int): Number of redistancing sweeps. ``<= 0`` uses the default.
        pad (bool): If ``True`` (default) dilate the grid by ``band`` voxels before redistancing so
            the output narrow band is a full ``band`` voxels wide even if the input grid had a
            thinner active region. Newly added voxels are seeded as *exterior* (``+band*vx``), which
            is correct when the dilation extends outward -- i.e. when the grid's interior (the
            ``phi < 0`` region) is already represented (the usual case for occupancy/TSDF/mesh-derived
            fields). For a thin shell that does not fill its interior, pass ``pad=False`` and supply a
            grid that already has an adequate band.
        prune (bool): If ``True`` prune to the narrow band; if ``False`` return the (possibly
            padded) grid and the re-initialized field unchanged.

    Returns:
        out_grid (GridBatch): The pruned (or, with ``prune=False``, the padded/original) grid batch.
        sdf (JaggedTensor): The narrow-band SDF, aligned with ``out_grid``.

    .. seealso:: :func:`retopologize_sdf_single`, :func:`reinitialize_sdf_batch`
    """
    if pad:
        # Seed fresh voxels as exterior. A single positive seed >= every grid's band width is fine:
        # reinitialize_sdf re-clamps it to that grid's +band*vx, so only its (positive) sign matters.
        seed = band * float(grid.voxel_sizes[:, 0].max())
        dilated = grid.dilated_grid(band)
        field = inject_batch(dilated, grid, field, default_value=seed)
        grid = dilated
    phi = reinitialize_sdf_batch(grid, field, band, smooth, order, smoothing, redistance_iters)
    if not prune:
        return grid, phi
    # per-voxel band half-width (voxel size may vary per grid in the batch)
    voxel_sizes = grid.voxel_sizes[:, 0].to(phi.jdata.device)
    band_width = band * voxel_sizes[phi.jidx.long()] * 0.999
    mask = phi.jdata.abs() < band_width
    return grid.pruned_grid(phi.jagged_like(mask)), phi.rmask(mask)


def retopologize_sdf_single(
    grid: Grid,
    field: torch.Tensor,
    band: int = 3,
    smooth: int = 0,
    order: int = 3,
    smoothing: SmoothingMode = SmoothingMode.MEAN_CURVATURE,
    redistance_iters: int = -1,
    pad: bool = True,
    prune: bool = True,
) -> tuple[Grid, torch.Tensor]:
    """Retopologize a signed field into a clean narrow-band SDF on a (possibly pruned) single grid.

    If ``pad`` is ``True`` the grid is first dilated by ``band`` voxels (so the eikonal solve has room
    to propagate a full-width band), then :func:`reinitialize_sdf_single` is run, and finally, if
    ``prune`` is ``True``, the grid is pruned to the voxels strictly inside the band
    (``|phi| < band*vx*0.999``).

    Args:
        grid (Grid): The single grid defining the sparse topology.
        field (torch.Tensor): Per-voxel signed field values, shape ``(num_voxels,)``.
        band (int): Narrow-band half-width in voxels.
        smooth (int): Number of smoothing passes (``0`` disables smoothing).
        order (int): TVD-RK order, one of ``1``, ``2``, or ``3``.
        smoothing (SmoothingMode): Which Laplacian flow each smoothing pass applies --
            :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` (default) or
            :attr:`~fvdb.SmoothingMode.TAUBIN` (volume-preserving). Only used when ``smooth > 0``.
        redistance_iters (int): Number of redistancing sweeps. ``<= 0`` uses the default.
        pad (bool): If ``True`` (default) dilate the grid by ``band`` voxels before redistancing so
            the output narrow band is a full ``band`` voxels wide even if the input grid had a
            thinner active region. Newly added voxels are seeded as *exterior* (``+band*vx``), which
            is correct when the dilation extends outward -- i.e. when the grid's interior (the
            ``phi < 0`` region) is already represented (the usual case for occupancy/TSDF/mesh-derived
            fields). For a thin shell that does not fill its interior, pass ``pad=False`` and supply a
            grid that already has an adequate band.
        prune (bool): If ``True`` prune to the narrow band; if ``False`` return the (possibly padded)
            grid and the re-initialized field unchanged.

    Returns:
        out_grid (Grid): The pruned (or, with ``prune=False``, the padded/original) grid.
        sdf (torch.Tensor): The narrow-band SDF, aligned with ``out_grid``.

    .. seealso:: :func:`retopologize_sdf_batch`, :func:`reinitialize_sdf_single`
    """
    band_width = band * float(grid.voxel_size[0])
    if pad:
        dilated = grid.dilated_grid(band)
        field = inject_single(dilated, grid, field, default_value=band_width)
        grid = dilated
    phi = reinitialize_sdf_single(grid, field, band, smooth, order, smoothing, redistance_iters)
    if not prune:
        return grid, phi
    mask = phi.abs() < band_width * 0.999
    return grid.pruned_grid(mask), phi[mask]
