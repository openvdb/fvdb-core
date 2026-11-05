# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for signed-distance-field (SDF) re-initialization and narrow-band rebuild."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _fvdb_cpp
from ..enums import SmoothingMode, _to_cpp_enum
from ..jagged_tensor import JaggedTensor
from ._dense import inject_batch, inject_single

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


def _to_cpp_smoothing(smoothing: SmoothingMode) -> "_fvdb_cpp.SmoothingMode":
    """Convert a public :class:`fvdb.SmoothingMode` to the bound C++ enum (matched by member name)."""
    return _to_cpp_enum(SmoothingMode, _fvdb_cpp.SmoothingMode, smoothing)


def _validated_scalar_field(field: torch.Tensor, num_voxels: int) -> torch.Tensor:
    """Validate before padding, where NaN is reserved for newly activated voxels."""
    if field.dim() not in (1, 2) or (field.dim() == 2 and field.shape[1] != 1):
        raise ValueError(f"field must be a scalar field with shape (N,) or (N, 1), got {tuple(field.shape)}")
    if field.shape[0] != num_voxels:
        raise ValueError(f"field must have one value per voxel, got {field.shape[0]} values for {num_voxels} voxels")
    if not torch.isfinite(field).all().item():
        raise ValueError("field must contain only finite values; leave no-data voxels inactive")
    return field.reshape(-1)


# ---------------------------------------------------------------------------
#  sign-aware padding (shared by rebuild_narrow_band_{single,batch})
# ---------------------------------------------------------------------------


def _seed_sign(neighbor_values_sum: torch.Tensor, band_width: torch.Tensor | float) -> torch.Tensor:
    """``-band_width`` where the summed neighbour values are negative, else ``+band_width``."""
    return torch.where(neighbor_values_sum < 0, -band_width, band_width)


def _pad_with_sign_single(grid: Grid, field: torch.Tensor, band: int, band_width: float) -> tuple[Grid, torch.Tensor]:
    """Dilate ``grid`` by ``band`` voxels, one layer at a time, seeding each fresh voxel with
    ``+/-band_width`` according to the sign of its already-present 26-neighbours.

    An IndexGrid cannot mark inactive space as interior or exterior, so a constant exterior seed
    would turn the inside of a hollow narrow band into "outside". Growing one layer at a time and
    copying the sign of the neighbours lets the padding continue the field inward and outward.
    """
    for _ in range(band):
        dilated = grid.dilated_grid(1)
        padded = inject_single(dilated, grid, field, default_value=float("nan"))
        is_new = torch.isnan(padded)
        if is_new.any():
            nbr = grid.neighbor_indexes(dilated.ijk[is_new], 1).reshape(int(is_new.sum()), -1)
            zero = torch.zeros((), dtype=field.dtype, device=field.device)
            nbr_sum = torch.where(nbr >= 0, field[nbr.clamp(min=0)], zero).sum(dim=1)
            padded[is_new] = _seed_sign(nbr_sum, band_width).to(field.dtype)
        grid, field = dilated, padded
    return grid, field


def _pad_with_sign_batch(
    grid: GridBatch, field: JaggedTensor, band: int, band_width: torch.Tensor
) -> tuple[GridBatch, JaggedTensor]:
    """Batched :func:`_pad_with_sign_single`; ``band_width`` holds one half-width per grid."""
    for _ in range(band):
        dilated = grid.dilated_grid(1)
        padded = inject_batch(dilated, grid, field, default_value=float("nan"))
        is_new = torch.isnan(padded.jdata)
        if is_new.any():
            new_ijk = dilated.ijk.rmask(is_new)
            # neighbor_indexes returns per-grid indices; shift them into the flat jdata.
            nbr = grid.neighbor_indexes(new_ijk, 1).jdata.reshape(int(is_new.sum()), -1)
            grid_of_new = new_ijk.jidx.long()
            flat = nbr + field.joffsets[grid_of_new, None].to(nbr.device)
            zero = torch.zeros((), dtype=field.jdata.dtype, device=field.jdata.device)
            nbr_sum = torch.where(nbr >= 0, field.jdata[flat.clamp(min=0)], zero).sum(dim=1)
            bw = band_width.to(field.jdata.device, field.jdata.dtype)[grid_of_new]
            data = padded.jdata.clone()
            data[is_new] = _seed_sign(nbr_sum, bw)
            padded = padded.jagged_like(data)
        grid, field = dilated, padded
    return grid, field


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

    Note:
        * ``field`` must represent the intended inside/outside regions and zero crossings. Its
          magnitudes need not be accurate distances, but affect convergence, accuracy, and sub-voxel
          surface location. With ``smooth=0`` redistancing aims to preserve the input surface, subject
          to discretization error; smoothing moves the surface and then re-redistances.
        * Values must be finite, with shape ``(N,)`` or ``(N, 1)`` (for a batch, the shape of
          ``field.jdata``). Invalid shapes and NaN/Inf values raise ``ValueError``.
        * The surface is where the field changes sign between *active* voxels; one active voxel of
          each sign across the crossing is sufficient (two or more per side gives the best sub-voxel
          accuracy). A grid whose active values are all one sign has no surface: the result is the
          constant ``-/+band*vx`` and :func:`rebuild_narrow_band_batch` returns an empty band, which
          is correct for e.g. a tile that lies entirely inside an object.
        * Inactive neighbours read as ``+/-band*vx`` using the adjacent voxel's frozen sign during
          redistancing and its current sign during smoothing. Both filled solids (interior active)
          and narrow bands whose interior is inactive are valid inputs. Leave no-data voxels
          *inactive* rather than assigning them NaN/Inf values.
        * Voxels whose value is exactly ``0`` have a zero frozen sign and are left at ``0`` by the
          redistance (a no-data pass-through relied on by the ray-implicit-intersection op, which
          treats exact ``0`` as a gap). Their signed neighbours, however, see them as an interface and
          are redistanced toward them, and smoothing blends them -- prune such voxels first when you
          can.
        * Each grid must have isotropic voxels (``ValueError`` otherwise); grids in the batch may
          differ from one another. CUDA only; ``float32`` or ``float64``.

    .. seealso:: :func:`reinitialize_sdf_single`, :func:`rebuild_narrow_band_batch`
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
        field (torch.Tensor): Per-voxel signed field values, shape ``(num_voxels,)`` or ``(num_voxels, 1)``.
        band (int): Narrow-band half-width in voxels.
        smooth (int): Number of smoothing passes (``0`` disables smoothing).
        order (int): TVD-RK order, one of ``1``, ``2``, or ``3``.
        smoothing (SmoothingMode): Which Laplacian flow each smoothing pass applies --
            :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` (default) or
            :attr:`~fvdb.SmoothingMode.TAUBIN` (volume-preserving). Only used when ``smooth > 0``.
        redistance_iters (int): Number of redistancing sweeps. ``<= 0`` uses the default.

    Returns:
        sdf (torch.Tensor): The re-initialized SDF, shape ``(num_voxels,)``.

    Note:
        See :func:`reinitialize_sdf_batch` for the input contract.

    .. seealso:: :func:`reinitialize_sdf_batch`, :func:`rebuild_narrow_band_single`
    """
    field_jt = JaggedTensor(field)
    result = _fvdb_cpp.reinitialize_sdf(
        grid.data, field_jt._impl, band, redistance_iters, order, smooth, _to_cpp_smoothing(smoothing)
    )
    return JaggedTensor(impl=result).jdata


# ---------------------------------------------------------------------------
#  rebuild_narrow_band  (pad + reinitialize + narrow-band prune)
# ---------------------------------------------------------------------------


def rebuild_narrow_band_batch(
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
    """Rebuild a signed field into a clean narrow-band SDF on a (possibly pruned) grid batch.

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
            thinner active region. The dilation grows one layer at a time and seeds each new voxel
            with ``+/-band*vx`` according to the sign of its existing neighbours, so padding
            continues a narrow band both outward (exterior) and inward (interior) and works for
            filled solids as well as narrow bands whose interior is inactive.
        prune (bool): If ``True`` prune to the narrow band; if ``False`` return the (possibly
            padded) grid and the re-initialized field unchanged.

    Returns:
        out_grid (GridBatch): The pruned (or, with ``prune=False``, the padded/original) grid batch.
        sdf (JaggedTensor): The narrow-band SDF, aligned with ``out_grid``.

    Note:
        Applying this to its own output reproduces it (up to a voxel layer at the band edge). See
        :func:`reinitialize_sdf_batch` for the input contract.

    .. seealso:: :func:`rebuild_narrow_band_single`, :func:`reinitialize_sdf_batch`
    """
    # per-grid narrow-band half-width; voxel size may vary across the batch
    band_width = band * grid.voxel_sizes[:, 0]
    field = field.jagged_like(_validated_scalar_field(field.jdata, grid.total_voxels))
    if pad:
        grid, field = _pad_with_sign_batch(grid, field, band, band_width)
    phi = reinitialize_sdf_batch(grid, field, band, smooth, order, smoothing, redistance_iters)
    if not prune:
        return grid, phi
    mask = phi.jdata.abs() < band_width.to(phi.jdata.device)[phi.jidx.long()] * 0.999
    return grid.pruned_grid(phi.jagged_like(mask)), phi.rmask(mask)


def rebuild_narrow_band_single(
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
    """Rebuild a signed field into a clean narrow-band SDF on a (possibly pruned) single grid.

    If ``pad`` is ``True`` the grid is first dilated by ``band`` voxels (so the eikonal solve has room
    to propagate a full-width band), then :func:`reinitialize_sdf_single` is run, and finally, if
    ``prune`` is ``True``, the grid is pruned to the voxels strictly inside the band
    (``|phi| < band*vx*0.999``).

    Args:
        grid (Grid): The single grid defining the sparse topology.
        field (torch.Tensor): Per-voxel signed field values, shape ``(num_voxels,)`` or ``(num_voxels, 1)``.
        band (int): Narrow-band half-width in voxels.
        smooth (int): Number of smoothing passes (``0`` disables smoothing).
        order (int): TVD-RK order, one of ``1``, ``2``, or ``3``.
        smoothing (SmoothingMode): Which Laplacian flow each smoothing pass applies --
            :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` (default) or
            :attr:`~fvdb.SmoothingMode.TAUBIN` (volume-preserving). Only used when ``smooth > 0``.
        redistance_iters (int): Number of redistancing sweeps. ``<= 0`` uses the default.
        pad (bool): If ``True`` (default) dilate the grid by ``band`` voxels before redistancing so
            the output narrow band is a full ``band`` voxels wide even if the input grid had a
            thinner active region. The dilation grows one layer at a time and seeds each new voxel
            with ``+/-band*vx`` according to the sign of its existing neighbours, so padding
            continues a narrow band both outward (exterior) and inward (interior) and works for
            filled solids as well as narrow bands whose interior is inactive.
        prune (bool): If ``True`` prune to the narrow band; if ``False`` return the (possibly padded)
            grid and the re-initialized field unchanged.

    Returns:
        out_grid (Grid): The pruned (or, with ``prune=False``, the padded/original) grid.
        sdf (torch.Tensor): The narrow-band SDF, aligned with ``out_grid``.

    Note:
        Applying this to its own output reproduces it (up to a voxel layer at the band edge). See
        :func:`reinitialize_sdf_batch` for the input contract.

    .. seealso:: :func:`rebuild_narrow_band_batch`, :func:`reinitialize_sdf_single`
    """
    # narrow-band half-width
    band_width = band * float(grid.voxel_size[0])
    field = _validated_scalar_field(field, grid.num_voxels)
    if pad:
        grid, field = _pad_with_sign_single(grid, field, band, band_width)
    phi = reinitialize_sdf_single(grid, field, band, smooth, order, smoothing, redistance_iters)
    if not prune:
        return grid, phi
    mask = phi.abs() < band_width * 0.999
    return grid.pruned_grid(mask), phi[mask]
