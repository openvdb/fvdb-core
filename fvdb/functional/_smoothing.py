# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for umbrella-Laplacian smoothing of per-voxel fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from .. import _fvdb_cpp
from ..enums import SmoothingMode, _to_cpp_enum
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


def _to_cpp_smoothing(mode: SmoothingMode) -> "_fvdb_cpp.SmoothingMode":
    """Convert a public :class:`fvdb.SmoothingMode` to the bound C++ enum (matched by member name)."""
    return _to_cpp_enum(SmoothingMode, _fvdb_cpp.SmoothingMode, mode)


def _boundary_values(
    boundary_value: float | Sequence[float] | torch.Tensor | None, grid_count: int
) -> torch.Tensor | None:
    """One boundary value per grid as a 1-D tensor, or ``None`` for zero flux.

    Python numbers stay on the host (as float64, so they are rounded once, to the field's dtype),
    where the op validates them without a device synchronization; a tensor keeps its device."""
    if boundary_value is None:
        return None
    if isinstance(boundary_value, torch.Tensor):
        values = boundary_value
        if not values.is_floating_point():
            values = values.to(torch.float64)
    else:
        values = torch.as_tensor(boundary_value, dtype=torch.float64)
    if values.dim() == 0:
        values = values.expand(grid_count)
    return values


def laplacian_smooth_batch(
    grid: GridBatch,
    field: JaggedTensor,
    iterations: int = 1,
    mode: SmoothingMode = SmoothingMode.MEAN_CURVATURE,
    boundary_value: float | Sequence[float] | torch.Tensor | None = None,
) -> JaggedTensor:
    """Smooth a per-voxel field on a grid batch with the 6-neighbour umbrella Laplacian.

    Each pass moves every voxel toward the mean of its six face neighbours,
    ``out = in + w * (mean(neighbours) - in)``, reading the field as it was before the pass, and
    treats every channel independently. :attr:`~fvdb.SmoothingMode.MEAN_CURVATURE` applies one
    pass with ``w = 1`` per iteration (on a signed distance field this approximates
    mean-curvature flow, which shrinks surfaces); :attr:`~fvdb.SmoothingMode.TAUBIN` applies a
    ``w = 0.5`` pass followed by a ``w = -0.53`` pass per iteration, which smooths with much less
    shrinkage. The grid topology is unchanged.

    Args:
        grid (GridBatch): The grid batch defining the sparse topology.
        field (JaggedTensor): Per-voxel values with ``jdata`` of shape ``(N,)`` or ``(N, C)``,
            ``float32`` or ``float64``.
        iterations (int): Number of smoothing iterations (``0`` returns a copy).
        mode (SmoothingMode): The smoothing schedule each iteration applies.
        boundary_value (float | Sequence[float] | torch.Tensor | None): How an inactive face
            neighbour is read. ``None`` (default) reads it as the centre voxel's own value, so the
            face contributes nothing (zero flux) -- the natural choice for features, colours and
            other unsigned data. A non-negative value ``b`` (one for the whole batch, or one per
            grid) reads it as ``-b`` beside a negative centre value and ``+b`` otherwise (sign
            continuation). Use this for signed fields: for a narrow-band SDF pass
            ``band * voxel_size`` so the inactive inside and outside of the band continue the
            adjacent sign, as :meth:`~fvdb.GridBatch.reinitialize_sdf` assumes.

    Returns:
        smoothed (JaggedTensor): The smoothed field, same shape and per-voxel ordering as ``field``.

    Note:
        * Smoothing moves the zero crossing of a signed field and breaks ``|grad phi| = 1``. To
          smooth an SDF, redistance afterwards::

              sdf = grid.reinitialize_sdf(field, band=band)
              sdf = grid.laplacian_smooth(sdf, iterations=n, boundary_value=band * grid.voxel_sizes[:, 0])
              sdf = grid.reinitialize_sdf(sdf, band=band)

        * With ``w = 1`` the update replaces each voxel with its neighbour mean, which does not
          damp a field that alternates sign from voxel to voxel. Use :attr:`~fvdb.SmoothingMode.TAUBIN`
          for such data.
        * The weights are uniform and do not depend on the voxel size. Not differentiable. CUDA only.

    .. seealso:: :func:`laplacian_smooth_single`, :func:`reinitialize_sdf_batch`
    """
    boundary = _boundary_values(boundary_value, grid.grid_count)
    result = _fvdb_cpp.laplacian_smooth(grid.data, field._impl, iterations, _to_cpp_smoothing(mode), boundary)
    return JaggedTensor(impl=result)


def laplacian_smooth_single(
    grid: Grid,
    field: torch.Tensor,
    iterations: int = 1,
    mode: SmoothingMode = SmoothingMode.MEAN_CURVATURE,
    boundary_value: float | None = None,
) -> torch.Tensor:
    """Smooth a per-voxel field on a single grid with the 6-neighbour umbrella Laplacian.

    Args:
        grid (Grid): The single grid defining the sparse topology.
        field (torch.Tensor): Per-voxel values, shape ``(num_voxels,)`` or ``(num_voxels, C)``,
            ``float32`` or ``float64``.
        iterations (int): Number of smoothing iterations (``0`` returns a copy).
        mode (SmoothingMode): The smoothing schedule each iteration applies.
        boundary_value (float | None): How an inactive face neighbour is read: ``None`` (default)
            for zero flux, or a non-negative ``b`` read as ``-b`` beside a negative centre value and
            ``+b`` otherwise. For a narrow-band SDF pass ``band * voxel_size``.

    Returns:
        smoothed (torch.Tensor): The smoothed field, same shape and per-voxel ordering as ``field``.

    Note:
        See :func:`laplacian_smooth_batch` for the update and the boundary rules.

    .. seealso:: :func:`laplacian_smooth_batch`, :func:`reinitialize_sdf_single`
    """
    boundary = _boundary_values(boundary_value, 1)
    field_jt = JaggedTensor(field)
    result = _fvdb_cpp.laplacian_smooth(grid.data, field_jt._impl, iterations, _to_cpp_smoothing(mode), boundary)
    return JaggedTensor(impl=result).jdata
