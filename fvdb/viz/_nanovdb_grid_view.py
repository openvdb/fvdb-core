# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Shared helpers for adding fvdb grids to the viewer as NanoVDB grid views.

The nanovdb-editor renders exactly one grid per view (the shader decodes a single
grid anchored at byte offset 0 of the uploaded buffer). A :class:`~fvdb.GridBatch`
may hold many grids, so a batch is expanded into one named editor view per grid:
the base ``name`` when the batch holds a single grid, or ``name[i]`` for grid ``i``
of a multi-grid batch.
"""
from typing import TYPE_CHECKING

import torch

from ._viewer_server import _get_viewer_server_cpp

if TYPE_CHECKING:
    from .. import Grid, GridBatch, JaggedTensor


def _to_grid_batch(grid: "Grid | GridBatch") -> "GridBatch":
    """Normalize a :class:`~fvdb.Grid` or :class:`~fvdb.GridBatch` to a ``GridBatch``."""
    from .. import Grid, GridBatch

    if isinstance(grid, Grid):
        return grid.to_gridbatch()
    if isinstance(grid, GridBatch):
        return grid
    raise TypeError(f"grid must be a fvdb.Grid or fvdb.GridBatch, got {type(grid).__name__}")


def _validate_values(values: "JaggedTensor", grid_batch: "GridBatch", arg_name: str) -> None:
    """Validate that ``values`` holds one float32 value per active voxel in ``grid_batch``."""
    from .. import JaggedTensor

    if not isinstance(values, JaggedTensor):
        raise TypeError(f"{arg_name} must be a fvdb.JaggedTensor, got {type(values).__name__}")
    if values.jdata.dtype != torch.float32:
        raise TypeError(f"{arg_name} must be a float32 JaggedTensor, got {values.jdata.dtype}")
    if values.jdata.ndim != 1 or values.jdata.shape[0] != grid_batch.total_voxels:
        raise ValueError(
            f"{arg_name} must be 1D with {grid_batch.total_voxels} entries "
            f"(one per active voxel across the batch), got shape {tuple(values.jdata.shape)}"
        )


def _sub_view_name(name: str, index: int, count: int) -> str:
    """Return the per-grid view name: ``name`` for a single grid, else ``name[index]``."""
    return name if count == 1 else f"{name}[{index}]"


def add_nanovdb_grid_views(
    scene_name: str,
    name: str,
    grid: "Grid | GridBatch",
    values: "JaggedTensor",
    server_method: str,
    arg_name: str,
) -> list[str]:
    """Add one NanoVDB grid view per grid in ``grid`` and return the created view names.

    Args:
        scene_name: The scene to add the views to.
        name: Base name for the view(s). A single grid uses ``name``; a multi-grid
            batch uses ``name[i]`` for grid ``i``.
        grid: A :class:`~fvdb.Grid` or :class:`~fvdb.GridBatch` defining the domain.
        values: A float32 :class:`~fvdb.JaggedTensor` with one value per active voxel
            across the whole batch.
        server_method: Name of the viewer-server method to call per grid, which selects the
            render pipeline (e.g. ``"add_level_set_view"`` or ``"add_fog_volume_view"``).
        arg_name: Name of the ``values`` argument, used in error messages (e.g. ``"sdf"``).

    Returns:
        view_names (list[str]): The names of the editor views that were created.
    """
    grid_batch = _to_grid_batch(grid)
    _validate_values(values, grid_batch, arg_name)

    server = _get_viewer_server_cpp()
    add_one = getattr(server, server_method)
    count = grid_batch.grid_count
    view_names: list[str] = []
    for i in range(count):
        sub_name = _sub_view_name(name, i, count)
        add_one(scene_name, sub_name, grid_batch[i].data, values[i]._impl)
        view_names.append(sub_name)
    return view_names


def update_nanovdb_grid_views(
    scene_name: str,
    name: str,
    prev_view_names: list[str],
    grid: "Grid | GridBatch",
    values: "JaggedTensor",
    server_method: str,
    arg_name: str,
) -> list[str]:
    """Re-add grid views for ``grid``/``values``, removing views left stale by a topology change.

    Views that share a name with the previous set are overwritten in place (no flicker);
    any previous view whose name is no longer produced (e.g. the batch shrank) is removed.

    Returns:
        view_names (list[str]): The names of the editor views after the update.
    """
    view_names = add_nanovdb_grid_views(scene_name, name, grid, values, server_method, arg_name)

    server = _get_viewer_server_cpp()
    for stale in set(prev_view_names) - set(view_names):
        server.remove_view(scene_name, stale)
    return view_names
