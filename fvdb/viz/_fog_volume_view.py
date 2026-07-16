# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import TYPE_CHECKING, Any

import torch

from ._nanovdb_grid_view import update_nanovdb_grid_views

if TYPE_CHECKING:
    from .. import Grid, GridBatch, JaggedTensor


class FogVolumeView:
    """
    A view for rendering an fvdb :class:`~fvdb.Grid` (or :class:`~fvdb.GridBatch`) as a
    volumetric fog in the viewer.

    The grid is rendered via ray-marching using the ``nanovdb_render`` pipeline.  Per-voxel
    density values are stored as float32 blind metadata on the ONINDEX NanoVDB grid.

    The nanovdb-editor renders one grid per view.  A :class:`~fvdb.GridBatch` with more
    than one grid is therefore expanded into one view per grid, named ``name[i]``.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        scene_name: str,
        name: str,
        view_names: list[str],
        _private: Any = None,
    ):
        """
        .. warning::

            This constructor is private.  Use :meth:`fvdb.viz.Scene.add_fog_volume` instead.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("FogVolumeView constructor is private. Use Scene.add_fog_volume().")
        self._scene_name = scene_name
        self._name = name
        # The editor view names backing this fog volume (one per grid in the batch).
        self._view_names = view_names

    @property
    def name(self) -> str:
        return self._name

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @torch.no_grad()
    def update(self, grid: "Grid | GridBatch", density: "JaggedTensor") -> None:
        """
        Replace the fog-volume data in the viewer.

        Args:
            grid: The sparse grid (or batch of grids) the density field lives on.
            density: Per-voxel float32 density values (one per active voxel, non-negative).
        """
        self._view_names = update_nanovdb_grid_views(
            self._scene_name,
            self._name,
            self._view_names,
            grid,
            density,
            "add_fog_volume_view",
            "density",
        )
