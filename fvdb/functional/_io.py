# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for loading and saving grid batches in NanoVDB format."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor
from ..types import DeviceIdentifier, resolve_device
if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


def _wrap_grid(cpp_impl):
    from ..grid_batch import GridBatch

    return GridBatch(data=cpp_impl)


# ---------------------------------------------------------------------------
#  Load
# ---------------------------------------------------------------------------


@overload
def load_nanovdb(
    path: str,
    *,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_nanovdb(
    path: str,
    *,
    indices: list[int],
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_nanovdb(
    path: str,
    *,
    index: int,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_nanovdb(
    path: str,
    *,
    names: list[str],
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load_nanovdb(
    path: str,
    *,
    name: str,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


def load_nanovdb(
    path: str,
    *,
    indices: list[int] | None = None,
    index: int | None = None,
    names: list[str] | None = None,
    name: str | None = None,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]:
    """
    Load a grid batch from a ``.nvdb`` file.

    Args:
        path: Path to the ``.nvdb`` file.
        indices: Optional list of grid indices to load.
        index: Optional single grid index to load.
        names: Optional list of grid names to load.
        name: Optional single grid name to load.
        device: Device to load onto. Defaults to ``"cpu"``.
        verbose: Print information about loaded grids.

    Returns:
        A tuple ``(grid_batch, data, names)``.
    """
    from .._fvdb_cpp import load as _load

    device = resolve_device(device)

    selectors = [indices is not None, index is not None, names is not None, name is not None]
    if sum(selectors) > 1:
        raise ValueError("Only one of indices, index, names, or name can be specified")

    if indices is not None:
        grid_impl, data_impl, names_out = _load(path, indices, device, verbose)
    elif index is not None:
        grid_impl, data_impl, names_out = _load(path, index, device, verbose)
    elif names is not None:
        grid_impl, data_impl, names_out = _load(path, names, device, verbose)
    elif name is not None:
        grid_impl, data_impl, names_out = _load(path, name, device, verbose)
    else:
        grid_impl, data_impl, names_out = _load(path, device, verbose)

    return _wrap_grid(grid_impl), JaggedTensor(impl=data_impl), names_out


# ---------------------------------------------------------------------------
#  Save
# ---------------------------------------------------------------------------


def save_nanovdb(
    grid: GridBatch,
    path: str,
    data: JaggedTensor | None = None,
    names: list[str] | str | None = None,
    name: str | None = None,
    compressed: bool = False,
    verbose: bool = False,
) -> None:
    """
    Save a grid batch and optional voxel data to a ``.nvdb`` file.

    Args:
        grid: The grid batch to save.
        path: File path (should have ``.nvdb`` extension).
        data: Optional voxel data to save with the grids.
        names: Names for each grid, or a single name for all.
        name: Single name for all grids (takes precedence over ``names``).
        compressed: Use Blosc compression. Default ``False``.
        verbose: Print information about saved grids. Default ``False``.
    """
    from .._fvdb_cpp import save as _save

    grid_data = grid.data
    data_impl = data._impl if data else None
    if name is not None:
        _save(path, grid_data, data_impl, name, compressed, verbose)
    elif names is not None:
        if isinstance(names, str):
            _save(path, grid_data, data_impl, names, compressed, verbose)
        else:
            _save(path, grid_data, data_impl, names, compressed, verbose)
    else:
        _save(path, grid_data, data_impl, [], compressed, verbose)


# ---------------------------------------------------------------------------
#  Single variants (Grid + torch.Tensor)
# ---------------------------------------------------------------------------


def _wrap_single_grid(cpp_impl):
    from ..grid import Grid

    return Grid(data=cpp_impl)


def load_nanovdb_single(
    path: str,
    *,
    index: int = 0,
    name: str | None = None,
    device: DeviceIdentifier = "cpu",
    verbose: bool = False,
) -> tuple[Grid, torch.Tensor, str]:
    """Load a single grid from a ``.nvdb`` file.

    Args:
        path: Path to the ``.nvdb`` file.
        index: Grid index to load. Default ``0``.
        name: Optional grid name to load (overrides ``index``).
        device: Device to load onto. Defaults to ``"cpu"``.
        verbose: Print information about loaded grids.

    Returns:
        A tuple ``(grid, data, name)``.
    """
    import torch

    if name is not None:
        gb, jt_data, names_out = load_nanovdb(path, name=name, device=device, verbose=verbose)
    else:
        gb, jt_data, names_out = load_nanovdb(path, index=index, device=device, verbose=verbose)

    return _wrap_single_grid(gb.data), jt_data.jdata, names_out[0] if names_out else ""


def save_nanovdb_single(
    grid: Grid,
    path: str,
    data: torch.Tensor | None = None,
    name: str | None = None,
    compressed: bool = False,
    verbose: bool = False,
) -> None:
    """Save a single grid and optional voxel data to a ``.nvdb`` file.

    Args:
        grid: The single grid to save.
        path: File path (should have ``.nvdb`` extension).
        data: Optional voxel data as a plain tensor.
        name: Optional name for the grid.
        compressed: Use Blosc compression.
        verbose: Print information about saved grids.
    """
    import torch
    from .._fvdb_cpp import save as _save

    grid_data = grid.data
    if data is not None:
        data_impl = JaggedTensor(data)._impl
    else:
        data_impl = None
    if name is not None:
        _save(path, grid_data, data_impl, name, compressed, verbose)
    else:
        _save(path, grid_data, data_impl, [], compressed, verbose)
