# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Union

import numpy as np
import torch
from fvdb.types import DeviceIdentifier, NumericMaxRank1, to_Vec3f, to_Vec3i

from fvdb import Grid, GridBatch, JaggedTensor


def make_dense_grid_and_point_data(
    num_voxels_single_axis: int, device: DeviceIdentifier, dtype: torch.dtype
) -> tuple[Grid, Grid, torch.Tensor]:
    """
    Create a dense grid and point data for testing.

    Args:
        num_voxels_single_axis: Number of voxels along a single side of the cube.
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        tuple: (primal_grid, dual_grid, points)
    """
    grid_origin = (0.0, 0.0, 0.0)
    voxel_size = 1.0 / (np.floor(0.5 * num_voxels_single_axis) + 0.5)
    target_vox = int(2 * np.floor(0.5 * num_voxels_single_axis) + 1) ** 3
    target_corners = int(2 * np.floor(0.5 * num_voxels_single_axis) + 2) ** 3

    p = (2.0 * torch.rand(1, 3) - 1.0).to(device).to(dtype)
    grid: Grid | None = None
    while grid is None or grid.num_voxels != target_vox:
        p = (2.0 * torch.rand(10 * p.shape[0], 3) - 1.0).to(device)
        p = torch.clip(p, -1.0 + 0.25 * voxel_size, 1.0 - 0.25 * voxel_size).to(dtype)
        grid = Grid.from_points(p, voxel_size, grid_origin, device=device)

    grid_d = grid.dual_grid()
    assert grid_d.num_voxels == target_corners
    dual_corners_xyz = grid_d.grid_to_world(grid_d.ijk.float())
    assert torch.allclose(dual_corners_xyz.min(0)[0], -torch.ones(3).to(dual_corners_xyz))
    assert torch.allclose(dual_corners_xyz.max(0)[0], torch.ones(3).to(dual_corners_xyz))

    return grid, grid_d, p


def make_dense_grid_batch_and_jagged_point_data(
    num_voxels_single_axis: int, device: DeviceIdentifier, dtype: torch.dtype
) -> tuple[GridBatch, GridBatch, JaggedTensor]:
    """
    Create a dense grid batch (size 1) and point data for testing.

    Args:
        num_voxels_single_axis: Number of voxels along a single side of the cube.
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        tuple: (primal_grid_batch, dual_grid_batch, jagged_points)
    """
    grid, grid_d, points = make_dense_grid_and_point_data(num_voxels_single_axis, device, dtype)
    grid_batch = GridBatch.from_grid(grid)
    grid_batch_d = GridBatch.from_grid(grid_d)
    jagged_points = JaggedTensor(points)
    return grid_batch, grid_batch_d, jagged_points


def make_grid_and_point_data(
    device: DeviceIdentifier, dtype: torch.dtype, include_boundary_points: bool = False, expand: int = 10
) -> tuple[Grid, Grid, torch.Tensor]:
    """Create a grid and point data for testing.

    Args:
        device: Device to create tensors on
        dtype: Data type for tensors
        include_boundary_points: If True, create points both inside and outside the grid
        expand: Number of times to replicate points when creating boundary samples

    Returns:
        tuple: (primal_grid, dual_grid, points)
    """
    p = torch.randn((100, 3), device=device, dtype=dtype)
    grid = Grid.from_points(p, voxel_size=0.05, origin=0, device=device).dilated_grid(1)
    grid_d = grid.dual_grid()

    if not include_boundary_points:
        return grid, grid_d, p

    # Ensure some samples land inside and outside the voxel grid
    # We create samples by sampling in a radius roughly the size of a voxel around each
    # voxel center so that some points will land outside but only slightly
    found = False
    mask = torch.zeros(1)
    samples = torch.zeros(1)
    while not found:
        # Do everything in double then cast so fp16 samples are
        # as close as possible from double and float
        primal_pts = grid.grid_to_world(grid.ijk.double())
        samples = torch.cat([primal_pts] * expand, dim=0)
        samples += torch.randn_like(samples) * grid.voxel_size
        mask = grid.points_in_active_voxel(samples)
        found = not (torch.all(mask) or torch.all(~mask))

    samples = samples.to(dtype)

    assert not torch.all(mask)
    assert not torch.all(~mask)

    return grid, grid_d, samples


def make_grid_batch_and_jagged_point_data(
    device: DeviceIdentifier, dtype: torch.dtype, include_boundary_points: bool = False, expand: int = 10
) -> tuple[GridBatch, GridBatch, JaggedTensor]:
    """Create a grid batch (batch size 1) and jagged point data for testing.

    Args:
        device: Device to create tensors on
        dtype: Data type for tensors
        include_boundary_points: If True, create points both inside and outside the grid
        expand: Number of times to replicate points when creating boundary samples

    Returns:
        tuple: (primal_grid_batch, dual_grid_batch, jagged_points)
    """
    grid, grid_d, points = make_grid_and_point_data(device, dtype, include_boundary_points, expand)
    grid_batch = GridBatch.from_grid(grid)
    grid_batch_d = GridBatch.from_grid(grid_d)
    jagged_points = JaggedTensor(points)
    return grid_batch, grid_batch_d, jagged_points
