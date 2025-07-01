# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Sparse grid batch data structure and operations for FVDB.

This module provides the core GridBatch class for managing sparse voxel grids:

Classes:
- GridBatch: A batch of sparse voxel grids with support for efficient operations

Module-level functions for creating GridBatch objects from various sources:
- gridbatch_from_dense: Create from dense grid dimensions
- gridbatch_from_ijk: Create from explicit voxel coordinates
- gridbatch_from_mesh: Create from triangle meshes
- gridbatch_from_points: Create from point clouds
- gridbatch_from_nearest_voxels_to_points: Create from nearest voxels to points
- load/save: Load and save grid batches to/from .nvdb files

GridBatch supports operations like convolution, pooling, interpolation, ray casting,
mesh extraction, and coordinate transformations on sparse voxel data.
"""

import typing
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import torch

from . import _parse_device_string
from ._Cpp import ConvPackBackend
from ._Cpp import GridBatch as GridBatchCpp
from ._Cpp import JaggedTensor
from .types import (
    GridBatchIndex,
    JaggedTensorOrTensor,
    Vec3d,
    Vec3dBatch,
    Vec3dBatchOrScalar,
    Vec3dOrScalar,
    Vec3i,
    Vec3iBatch,
    Vec3iOrScalar,
    is_GridBatchIndex,
    is_JaggedTensorOrTensor,
    is_Vec3d,
    is_Vec3dBatch,
    is_Vec3dBatchOrScalar,
    is_Vec3dOrScalar,
    is_Vec3i,
    is_Vec3iBatch,
    is_Vec3iOrScalar,
)

if TYPE_CHECKING:
    from .sparse_conv_pack_info import SparseConvPackInfo


class GridBatch:
    """
    A batch of sparse voxel grids with support for efficient operations.

    GridBatch represents a collection of sparse 3D voxel grids that can be processed
    together efficiently on GPU. Each grid in the batch can have different resolutions,
    origins, and voxel sizes. The class provides methods for common operations like
    sampling, convolution, pooling, and other operations.

    The grids are stored in a sparse format where only active (non-empty) voxels are
    allocated, making it memory efficient for representing large volumes with sparse
    occupancy.

    Attributes:
        max_grids_per_batch (int): Maximum number of grids that can be stored in a single batch.
    """

    # Class variable
    max_grids_per_batch: int = GridBatchCpp.max_grids_per_batch

    @overload
    def __init__(self, device: torch.device | str | None = ...) -> None:
        """
        Create a new GridBatch on a specific device:
        >>> grid = GridBatch("cuda")  # string
        >>> grid = GridBatch(torch.device("cuda:0")) # device directly
        >>> grid = GridBatch()  # defaults to CPU

        Args:
            device: The device to create the GridBatch on. Can be a string (e.g., "cuda", "cpu")
                or a torch.device object. If None, defaults to CPU.
        """
        ...

    @overload
    def __init__(self, *, impl: GridBatchCpp) -> None: ...

    def __init__(self, device: torch.device | str | None = None, impl: GridBatchCpp | None = None):
        if impl is not None:
            self._impl = impl
        else:
            if device is None:
                device = torch.device("cpu")
            elif isinstance(device, str):
                device = _parse_device_string(device)
            self._impl = GridBatchCpp(device)

    def avg_pool(
        self,
        pool_factor: Vec3iOrScalar,
        data: JaggedTensorOrTensor,
        stride: Vec3iOrScalar = 0,
        coarse_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Downsample grid data using average pooling.

        Performs average pooling on the voxel data, reducing the resolution by the specified
        pool factor. Each output voxel contains the average of the corresponding input voxels
        within the pooling window.

        Args:
            pool_factor (int or 3-tuple of ints): The factor by which to downsample the grid.
                If an int, the same factor is used for all dimensions.
            data (JaggedTensor or torch.Tensor): The voxel data to pool. Shape should be
                (total_voxels, channels).
            stride (int or 3-tuple of ints): The stride to use when pooling. If 0 (default),
                stride equals pool_factor. If an int, the same stride is used for all dimensions.
            coarse_grid (GridBatch, optional): Pre-allocated coarse grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The pooled data as a JaggedTensor
                - The coarse GridBatch containing the pooled structure
        """

        if not (
            is_Vec3iOrScalar(pool_factor)
            and is_JaggedTensorOrTensor(data)
            and is_Vec3iOrScalar(stride)
            and (coarse_grid is None or isinstance(coarse_grid, GridBatch))
        ):
            raise TypeError(
                f"pool_factor must be a Vec3iOrScalar, but got {type(pool_factor)}, "
                f"data must be a JaggedTensorOrTensor, but got {type(data)}, "
                f"stride must be a Vec3iOrScalar, but got {type(stride)}, "
                f"coarse_grid must be a GridBatch|None, but got {type(coarse_grid)}"
            )

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.avg_pool(pool_factor, data, stride, coarse_grid_impl)
        return result_data, GridBatch(impl=cast(GridBatchCpp, result_grid_impl))

    def bbox_at(self, bi: int) -> torch.Tensor:
        """
        Get the bounding box of a specific grid in the batch.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            torch.Tensor: A tensor of shape (2, 3) containing the minimum and maximum
                coordinates of the bounding box in index space.
        """
        return self._impl.bbox_at(bi)

    def clip(
        self, features: JaggedTensorOrTensor, ijk_min: Vec3iBatch, ijk_max: Vec3iBatch
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Clip the grid to a bounding box and return clipped features.

        Creates a new grid containing only the voxels that fall within the specified
        bounding box range [ijk_min, ijk_max] for each grid in the batch.

        Args:
            features (JaggedTensor or torch.Tensor): The voxel features to clip.
                Shape should be (total_voxels, channels).
            ijk_min (list of 3-tuples): Minimum bounds in index space for each grid.
            ijk_max (list of 3-tuples): Maximum bounds in index space for each grid.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The clipped features as a JaggedTensor
                - A new GridBatch containing only voxels within the bounds
        """
        if not (is_JaggedTensorOrTensor(features) and is_Vec3iBatch(ijk_min) and is_Vec3iBatch(ijk_max)):
            raise TypeError(
                f"features must be a JaggedTensorOrTensor, but got {type(features)}, "
                f"ijk_min must be a Vec3iBatch, but got {type(ijk_min)}, "
                f"ijk_max must be a Vec3iBatch, but got {type(ijk_max)}"
            )

        if isinstance(features, torch.Tensor):
            features = JaggedTensor(features)

        result_features, result_grid_impl = self._impl.clip(features, ijk_min, ijk_max)
        return result_features, GridBatch(impl=result_grid_impl)

    def clipped_grid(
        self,
        ijk_min: Vec3iBatch,
        ijk_max: Vec3iBatch,
    ) -> "GridBatch":
        """
        Return a batch of grids representing the clipped version of this batch of grids.
        Each voxel `[i, j, k]` in the input batch is included in the output if it lies within `ijk_min` and `ijk_max`.

        Args:
            ijk_min (list of int triplets): Index space minimum bound of the clip region.
            ijk_max (list of int triplets): Index space maximum bound of the clip region.

        Returns:
            clipped_grid (GridBatch): A GridBatch representing the clipped version of this grid batch.
        """
        if not (is_Vec3iBatch(ijk_min) and is_Vec3iBatch(ijk_max)):
            raise TypeError(
                f"ijk_min must be a Vec3iBatch, but got {type(ijk_min)}, "
                f"ijk_max must be a Vec3iBatch, but got {type(ijk_max)}"
            )

        return GridBatch(impl=self._impl.clipped_grid(ijk_min, ijk_max))

    def coarsened_grid(self, coarsening_factor: Vec3iOrScalar) -> "GridBatch":
        """
        Return a batch of grids representing the coarsened version of this batch of grids.
        Each voxel `[i, j, k]` in the input batch is included in the output if it lies within `ijk_min` and `ijk_max`.

        Args:
            coarsening_factor (int or 3-tuple of ints): The factor by which to coarsen the grid.

        Returns:
            coarsened_grid (GridBatch): A GridBatch representing the coarsened version of this grid batch.
        """
        if not is_Vec3iOrScalar(coarsening_factor):
            raise TypeError(f"coarsening_factor must be a Vec3iOrScalar, but got {type(coarsening_factor)}")

        return GridBatch(impl=self._impl.coarsened_grid(coarsening_factor))

    def contiguous(self) -> "GridBatch":
        """
        Return a contiguous copy of the grid batch.

        Ensures that the underlying data is stored contiguously in memory,
        which can improve performance for subsequent operations.

        Returns:
            GridBatch: A new GridBatch with contiguous memory layout.
        """
        return GridBatch(impl=self._impl.contiguous())

    def integrate_tsdf(
        self,
        tsdf: JaggedTensorOrTensor,
        weights: JaggedTensorOrTensor,
        depth_images: torch.Tensor,
        projection_matrices: torch.Tensor,
        cam_to_world_matrices: torch.Tensor,
        voxel_truncation_distance: float,
    ) -> tuple["GridBatch", JaggedTensor, JaggedTensor]:
        """
        Integrate depth images into a Truncated Signed Distance Function (TSDF) volume.

        Updates the TSDF values and weights in the voxel grid by integrating new depth
        observations from multiple camera viewpoints. This is commonly used for 3D
        reconstruction from RGB-D sensors.

        Args:
            tsdf (JaggedTensor or torch.Tensor): Current TSDF values for each voxel.
                Shape: (total_voxels, 1).
            weights (JaggedTensor or torch.Tensor): Current integration weights for each voxel.
                Shape: (total_voxels, 1).
            depth_images (torch.Tensor): Depth images from cameras.
                Shape: (batch_size, num_views, height, width).
            projection_matrices (torch.Tensor): Camera projection matrices.
                Shape: (batch_size, num_views, 4, 4).
            cam_to_world_matrices (torch.Tensor): Camera to world transformation matrices.
                Shape: (batch_size, num_views, 4, 4).
            voxel_truncation_distance (float): Maximum distance to truncate TSDF values.

        Returns:
            tuple[GridBatch, JaggedTensor, JaggedTensor]: A tuple containing:
                - Updated GridBatch with potentially expanded voxels
                - Updated TSDF values as JaggedTensor
                - Updated weights as JaggedTensor
        """
        if not (
            is_JaggedTensorOrTensor(tsdf)
            and is_JaggedTensorOrTensor(weights)
            and isinstance(depth_images, torch.Tensor)
            and isinstance(projection_matrices, torch.Tensor)
            and isinstance(cam_to_world_matrices, torch.Tensor)
        ):
            raise TypeError(
                f"tsdf must be a JaggedTensorOrTensor, but got {type(tsdf)}, "
                f"weights must be a JaggedTensorOrTensor, but got {type(weights)}, "
                f"depth_images must be a torch.Tensor, but got {type(depth_images)}, "
                f"projection_matrices must be a torch.Tensor, but got {type(projection_matrices)}, "
                f"cam_to_world_matrices must be a torch.Tensor, but got {type(cam_to_world_matrices)}"
            )

        if isinstance(tsdf, torch.Tensor):
            tsdf = JaggedTensor(tsdf)

        if isinstance(weights, torch.Tensor):
            weights = JaggedTensor(weights)

        result_grid_impl, result_jagged_1, result_jagged_2 = self._impl.integrate_tsdf(
            tsdf,
            weights,
            depth_images,
            projection_matrices,
            cam_to_world_matrices,
            voxel_truncation_distance,
        )

        return GridBatch(impl=result_grid_impl), result_jagged_1, result_jagged_2

    def integrate_tsdf_with_features(
        self,
        tsdf: JaggedTensorOrTensor,
        weights: JaggedTensorOrTensor,
        features: JaggedTensorOrTensor,
        depth_images: torch.Tensor,
        feature_images: torch.Tensor,
        projection_matrices: torch.Tensor,
        cam_to_world_matrices: torch.Tensor,
        voxel_truncation_distance: float,
    ) -> tuple["GridBatch", JaggedTensor, JaggedTensor, JaggedTensor]:
        """
        Integrate depth and feature images into TSDF volume with features.

        Similar to integrate_tsdf but also integrates feature observations (e.g., color)
        along with the depth information. This is useful for colored 3D reconstruction.

        Args:
            tsdf (JaggedTensor or torch.Tensor): Current TSDF values for each voxel.
                Shape: (total_voxels, 1).
            weights (JaggedTensor or torch.Tensor): Current integration weights for each voxel.
                Shape: (total_voxels, 1).
            features (JaggedTensor or torch.Tensor): Current feature values for each voxel.
                Shape: (total_voxels, feature_dim).
            depth_images (torch.Tensor): Depth images from cameras.
                Shape: (batch_size, num_views, height, width).
            feature_images (torch.Tensor): Feature images (e.g., RGB) from cameras.
                Shape: (batch_size, num_views, height, width, feature_dim).
            projection_matrices (torch.Tensor): Camera projection matrices.
                Shape: (batch_size, num_views, 4, 4).
            cam_to_world_matrices (torch.Tensor): Camera to world transformation matrices.
                Shape: (batch_size, num_views, 4, 4).
            voxel_truncation_distance (float): Maximum distance to truncate TSDF values.

        Returns:
            tuple[GridBatch, JaggedTensor, JaggedTensor, JaggedTensor]: A tuple containing:
                - Updated GridBatch with potentially expanded voxels
                - Updated TSDF values as JaggedTensor
                - Updated weights as JaggedTensor
                - Updated features as JaggedTensor
        """
        if not (
            is_JaggedTensorOrTensor(tsdf)
            and is_JaggedTensorOrTensor(weights)
            and is_JaggedTensorOrTensor(features)
            and isinstance(depth_images, torch.Tensor)
            and isinstance(feature_images, torch.Tensor)
            and isinstance(projection_matrices, torch.Tensor)
            and isinstance(cam_to_world_matrices, torch.Tensor)
        ):
            raise TypeError(
                f"tsdf must be a JaggedTensorOrTensor, but got {type(tsdf)}, "
                f"weights must be a JaggedTensorOrTensor, but got {type(weights)}, "
                f"features must be a JaggedTensorOrTensor, but got {type(features)}, "
                f"depth_images must be a torch.Tensor, but got {type(depth_images)}, "
                f"feature_images must be a torch.Tensor, but got {type(feature_images)}, "
                f"projection_matrices must be a torch.Tensor, but got {type(projection_matrices)}, "
                f"cam_to_world_matrices must be a torch.Tensor, but got {type(cam_to_world_matrices)}"
            )

        if isinstance(tsdf, torch.Tensor):
            tsdf = JaggedTensor(tsdf)

        if isinstance(weights, torch.Tensor):
            weights = JaggedTensor(weights)

        if isinstance(features, torch.Tensor):
            features = JaggedTensor(features)

        result_grid_impl, result_jagged_1, result_jagged_2, result_jagged_3 = self._impl.integrate_tsdf_with_features(
            tsdf,
            weights,
            features,
            depth_images,
            feature_images,
            projection_matrices,
            cam_to_world_matrices,
            voxel_truncation_distance,
        )

        return GridBatch(impl=result_grid_impl), result_jagged_1, result_jagged_2, result_jagged_3

    def conv_grid(self, kernel_size: Vec3iOrScalar, stride: Vec3iOrScalar = 0) -> "GridBatch":
        """
        Return a batch of grids representing the convolution of this batch with a given kernel.

        Args:
            kernel_size (int or 3-tuple of ints): The size of the kernel to convolve with.
            stride (int or 3-tuple of ints): The stride to use when convolving.

        Returns:
            conv_grid (GridBatch): A GridBatch representing the convolution of this grid batch.
        """

        if not (is_Vec3iOrScalar(kernel_size) and is_Vec3iOrScalar(stride)):
            raise TypeError(
                f"kernel_size must be a Vec3iOrScalar, but got {type(kernel_size)}, "
                f"stride must be a Vec3iOrScalar, but got {type(stride)}"
            )

        return GridBatch(impl=self._impl.conv_grid(kernel_size, stride))

    def coords_in_active_voxel(self, ijk: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Check if voxel coordinates are in active voxels.

        Args:
            ijk (JaggedTensor or torch.Tensor): Voxel coordinates to check.
                Shape: (num_queries, 3) with integer coordinates.

        Returns:
            JaggedTensor: Boolean mask indicating which coordinates correspond to
                active voxels. Shape: (num_queries,).
        """
        if not is_JaggedTensorOrTensor(ijk):
            raise TypeError(f"ijk must be a JaggedTensorOrTensor, but got {type(ijk)}")

        if isinstance(ijk, torch.Tensor):
            ijk = JaggedTensor(ijk)

        return self._impl.coords_in_active_voxel(ijk)

    def cpu(self) -> "GridBatch":
        """
        Move the grid batch to CPU.

        Returns:
            GridBatch: A new GridBatch on CPU device.
        """
        return GridBatch(impl=self._impl.cpu())

    def cubes_in_grid(
        self, cube_centers: JaggedTensorOrTensor, cube_min: Vec3dOrScalar = 0.0, cube_max: Vec3dOrScalar = 0.0
    ) -> JaggedTensor:
        """
        Check if axis-aligned cubes are fully contained within the grid.

        Tests whether cubes defined by their centers and bounds are completely inside
        the active voxels of the grid.

        Args:
            cube_centers (JaggedTensor or torch.Tensor): Centers of the cubes in world coordinates.
                Shape: (num_cubes, 3).
            cube_min (float or 3-tuple or torch.Tensor): Minimum offsets from center defining cube bounds.
                If scalar, same offset used for all dimensions. Can be per-grid or global.
            cube_max (float or 3-tuple or torch.Tensor): Maximum offsets from center defining cube bounds.
                If scalar, same offset used for all dimensions. Can be per-grid or global.

        Returns:
            JaggedTensor: Boolean mask indicating which cubes are fully contained in the grid.
                Shape: (num_cubes,).
        """
        if not (
            is_JaggedTensorOrTensor(cube_centers)
            and is_Vec3dBatchOrScalar(cube_min)
            and is_Vec3dBatchOrScalar(cube_max)
        ):
            if not is_JaggedTensorOrTensor(cube_centers):
                print(f"cube_centers is a {type(cube_centers)}")

            if not is_Vec3dBatchOrScalar(cube_min):
                print(f"cube_min is a {type(cube_min)}, is: {cube_min}")

            if not is_Vec3dBatchOrScalar(cube_max):
                print(f"cube_max is a {type(cube_max)}, is: {cube_max}")

            raise TypeError(
                f"cube_centers must be a JaggedTensorOrTensor, but got {type(cube_centers)}, "
                f"cube_min must be a Vec3dBatchOrScalar, but got {type(cube_min)}, "
                f"cube_max must be a Vec3dBatchOrScalar, but got {type(cube_max)}"
            )

        if isinstance(cube_centers, torch.Tensor):
            cube_centers = JaggedTensor(cube_centers)

        if isinstance(cube_min, torch.Tensor) and cube_min.ndim > 1:
            cube_min = cube_min.squeeze(0)

        if isinstance(cube_max, torch.Tensor) and cube_max.ndim > 1:
            cube_max = cube_max.squeeze(0)

        return self._impl.cubes_in_grid(cube_centers, cube_min, cube_max)

    def cubes_intersect_grid(
        self, cube_centers: JaggedTensorOrTensor, cube_min: Vec3dBatchOrScalar = 0.0, cube_max: Vec3dBatchOrScalar = 0.0
    ) -> JaggedTensor:
        """
        Check if axis-aligned cubes intersect with the grid.

        Tests whether cubes defined by their centers and bounds have any intersection
        with the active voxels of the grid.

        Args:
            cube_centers (JaggedTensor or torch.Tensor): Centers of the cubes in world coordinates.
                Shape: (num_cubes, 3).
            cube_min (float or 3-tuple or torch.Tensor): Minimum offsets from center defining cube bounds.
                If scalar, same offset used for all dimensions. Can be per-grid or global.
            cube_max (float or 3-tuple or torch.Tensor): Maximum offsets from center defining cube bounds.
                If scalar, same offset used for all dimensions. Can be per-grid or global.

        Returns:
            JaggedTensor: Boolean mask indicating which cubes intersect the grid.
                Shape: (num_cubes,).
        """
        if not (
            is_JaggedTensorOrTensor(cube_centers)
            and is_Vec3dBatchOrScalar(cube_min)
            and is_Vec3dBatchOrScalar(cube_max)
        ):
            raise TypeError(
                f"cube_centers must be a JaggedTensorOrTensor, but got {type(cube_centers)}, "
                f"cube_min must be a Vec3dBatchOrScalar, but got {type(cube_min)}, "
                f"cube_max must be a Vec3dBatchOrScalar, but got {type(cube_max)}"
            )

        if isinstance(cube_centers, torch.Tensor):
            cube_centers = JaggedTensor(cube_centers)

        if isinstance(cube_min, torch.Tensor) and cube_min.ndim > 1:
            cube_min = cube_min.squeeze(0)

        if isinstance(cube_max, torch.Tensor) and cube_max.ndim > 1:
            cube_max = cube_max.squeeze(0)

        return self._impl.cubes_intersect_grid(cube_centers, cube_min, cube_max)  # type: ignore

    def cuda(self) -> "GridBatch":
        """
        Move the grid batch to CUDA device.

        Returns:
            GridBatch: A new GridBatch on CUDA device.
        """
        return GridBatch(impl=self._impl.cuda())

    def cum_voxels_at(self, bi: int) -> int:
        """
        Get the cumulative number of voxels up to and including a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            int: The cumulative number of voxels up to and including grid bi.
        """
        return self._impl.cum_voxels_at(bi)

    def dilated_grid(self, dilation: int) -> "GridBatch":
        """
        Return the grid dilated by a given number of voxels.

        Args:
            dilation (int): The dilation radius in voxels.

        Returns:
            GridBatch: A new GridBatch with dilated active regions.
        """
        return GridBatch(impl=self._impl.dilated_grid(dilation))

    def merged_grid(self, other: "GridBatch") -> "GridBatch":
        """
        Return a grid batch that is the union of this grid batch with another.

        Merges two grid batches by taking the union of their active voxels.
        The grids must have compatible dimensions and transforms.

        Args:
            other (GridBatch): The other grid batch to merge with.

        Returns:
            GridBatch: A new GridBatch containing the union of active voxels from both grids.
        """
        return GridBatch(impl=self._impl.merged_grid(other._impl))

    def pruned_grid(self, mask: JaggedTensorOrTensor) -> "GridBatch":
        """
        Return a pruned grid based on a boolean mask.

        Creates a new grid containing only the voxels where the mask is True.

        Args:
            mask (JaggedTensor or torch.Tensor): Boolean mask for each voxel.
                Shape: (total_voxels,).

        Returns:
            GridBatch: A new GridBatch containing only voxels where mask is True.
        """
        if not is_JaggedTensorOrTensor(mask):
            raise TypeError(f"mask must be a JaggedTensorOrTensor, but got {type(mask)}")

        if isinstance(mask, torch.Tensor):
            mask = JaggedTensor(mask)

        return GridBatch(impl=self._impl.pruned_grid(mask))

    def inject_to(self, dst_grid: "GridBatch", src: JaggedTensorOrTensor, dst: JaggedTensorOrTensor) -> None:
        """
        Inject data from this grid to a destination grid.

        Copies data from voxels in this grid to corresponding voxels in the destination grid.
        This is an in-place operation that modifies the dst tensor.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        Args:
            dst_grid (GridBatch): The destination grid to inject data into.
            src (JaggedTensor or torch.Tensor): Source data from this grid.
                Shape: (total_voxels, channels).
            dst (JaggedTensor or torch.Tensor): Destination data to be modified in-place.
                Shape: (dst_total_voxels, channels).

        Returns:
            None  (this modifies the dst_grid in place)
        """
        if not (is_JaggedTensorOrTensor(src) and is_JaggedTensorOrTensor(dst)):
            raise TypeError(
                f"src must be a JaggedTensorOrTensor, but got {type(src)}, "
                f"dst must be a JaggedTensorOrTensor, but got {type(dst)}"
            )

        if isinstance(src, torch.Tensor):
            src = JaggedTensor(src)

        if isinstance(dst, torch.Tensor):
            dst = JaggedTensor(dst)

        self._impl.inject_to(dst_grid._impl, src, dst)

    def inject_from(self, src_grid: "GridBatch", src: JaggedTensorOrTensor, dst: JaggedTensorOrTensor) -> None:
        """
        Inject data from a source grid into this grid.

        Copies data from voxels in the source grid to corresponding voxels in this grid.
        This is an in-place operation that modifies this grid.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        Args:
            src_grid (GridBatch): The source grid to inject data from.
            src (JaggedTensor or torch.Tensor): Source data from the source grid.
                Shape: (src_total_voxels, channels).
            dst (JaggedTensor or torch.Tensor): Destination data in this grid to be modified in-place.
                Shape: (total_voxels, channels).

        Returns:
            None  (this modifies the dst tensor in place)
        """
        if not (is_JaggedTensorOrTensor(src) and is_JaggedTensorOrTensor(dst)):
            raise TypeError(
                f"src must be a JaggedTensorOrTensor, but got {type(src)}, "
                f"dst must be a JaggedTensorOrTensor, but got {type(dst)}"
            )

        if isinstance(src, torch.Tensor):
            src = JaggedTensor(src)

        if isinstance(dst, torch.Tensor):
            dst = JaggedTensor(dst)

        self._impl.inject_from(src_grid._impl, src, dst)

    def dual_bbox_at(self, bi: int) -> torch.Tensor:
        """
        Get the dual bounding box of a specific grid in the batch.

        The dual grid has voxel centers at the corners of the primal grid voxels.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            torch.Tensor: A tensor of shape (2, 3) containing the minimum and maximum
                coordinates of the dual bounding box in index space.
        """
        return self._impl.dual_bbox_at(bi)

    def dual_grid(self, exclude_border: bool = False) -> "GridBatch":
        """
        Return the dual grid where voxel centers correspond to corners of the primal grid.

        The dual grid is useful for staggered grid discretizations and finite difference operations.

        Args:
            exclude_border (bool): If True, excludes border voxels that would extend beyond
                the primal grid bounds. Default is False.

        Returns:
            GridBatch: A new GridBatch representing the dual grid.
        """
        return GridBatch(impl=self._impl.dual_grid(exclude_border))

    def fill_from_grid(
        self, other_features: JaggedTensorOrTensor, other_grid: "GridBatch", default_value: float = 0.0
    ) -> JaggedTensor:
        """
        Fill voxel features from another grid's features.

        For each voxel in this grid, looks up the corresponding voxel in the other grid
        and copies its features. If a voxel doesn't exist in the other grid, uses the
        default value.

        The copy occurs in "index-space", the grid-to-world transform is not applied.

        Args:
            other_features (JaggedTensor or torch.Tensor): Features from the other grid.
                Shape: (other_total_voxels, channels).
            other_grid (GridBatch): The other grid to copy features from.
            default_value (float): Value to use for voxels that don't exist in the other grid.
                Default is 0.0.

        Returns:
            JaggedTensor: Features for this grid's voxels, copied from the other grid where available.
                Shape: (total_voxels, channels).
        """
        if not (is_JaggedTensorOrTensor(other_features) and isinstance(other_grid, GridBatch)):
            raise TypeError(
                f"other_features must be a JaggedTensorOrTensor, but got {type(other_features)}, "
                f"other_grid must be a GridBatch, but got {type(other_grid)}"
            )

        if isinstance(other_features, torch.Tensor):
            other_features = JaggedTensor(other_features)

        return self._impl.fill_from_grid(
            other_features, other_grid._impl if isinstance(other_grid, GridBatch) else other_grid, default_value
        )

    def grid_to_world(self, ijk: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Convert grid (index) coordinates to world coordinates.

        Transforms voxel indices to their corresponding positions in world space
        using the grid's origin and voxel size.

        Args:
            ijk (JaggedTensor or torch.Tensor): Grid coordinates to convert.
                Shape: (num_points, 3). Can be fractional for interpolation.

        Returns:
            JaggedTensor: World coordinates. Shape: (num_points, 3).
        """
        if not is_JaggedTensorOrTensor(ijk):
            raise TypeError(f"ijk must be a JaggedTensorOrTensor, but got {type(ijk)}")

        if isinstance(ijk, torch.Tensor):
            ijk = JaggedTensor(ijk)

        return self._impl.grid_to_world(ijk)  # type: ignore

    def ijk_to_index(self, ijk: JaggedTensorOrTensor, cumulative: bool = False) -> JaggedTensor:
        """
        Convert voxel coordinates to linear indices.

        Maps 3D voxel coordinates to their corresponding linear indices in the sparse storage.
        Returns -1 for coordinates that don't correspond to active voxels.

        Args:
            ijk (JaggedTensor or torch.Tensor): Voxel coordinates to convert.
                Shape: (num_queries, 3) with integer coordinates.
            cumulative (bool): If True, returns cumulative indices across the entire batch.
                If False, returns per-grid indices. Default is False.

        Returns:
            JaggedTensor: Linear indices for each coordinate, or -1 if not active.
                Shape: (num_queries,).
        """
        if not is_JaggedTensorOrTensor(ijk):
            raise TypeError(f"ijk must be a JaggedTensorOrTensor, but got {type(ijk)}")

        if isinstance(ijk, torch.Tensor):
            ijk = JaggedTensor(ijk)

        return self._impl.ijk_to_index(ijk, cumulative)  # type: ignore

    def ijk_to_inv_index(self, ijk: JaggedTensorOrTensor, cumulative: bool = False) -> JaggedTensor:
        """
        Get inverse permutation for ijk_to_index.

        Args:
            ijk (JaggedTensor or torch.Tensor): Voxel coordinates to convert.
                Shape: (num_queries, 3) with integer coordinates.
            cumulative (bool): If True, returns cumulative indices across the entire batch.
                If False, returns per-grid indices. Default is False.

        Returns:
            JaggedTensor: Inverse permutation for ijk_to_index.
                Shape: (num_queries,).
        """
        if not is_JaggedTensorOrTensor(ijk):
            raise TypeError(f"ijk must be a JaggedTensorOrTensor, but got {type(ijk)}")

        if isinstance(ijk, torch.Tensor):
            ijk = JaggedTensor(ijk)

        return self._impl.ijk_to_inv_index(ijk, cumulative)  # type: ignore

    def is_contiguous(self) -> bool:
        """
        Check if the grid batch data is stored contiguously in memory.

        Returns:
            bool: True if the data is contiguous, False otherwise.
        """
        return self._impl.is_contiguous()

    def is_same(self, other: "GridBatch") -> bool:
        """
        Check if two grid batches have the same structure.

        Compares the voxel structure, dimensions, and origins of two grid batches.

        Args:
            other (GridBatch): The other grid batch to compare with.

        Returns:
            bool: True if the grids have identical structure, False otherwise.
        """
        return self._impl.is_same(other._impl)

    def jagged_like(self, data: torch.Tensor) -> JaggedTensor:
        """
        Create a JaggedTensor with the same jagged structure as this grid batch.

        Useful for creating feature tensors that match the grid's voxel layout.

        Args:
            data (torch.Tensor): Dense data to convert to jagged format.
                Shape: (total_voxels, channels).

        Returns:
            JaggedTensor: Data in jagged format matching the grid structure.
        """
        return self._impl.jagged_like(data)

    def marching_cubes(
        self, field: JaggedTensorOrTensor, level: float = 0.0
    ) -> tuple[JaggedTensor, JaggedTensor, JaggedTensor]:
        """
        Extract isosurface mesh using the marching cubes algorithm.

        Generates a triangle mesh representing the isosurface at the specified level
        from a scalar field defined on the voxels.

        Args:
            field (JaggedTensor or torch.Tensor): Scalar field values at each voxel.
                Shape: (total_voxels, 1).
            level (float): The isovalue to extract the surface at. Default is 0.0.

        Returns:
            tuple[JaggedTensor, JaggedTensor, JaggedTensor]: A tuple containing:
                - Vertex positions of the mesh. Shape: (num_vertices, 3).
                - Triangle face indices. Shape: (num_faces, 3).
                - Vertex normals (computed from gradients). Shape: (num_vertices, 3).
        """
        if not is_JaggedTensorOrTensor(field):
            raise TypeError(f"field must be a JaggedTensorOrTensor, but got {type(field)}")

        if isinstance(field, torch.Tensor):
            field = JaggedTensor(field)

        return self._impl.marching_cubes(field, level)

    def max_pool(
        self,
        pool_factor: Vec3iOrScalar,
        data: JaggedTensorOrTensor,
        stride: Vec3iOrScalar = 0,
        coarse_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Downsample grid data using max pooling.

        Performs max pooling on the voxel data, reducing the resolution by the specified
        pool factor. Each output voxel contains the maximum of the corresponding input voxels
        within the pooling window.

        Args:
            pool_factor (int or 3-tuple of ints): The factor by which to downsample the grid.
                If an int, the same factor is used for all dimensions.
            data (JaggedTensor or torch.Tensor): The voxel data to pool. Shape should be
                (total_voxels, channels).
            stride (int or 3-tuple of ints): The stride to use when pooling. If 0 (default),
                stride equals pool_factor. If an int, the same stride is used for all dimensions.
            coarse_grid (GridBatch, optional): Pre-allocated coarse grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The pooled data as a JaggedTensor
                - The coarse GridBatch containing the pooled structure
        """
        if not (
            is_Vec3iOrScalar(pool_factor)
            and is_JaggedTensorOrTensor(data)
            and is_Vec3iOrScalar(stride)
            and (coarse_grid is None or isinstance(coarse_grid, GridBatch))
        ):
            raise TypeError(
                f"pool_factor must be a Vec3iOrScalar, but got {type(pool_factor)}, "
                f"data must be a JaggedTensorOrTensor, but got {type(data)}, "
                f"stride must be a Vec3iOrScalar, but got {type(stride)}, "
                f"coarse_grid must be a GridBatch|None, but got {type(coarse_grid)}"
            )

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        coarse_grid_impl = coarse_grid._impl if coarse_grid else None

        result_data, result_grid_impl = self._impl.max_pool(pool_factor, data, stride, coarse_grid_impl)
        return result_data, GridBatch(impl=result_grid_impl)

    def neighbor_indexes(self, ijk: JaggedTensorOrTensor, extent: int, bitshift: int = 0) -> JaggedTensor:
        """
        Get indices of neighbors in N-ring neighborhood.

        Finds the linear indices of all voxels within a specified neighborhood ring
        around the given voxel coordinates.

        Args:
            ijk (JaggedTensor or torch.Tensor): Voxel coordinates to find neighbors for.
                Shape: (num_queries, 3) with integer coordinates.
            extent (int): Size of the neighborhood ring (N-ring).
            bitshift (int): Bit shift value for encoding. Default is 0.

        Returns:
            JaggedTensor: Linear indices of neighboring voxels.
        """
        if not is_JaggedTensorOrTensor(ijk):
            raise TypeError(f"ijk must be a JaggedTensorOrTensor, but got {type(ijk)}")

        if isinstance(ijk, torch.Tensor):
            ijk = JaggedTensor(ijk)

        return self._impl.neighbor_indexes(ijk, extent, bitshift)  # type: ignore

    def num_voxels_at(self, bi: int) -> int:
        """
        Get the number of active voxels in a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            int: Number of active voxels in the specified grid.
        """
        return self._impl.num_voxels_at(bi)

    def origin_at(self, bi: int) -> torch.Tensor:
        """
        Get the world-space origin of a specific grid.

        Args:
            bi (int): The batch index of the grid.

        Returns:
            torch.Tensor: The origin coordinates in world space. Shape: (3,).
        """
        return self._impl.origin_at(bi)

    def points_in_active_voxel(self, points: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Check if world-space points are located within active voxels.

        Tests whether the given points fall within voxels that are active in the grid.

        Args:
            points (JaggedTensor or torch.Tensor): World-space points to test.
                Shape: (num_points, 3).

        Returns:
            JaggedTensor: Boolean mask indicating which points are in active voxels.
                Shape: (num_points,).
        """
        if not is_JaggedTensorOrTensor(points):
            raise TypeError(f"points must be a JaggedTensorOrTensor, but got {type(points)}")

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        return self._impl.points_in_active_voxel(points)  # type: ignore

    def ray_implicit_intersection(
        self,
        ray_origins: JaggedTensorOrTensor,
        ray_directions: JaggedTensorOrTensor,
        grid_scalars: JaggedTensorOrTensor,
        eps: float = 0.0,
    ) -> JaggedTensor:
        """
        Find ray intersections with implicit surface defined by grid scalars.

        Computes intersection points between rays and an implicit surface defined by
        scalar values stored in the grid voxels (e.g., signed distance function).

        Args:
            ray_origins (JaggedTensor or torch.Tensor): Starting points of rays in world space.
                Shape: (num_rays, 3).
            ray_directions (JaggedTensor or torch.Tensor): Direction vectors of rays.
                Shape: (num_rays, 3). Should be normalized.
            grid_scalars (JaggedTensor or torch.Tensor): Scalar field values at each voxel.
                Shape: (total_voxels, 1).
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            JaggedTensor: Intersection information for each ray.
        """
        if (
            not is_JaggedTensorOrTensor(ray_origins)
            or not is_JaggedTensorOrTensor(ray_directions)
            or not is_JaggedTensorOrTensor(grid_scalars)
        ):
            raise TypeError(
                "ray_origins must be a JaggedTensorOrTensor, ray_directions must be a JaggedTensorOrTensor, "
                f"and grid_scalars must be a JaggedTensorOrTensor, but got {type(ray_origins)}, "
                f"{type(ray_directions)}, {type(grid_scalars)}"
            )

        if isinstance(ray_origins, torch.Tensor):
            ray_origins = JaggedTensor(ray_origins)

        if isinstance(ray_directions, torch.Tensor):
            ray_directions = JaggedTensor(ray_directions)

        if isinstance(grid_scalars, torch.Tensor):
            grid_scalars = JaggedTensor(grid_scalars)

        return self._impl.ray_implicit_intersection(ray_origins, ray_directions, grid_scalars, eps)

    def read_from_dense(self, dense_data: torch.Tensor, dense_origins: Vec3i | None = None) -> JaggedTensor:
        """
        Read values from a dense tensor into sparse grid structure.

        Extracts values from a dense tensor at locations corresponding to active voxels
        in the sparse grid. Useful for converting dense data to sparse representation.

        Args:
            dense_data (torch.Tensor): Dense tensor to read from.
                Shape: (batch_size, channels, depth, height, width) or
                       (batch_size, depth, height, width, channels).
            dense_origins (3-tuple of ints, optional): Origin of the dense tensor in
                grid index space. Default is (0, 0, 0).

        Returns:
            JaggedTensor: Values from the dense tensor at active voxel locations.
                Shape: (total_voxels, channels).
        """

        if not isinstance(dense_data, torch.Tensor):
            raise TypeError(f"dense_data must be a torch.Tensor, but got {type(dense_data)}")

        if dense_origins is None:
            dense_origins = torch.zeros(3, dtype=torch.int32)

        if not is_Vec3i(dense_origins):
            raise TypeError(f"dense_origins must be a Vec3i, but got {type(dense_origins)}")

        return self._impl.read_from_dense(dense_data, dense_origins)  # type: ignore

    def sample_bezier(self, points: JaggedTensorOrTensor, voxel_data: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Sample voxel features at arbitrary points using Bézier interpolation.

        Interpolates voxel data at continuous world-space positions using cubic Bézier
        interpolation. This provides smoother interpolation than trilinear but is more
        computationally expensive.

        Args:
            points (JaggedTensor or torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (JaggedTensor or torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            JaggedTensor: Interpolated features at each point.
                Shape: (num_points, channels).
        """
        if not (is_JaggedTensorOrTensor(points) and is_JaggedTensorOrTensor(voxel_data)):
            raise TypeError(
                f"points must be a JaggedTensorOrTensor, but got {type(points)}, "
                f"voxel_data must be a JaggedTensorOrTensor, but got {type(voxel_data)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        if isinstance(voxel_data, torch.Tensor):
            voxel_data = JaggedTensor(voxel_data)

        return self._impl.sample_bezier(points, voxel_data)

    def sample_bezier_with_grad(
        self, points: JaggedTensorOrTensor, voxel_data: JaggedTensorOrTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Sample voxel features and their gradients using Bézier interpolation.

        Similar to sample_bezier but also computes the spatial gradient of the
        interpolated values with respect to the world-space coordinates.

        Args:
            points (JaggedTensor or torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (JaggedTensor or torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            tuple[JaggedTensor, JaggedTensor]: A tuple containing:
                - Interpolated features at each point. Shape: (num_points, channels).
                - Gradients of features with respect to world coordinates.
                  Shape: (num_points, 3, channels).
        """
        if not (is_JaggedTensorOrTensor(points) and is_JaggedTensorOrTensor(voxel_data)):
            raise TypeError(
                f"points must be a JaggedTensorOrTensor, but got {type(points)}, "
                f"voxel_data must be a JaggedTensorOrTensor, but got {type(voxel_data)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        if isinstance(voxel_data, torch.Tensor):
            voxel_data = JaggedTensor(voxel_data)

        return self._impl.sample_bezier_with_grad(points, voxel_data)

    def sample_trilinear(self, points: JaggedTensorOrTensor, voxel_data: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Sample voxel features at arbitrary points using trilinear interpolation.

        Interpolates voxel data at continuous world-space positions using trilinear
        interpolation from the 8 nearest voxels. Points outside the grid return zero.

        Args:
            points (JaggedTensor or torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (JaggedTensor or torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            JaggedTensor: Interpolated features at each point.
                Shape: (num_points, channels).
        """
        if not (is_JaggedTensorOrTensor(points) and is_JaggedTensorOrTensor(voxel_data)):
            raise TypeError(
                f"points must be a JaggedTensorOrTensor, but got {type(points)}, "
                f"voxel_data must be a JaggedTensorOrTensor, but got {type(voxel_data)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        if isinstance(voxel_data, torch.Tensor):
            voxel_data = JaggedTensor(voxel_data)

        return self._impl.sample_trilinear(points, voxel_data)

    def sample_trilinear_with_grad(
        self, points: JaggedTensorOrTensor, voxel_data: JaggedTensorOrTensor
    ) -> tuple[JaggedTensor, JaggedTensor]:
        """
        Sample voxel features and their gradients using trilinear interpolation.

        Similar to sample_trilinear but also computes the spatial gradient of the
        interpolated values with respect to the world-space coordinates.

        Args:
            points (JaggedTensor or torch.Tensor): World-space points to sample at.
                Shape: (num_points, 3).
            voxel_data (JaggedTensor or torch.Tensor): Features stored at each voxel.
                Shape: (total_voxels, channels).

        Returns:
            tuple[JaggedTensor, JaggedTensor]: A tuple containing:
                - Interpolated features at each point. Shape: (num_points, channels).
                - Gradients of features with respect to world coordinates.
                  Shape: (num_points, 3, channels).
        """
        if not (is_JaggedTensorOrTensor(points) and is_JaggedTensorOrTensor(voxel_data)):
            raise TypeError(
                f"points must be a JaggedTensorOrTensor, but got {type(points)}, "
                f"voxel_data must be a JaggedTensorOrTensor, but got {type(voxel_data)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        if isinstance(voxel_data, torch.Tensor):
            voxel_data = JaggedTensor(voxel_data)

        return self._impl.sample_trilinear_with_grad(points, voxel_data)

    def segments_along_rays(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, max_segments: int, eps: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enumerate segments along rays.

        Args:
            ray_origins (torch.Tensor): Origin of each ray.
                Shape: (num_rays, 3).
            ray_directions (torch.Tensor): Direction of each ray.
                Shape: (num_rays, 3).
            max_segments (int): Maximum number of segments to enumerate.
            eps (float): Small epsilon value to avoid numerical issues.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Segment origins. Shape: (num_segments, 3).
        """
        return self._impl.segments_along_rays(ray_origins, ray_directions, max_segments, eps)

    def set_from_dense_grid(
        self,
        num_grids: int,
        dense_dims: Vec3i,
        ijk_min: Vec3i = torch.zeros(3, dtype=torch.int32),
        voxel_sizes: Vec3dBatchOrScalar = 1.0,
        origins: Vec3dBatch = torch.zeros(3),
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        A dense grid has a voxel for every coordinate in an axis-aligned box of Vec3,
        which can in turn be mapped to a world-space box.

        for each grid in the batch, the dense grid is defined by:
        - dense_dims: the size of the dense grid (shape [3,] = [W, H, D])
        - ijk_min: the minimum voxel index for each grid in the batch (Vec3i)
        - voxel_sizes: the world-space size of each voxel (Vec3d or scalar)
        - origins: the world-space coordinate of the 0,0,0 voxel of each grid
        - mask: indicates which voxels are "active" in the resulting grid.

        The voxel sizes and world space origins can be per-grid or per-batch.
        The ijk-min and sizes are the same for all grids in the batch.
        The mask is the same for all grids in the batch.

        Args:
            num_grids (int): Number of grids to populate.
            dense_dims (Vec3i): Dimensions of the dense grid.
            ijk_min (Vec3i): Minimum voxel index for each grid.
            voxel_sizes (Vec3dBatchOrScalar): World space size of each voxel.
            origins (Vec3dBatch): World space coordinate of the 0,0,0 voxel of each grid.
            mask (torch.Tensor | None): Mask to apply to the grid.
                Shape: (num_grids,).

        Returns:
            None  (this modifies the grid in place)
        """
        if is_Vec3i(dense_dims) and is_Vec3i(ijk_min) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins):
            self._impl.set_from_dense_grid(num_grids, dense_dims, ijk_min, voxel_sizes, origins, mask)  # type: ignore
        else:
            raise TypeError(
                "Unsupported types for set_from_dense_grid(): "
                f"{type(dense_dims)}, {type(ijk_min)}, {type(voxel_sizes)}, {type(origins)}"
            )

    def set_from_ijk(
        self,
        ijk: JaggedTensorOrTensor,
        voxel_sizes: Vec3dBatchOrScalar = 1.0,
        origins: Vec3dBatch = torch.zeros(3),
    ) -> None:
        """
        Populate grid from voxel coordinates.

        Args:
            ijk (JaggedTensor or torch.Tensor): Voxel coordinates to populate.
                Shape: (num_grids, 3) with integer coordinates.
            voxel_sizes (Vec3dBatchOrScalar): Size of each voxel.
            origins (Vec3dBatch): Origin of each grid.

        Returns:
            None  (this modifies the grid in place)
        """
        if not (is_JaggedTensorOrTensor(ijk) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins)):
            raise TypeError(f"Unsupported types for set_from_ijk(): {type(ijk)}, {type(voxel_sizes)}, {type(origins)}")

        if isinstance(ijk, torch.Tensor):
            ijk = JaggedTensor(ijk)

        self._impl.set_from_ijk(ijk, voxel_sizes, origins)  # type: ignore

    def set_from_mesh(
        self,
        mesh_vertices: JaggedTensorOrTensor,
        mesh_faces: JaggedTensorOrTensor,
        voxel_sizes: Vec3dBatchOrScalar = 1.0,
        origins: Vec3dBatch = torch.zeros(3),
    ) -> None:
        """
        Populate grid from triangle mesh.

        Args:
            mesh_vertices (JaggedTensor or torch.Tensor): Vertices of the mesh.
                Shape: (num_vertices, 3).
            mesh_faces (JaggedTensor or torch.Tensor): Faces of the mesh.
                Shape: (num_faces, 3).
            voxel_sizes (Vec3dBatchOrScalar): Size of each voxel.
            origins (Vec3dBatch): Origin of each grid.

        Returns:
            None  (this modifies the grid in place)
        """
        if not (
            is_JaggedTensorOrTensor(mesh_vertices)
            and is_JaggedTensorOrTensor(mesh_faces)
            and is_Vec3dBatchOrScalar(voxel_sizes)
            and is_Vec3dBatch(origins)
        ):
            raise TypeError(
                "Unsupported types for set_from_mesh(): "
                f"{type(mesh_vertices)}, {type(mesh_faces)}, {type(voxel_sizes)}, {type(origins)}"
            )

        if isinstance(mesh_vertices, torch.Tensor):
            mesh_vertices = JaggedTensor(mesh_vertices)

        if isinstance(mesh_faces, torch.Tensor):
            mesh_faces = JaggedTensor(mesh_faces)

        self._impl.set_from_mesh(mesh_vertices, mesh_faces, voxel_sizes, origins)  # type: ignore

    def set_from_nearest_voxels_to_points(
        self,
        points: JaggedTensorOrTensor,
        voxel_sizes: Vec3dBatchOrScalar = 1.0,
        origins: Vec3dBatch = torch.zeros(3),
    ) -> None:
        """
        Populate grid from nearest voxels to points.

        Args:
            points (JaggedTensor or torch.Tensor): Points to populate the grid from.
                Shape: (num_points, 3).
            voxel_sizes (Vec3dBatchOrScalar): Size of each voxel.
            origins (Vec3dBatch): Origin of each grid.

        Returns:
            None  (this modifies the grid in place)
        """
        if not (is_JaggedTensorOrTensor(points) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins)):
            raise TypeError(
                "Unsupported types for set_from_nearest_voxels_to_points(): "
                f"{type(points)}, {type(voxel_sizes)}, {type(origins)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        self._impl.set_from_nearest_voxels_to_points(points, voxel_sizes, origins)  # type: ignore

    def set_from_points(
        self,
        points: JaggedTensorOrTensor,
        voxel_sizes: Vec3dBatchOrScalar = 1.0,
        origins: Vec3dBatch = torch.zeros(3),
    ) -> None:
        """
        Populate grid from point cloud.

        Args:
            points (JaggedTensor or torch.Tensor): Points to populate the grid from.
                Shape: (num_points, 3).
            voxel_sizes (Vec3dBatchOrScalar): Size of each voxel.
            origins (Vec3dBatch): Origin of each grid.

        Returns:
            None  (this modifies the grid in place)
        """
        if not (is_JaggedTensorOrTensor(points) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins)):
            raise TypeError(
                f"Unsupported types for set_from_points(): {type(points)}, {type(voxel_sizes)}, {type(origins)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        self._impl.set_from_points(points, voxel_sizes, origins)  # type: ignore

    def set_global_origin(self, origin: Vec3d) -> None:
        """
        Set the voxel origin of all grids.

        Args:
            origin (Vec3d): Origin of the grid.

        Returns:
            None  (this modifies the grid in place)
        """
        self._impl.set_global_origin(origin)

    def set_global_voxel_size(self, voxel_size: Vec3dOrScalar) -> None:
        """
        Set the voxel size of all grids.

        Args:
            voxel_size (Vec3dOrScalar): Size of each voxel.

        Returns:
            None  (this modifies the grid in place)
        """
        self._impl.set_global_voxel_size(voxel_size)

    def sparse_conv_halo(self, input: JaggedTensorOrTensor, weight: torch.Tensor, variant: int = 8) -> JaggedTensor:
        """
        Perform sparse convolution with halo exchange optimization.

        Applies sparse convolution using halo exchange to efficiently handle boundary
        conditions in distributed or multi-block sparse grids.

        Args:
            input (JaggedTensor or torch.Tensor): Input features for each voxel.
                Shape: (total_voxels, in_channels).
            weight (torch.Tensor): Convolution weights.
            variant (int): Variant of the halo implementation to use. Default is 8.

        Returns:
            JaggedTensor: Output features after convolution.
        """
        if not (is_JaggedTensorOrTensor(input) and is_JaggedTensorOrTensor(weight)):
            raise TypeError(
                f"input must be a JaggedTensorOrTensor, but got {type(input)}, "
                f"weight must be a Tensor, but got {type(weight)}"
            )

        if isinstance(input, torch.Tensor):
            input = JaggedTensor(input)

        return self._impl.sparse_conv_halo(input, weight, variant)

    def sparse_conv_kernel_map(
        self, kernel_size: Vec3iOrScalar, stride: Vec3iOrScalar, target_grid: "GridBatch | None" = None
    ) -> tuple["SparseConvPackInfo", "GridBatch"]:
        """
        Map sparse convolution kernel to target grid.

        Args:
            kernel_size (Vec3iOrScalar): Size of the convolution kernel.
            stride (Vec3iOrScalar): Stride of the convolution.
            target_grid (GridBatch | None): Target grid to map the kernel to.
                If None, the kernel is mapped to the current grid.

        Returns:
            tuple[SparseConvPackInfo, GridBatch]: A tuple containing:
                - SparseConvPackInfo: Information about the sparse convolution kernel.
                - GridBatch: The target grid.
        """
        # Import here to avoid circular dependency
        from .sparse_conv_pack_info import SparseConvPackInfo

        if not is_Vec3iOrScalar(kernel_size) or not is_Vec3iOrScalar(stride):
            raise TypeError(
                f"kernel_size and stride must be of type Vec3iOrScalar, but got {type(kernel_size)} and {type(stride)}"
            )
        if target_grid is not None:
            if not isinstance(target_grid, GridBatch):
                raise TypeError(f"target_grid must be a GridBatch, but got {type(target_grid)}")

        target_impl = target_grid._impl if target_grid is not None else None

        sparse_impl, grid_impl = self._impl.sparse_conv_kernel_map(kernel_size, stride, target_impl)  # type: ignore
        return (SparseConvPackInfo(impl=sparse_impl), GridBatch(impl=grid_impl))

    def splat_bezier(self, points: JaggedTensorOrTensor, points_data: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Splat point features onto voxels using Bézier interpolation.

        Distributes features from point locations onto the surrounding voxels using
        cubic Bézier interpolation weights. This provides smoother distribution than
        trilinear splatting but is more computationally expensive.

        Args:
            points (JaggedTensor or torch.Tensor): World-space positions of points.
                Shape: (num_points, 3).
            points_data (JaggedTensor or torch.Tensor): Features to splat from each point.
                Shape: (num_points, channels).

        Returns:
            JaggedTensor: Accumulated features at each voxel after splatting.
                Shape: (total_voxels, channels).
        """
        if not (is_JaggedTensorOrTensor(points) and is_JaggedTensorOrTensor(points_data)):
            raise TypeError(
                f"points must be a JaggedTensorOrTensor, but got {type(points)}, "
                f"points_data must be a JaggedTensorOrTensor, but got {type(points_data)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        if isinstance(points_data, torch.Tensor):
            points_data = JaggedTensor(points_data)

        return self._impl.splat_bezier(points, points_data)

    def splat_trilinear(self, points: JaggedTensorOrTensor, points_data: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Splat point features onto voxels using trilinear interpolation.

        Distributes features from point locations onto the surrounding 8 voxels using
        trilinear interpolation weights. This is the standard method for converting
        point-based data to voxel grids.

        Args:
            points (JaggedTensor or torch.Tensor): World-space positions of points.
                Shape: (num_points, 3).
            points_data (JaggedTensor or torch.Tensor): Features to splat from each point.
                Shape: (num_points, channels).

        Returns:
            JaggedTensor: Accumulated features at each voxel after splatting.
                Shape: (total_voxels, channels).
        """
        if not (is_JaggedTensorOrTensor(points) and is_JaggedTensorOrTensor(points_data)):
            raise TypeError(
                f"points must be a JaggedTensorOrTensor, but got {type(points)}, "
                f"points_data must be a JaggedTensorOrTensor, but got {type(points_data)}"
            )

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        if isinstance(points_data, torch.Tensor):
            points_data = JaggedTensor(points_data)

        return self._impl.splat_trilinear(points, points_data)

    def subdivide(
        self,
        subdiv_factor: Vec3iOrScalar,
        data: JaggedTensorOrTensor,
        mask: JaggedTensorOrTensor | None = None,
        fine_grid: "GridBatch | None" = None,
    ) -> tuple[JaggedTensor, "GridBatch"]:
        """
        Subdivide grid using nearest neighbor interpolation.

        Increases the resolution of the grid by the specified subdivision factor,
        filling in new voxels using nearest neighbor interpolation of the existing data.

        Args:
            subdiv_factor (int or 3-tuple of ints): Factor by which to subdivide the grid.
                If an int, the same factor is used for all dimensions.
            data (JaggedTensor or torch.Tensor): Voxel data to subdivide.
                Shape: (total_voxels, channels).
            mask (JaggedTensor or torch.Tensor, optional): Boolean mask indicating which
                voxels to subdivide. If None, all voxels are subdivided.
            fine_grid (GridBatch, optional): Pre-allocated fine grid to use for output.
                If None, a new grid is created.

        Returns:
            tuple[JaggedTensor, GridBatch]: A tuple containing:
                - The subdivided data as a JaggedTensor
                - The fine GridBatch containing the subdivided structure
        """

        if not (
            is_Vec3iOrScalar(subdiv_factor)
            and is_JaggedTensorOrTensor(data)
            and (mask is None or is_JaggedTensorOrTensor(mask))
            and (fine_grid is None or isinstance(fine_grid, GridBatch))
        ):
            raise TypeError(
                f"subdiv_factor must be a Vec3iOrScalar, but got {type(subdiv_factor)}, "
                f"data must be a JaggedTensorOrTensor, but got {type(data)}, "
                f"mask must be a JaggedTensorOrTensor|None, but got {type(mask)}, "
                f"fine_grid must be a GridBatch|None, but got {type(fine_grid)}"
            )

        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = JaggedTensor(mask)

        fine_grid_impl = fine_grid._impl if fine_grid else None

        result_data, result_grid_impl = self._impl.subdivide(subdiv_factor, data, mask, fine_grid_impl)  # type: ignore
        return result_data, GridBatch(impl=result_grid_impl)

    def subdivided_grid(self, subdiv_factor: "Vec3iOrScalar", mask: JaggedTensorOrTensor | None = None) -> "GridBatch":
        """
        Return a subdivided version of the grid structure.

        Creates a new grid with higher resolution by subdividing existing voxels.
        Only the grid structure is returned, not the data.

        Args:
            subdiv_factor (int or 3-tuple of ints): Factor by which to subdivide the grid.
                If an int, the same factor is used for all dimensions.
            mask (JaggedTensor or torch.Tensor, optional): Boolean mask indicating which
                voxels to subdivide. If None, all voxels are subdivided.

        Returns:
            GridBatch: A new GridBatch with subdivided structure.
        """

        if not (is_Vec3iOrScalar(subdiv_factor) and (mask is None or is_JaggedTensorOrTensor(mask))):
            raise TypeError(
                f"subdiv_factor must be a Vec3iOrScalar, but got {type(subdiv_factor)}, "
                f"mask must be a JaggedTensorOrTensor, but got {type(mask)}"
            )

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = JaggedTensor(mask)

        return GridBatch(impl=self._impl.subdivided_grid(subdiv_factor, mask=mask))  # type: ignore

    def to(self, target: "str | torch.device | torch.Tensor | JaggedTensor | GridBatch") -> "GridBatch":
        """
        Move grid batch to a target device or match device of target object.

        Args:
            target: Target to determine device. Can be:
                - str: Device string (e.g., "cuda", "cpu")
                - torch.device: PyTorch device object
                - torch.Tensor: Match device of this tensor
                - JaggedTensor: Match device of this JaggedTensor
                - GridBatch: Match device of this GridBatch

        Returns:
            GridBatch: A new GridBatch on the target device.
        """
        if isinstance(target, str):
            device = _parse_device_string(target)
            return GridBatch(impl=self._impl.to(device))
        elif isinstance(target, torch.device):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, torch.Tensor):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, JaggedTensor):
            return GridBatch(impl=self._impl.to(target))
        elif isinstance(target, GridBatch):
            return GridBatch(impl=self._impl.to(target._impl))
        else:
            raise TypeError(f"Unsupported type for to(): {type(target)}")

    def uniform_ray_samples(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        t_min: float,
        t_max: float,
        step_size: float,
        cone_angle: float = 0.0,
        include_end_segments: bool = True,
        return_midpoints: bool = False,
        eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate uniform samples along rays within the grid.

        Creates sample points at regular intervals along rays, but only for segments
        that intersect with active voxels. Useful for volume rendering and ray marching.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space.
                Shape: (batch_size, num_rays, 3).
            ray_directions (torch.Tensor): Direction vectors of rays (should be normalized).
                Shape: (batch_size, num_rays, 3).
            t_min (float): Minimum distance along rays to start sampling.
            t_max (float): Maximum distance along rays to stop sampling.
            step_size (float): Distance between samples along each ray.
            cone_angle (float): Cone angle for cone tracing (in radians). Default is 0.0.
            include_end_segments (bool): Whether to include partial segments at ray ends.
                Default is True.
            return_midpoints (bool): Whether to return segment midpoints instead of start points.
                Default is False.
            eps (float): Epsilon value for numerical stability. Default is 0.0.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Sample positions in world space. Shape: (total_samples, 3).
                - Ray indices for each sample. Shape: (total_samples,).
                - T values (distances) for each sample. Shape: (total_samples,).
                - Segment lengths for each sample. Shape: (total_samples,).
        """
        return self._impl.uniform_ray_samples(
            ray_origins,
            ray_directions,
            t_min,
            t_max,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )

    def voxel_size_at(self, bi: int) -> torch.Tensor:
        """
        Get voxel size at a specific grid index.

        Args:
            bi (int): Grid index.

        Returns:
            torch.Tensor: Voxel size at the specified grid index.
                Shape: (3,).
        """
        return self._impl.voxel_size_at(bi)

    def voxels_along_rays(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_voxels: int,
        eps: float = 0.0,
        return_ijk: bool = True,
        cumulative: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enumerate voxels intersected by rays.

        Finds all active voxels that are intersected by the given rays using a
        DDA (Digital Differential Analyzer) algorithm.

        Args:
            ray_origins (torch.Tensor): Starting points of rays in world space.
                Shape: (batch_size, num_rays, 3).
            ray_directions (torch.Tensor): Direction vectors of rays (should be normalized).
                Shape: (batch_size, num_rays, 3).
            max_voxels (int): Maximum number of voxels to return per ray.
            eps (float): Epsilon value for numerical stability. Default is 0.0.
            return_ijk (bool): Whether to return voxel indices. If False, returns
                linear indices instead. Default is True.
            cumulative (bool): Whether to return cumulative indices across the batch.
                Default is False.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Voxel indices or positions. Shape depends on return_ijk.
                - Ray indices for each voxel. Shape: (total_intersections,).
                - Entry distances for each voxel. Shape: (total_intersections,).
                - Exit distances for each voxel. Shape: (total_intersections,).
        """
        return self._impl.voxels_along_rays(ray_origins, ray_directions, max_voxels, eps, return_ijk, cumulative)

    def world_to_grid(self, points: JaggedTensorOrTensor) -> JaggedTensor:
        """
        Convert world coordinates to grid (index) coordinates.

        Transforms positions in world space to their corresponding voxel indices
        using the grid's origin and voxel size. The resulting coordinates can be
        fractional for use in interpolation.

        Args:
            points (JaggedTensor or torch.Tensor): World-space positions to convert.
                Shape: (num_points, 3).

        Returns:
            JaggedTensor: Grid coordinates. Shape: (num_points, 3).
                Can contain fractional values.
        """
        if not is_JaggedTensorOrTensor(points):
            raise TypeError(f"points must be a JaggedTensorOrTensor, but got {type(points)}")

        if isinstance(points, torch.Tensor):
            points = JaggedTensor(points)

        return self._impl.world_to_grid(points)  # type: ignore

    def write_to_dense(
        self, sparse_data: JaggedTensorOrTensor, min_coord: Vec3iBatch | None = None, grid_size: Vec3i | None = None
    ) -> torch.Tensor:
        """
        Write sparse voxel data to a dense tensor.

        Creates a dense tensor and fills it with values from the sparse grid.
        Voxels not present in the sparse grid are filled with zeros.

        Args:
            sparse_data (JaggedTensor or torch.Tensor): Sparse voxel features to write.
                Shape: (total_voxels, channels).
            min_coord (list of 3-tuples, optional): Minimum coordinates for each grid
                in the batch. If None, computed from the grid bounds.
            grid_size (3-tuple of ints, optional): Size of the output dense tensor.
                If None, computed to fit all active voxels.

        Returns:
            torch.Tensor: Dense tensor containing the sparse data.
                Shape: (batch_size, channels, depth, height, width).
        """
        if not (
            is_JaggedTensorOrTensor(sparse_data)
            and (min_coord is None or is_Vec3iBatch(min_coord))
            and (grid_size is None or is_Vec3i(grid_size))
        ):
            raise TypeError(
                f"sparse_data must be a JaggedTensorOrTensor, but got {type(sparse_data)}, "
                f"min_coord must be a Vec3iBatch|None, but got {type(min_coord)}, "
                f"grid_size must be a Vec3i|None, but got {type(grid_size)}"
            )

        if isinstance(sparse_data, torch.Tensor):
            sparse_data = JaggedTensor(sparse_data)

        return self._impl.write_to_dense(sparse_data, min_coord, grid_size)  # type: ignore

    # Index methods
    def index_int(self, bi: int | np.integer) -> "GridBatch":
        """
        Get a subset of grids from the batch using integer indexing.

        Args:
            bi (int | np.integer): Grid index.

        Returns:
            GridBatch: A new GridBatch containing the selected grid.
        """
        return GridBatch(impl=self._impl.index_int(int(bi)))

    def index_slice(self, s: slice) -> "GridBatch":
        """
        Get a subset of grids from the batch using slicing.

        Args:
            s (slice): Slicing object.

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_slice(s))

    def index_list(self, indices: list[bool] | list[int]) -> "GridBatch":
        """
        Get a subset of grids from the batch using list indexing.

        Args:
            indices (list[bool] | list[int]): List of indices.

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_list(indices))

    def index_tensor(self, indices: torch.Tensor) -> "GridBatch":
        """
        Get a subset of grids from the batch using tensor indexing.

        Args:
            indices (torch.Tensor): Tensor of indices.

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        return GridBatch(impl=self._impl.index_tensor(indices))

    # Special methods
    def __getitem__(self, index: GridBatchIndex) -> "GridBatch":
        """
        Get a subset of grids from the batch using indexing.

        Supports integer indexing, slicing, list indexing, and boolean/integer tensor indexing.

        Args:
            index: Index to select grids. Can be:
                - int: Select a single grid
                - slice: Select a range of grids
                - list[int] or list[bool]: Select specific grids
                - torch.Tensor: Boolean or integer tensor for advanced indexing

        Returns:
            GridBatch: A new GridBatch containing the selected grids.
        """
        if not is_GridBatchIndex(index):
            raise TypeError(f"index must be a GridBatchIndex, but got {type(index)}")

        if isinstance(index, (int, np.integer)):
            return self.index_int(int(index))
        elif isinstance(index, slice):
            return self.index_slice(index)
        elif isinstance(index, list):
            return self.index_list(index)
        elif isinstance(index, torch.Tensor):
            return self.index_tensor(index)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

    def __iter__(self) -> Iterator["GridBatch"]:
        """
        Iterate over individual grids in the batch.

        Yields:
            GridBatch: Single-grid batches for each grid in the batch.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """
        Get the number of grids in the batch.

        Returns:
            int: Number of grids in this batch.
        """
        return self._impl.grid_count

    def has_same_address_and_grid_count(self, other: "GridBatch|GridBatchCpp") -> bool:
        """
        Check if two GridBatch objects have the same address and grid count.

        Args:
            other (GridBatch|GridBatchCpp): Other GridBatch object to compare.

        Returns:
            bool: True if the two GridBatch objects have the same address and grid count, False otherwise.
        """
        if isinstance(other, GridBatch):
            return self._impl.address == other._impl.address and self._impl.grid_count == other._impl.grid_count
        elif isinstance(other, GridBatchCpp):
            return self._impl.address == other.address and self._impl.grid_count == other.grid_count
        else:
            raise TypeError(f"Unsupported type for has_same_address_and_grid_count(): {type(other)}")

    # Properties
    @property
    def address(self) -> int:
        return self._impl.address

    @property
    def bbox(self) -> torch.Tensor:
        return self._impl.bbox

    @property
    def cum_voxels(self) -> torch.Tensor:
        return self._impl.cum_voxels

    @property
    def device(self) -> torch.device:
        return self._impl.device

    @property
    def dual_bbox(self) -> torch.Tensor:
        return self._impl.dual_bbox

    @property
    def grid_count(self) -> int:
        return self._impl.grid_count

    @property
    def grid_to_world_matrices(self) -> torch.Tensor:
        return self._impl.grid_to_world_matrices

    @property
    def ijk(self) -> JaggedTensor:
        return self._impl.ijk

    @property
    def jidx(self) -> torch.Tensor:
        return self._impl.jidx

    @property
    def joffsets(self) -> torch.Tensor:
        return self._impl.joffsets

    @property
    def num_bytes(self) -> torch.Tensor:
        return self._impl.num_bytes

    @property
    def num_leaf_nodes(self) -> torch.Tensor:
        return self._impl.num_leaf_nodes

    @property
    def num_voxels(self) -> torch.Tensor:
        return self._impl.num_voxels

    @property
    def origins(self) -> torch.Tensor:
        return self._impl.origins

    @property
    def total_bbox(self) -> torch.Tensor:
        return self._impl.total_bbox

    @property
    def total_bytes(self) -> int:
        return self._impl.total_bytes

    @property
    def total_leaf_nodes(self) -> int:
        return self._impl.total_leaf_nodes

    @property
    def total_voxels(self) -> int:
        return self._impl.total_voxels

    @property
    def viz_edge_network(self) -> tuple[JaggedTensor, JaggedTensor]:
        return self._impl.viz_edge_network

    @property
    def voxel_sizes(self) -> torch.Tensor:
        return self._impl.voxel_sizes

    @property
    def world_to_grid_matrices(self) -> torch.Tensor:
        return self._impl.world_to_grid_matrices

    # Expose underlying implementation for compatibility
    @property
    def _gridbatch(self):
        # Access underlying GridBatchCpp - use sparingly during migration
        return self._impl


# Module-level functions that create GridBatch objects
def gridbatch_from_dense(
    num_grids: int,
    dense_dims: Vec3i,
    ijk_min: Vec3i = torch.zeros(3, dtype=torch.int32),
    voxel_sizes: Vec3dBatchOrScalar = 1.0,
    origins: Vec3dBatch = torch.zeros(3),
    mask: torch.Tensor | None = None,
    device: torch.device | str = torch.device("cpu"),
) -> GridBatch:
    """
    Create a GridBatch from dense grid dimensions.

    Creates a sparse grid batch by allocating all voxels within the specified
    dense dimensions. Optionally, a mask can be provided to specify which voxels
    should be active.

    Args:
        num_grids (int): Number of grids in the batch.
        dense_dims (3-tuple of ints): Dimensions of the dense grid (width, height, depth).
        ijk_min (3-tuple of ints): Minimum corner of the dense region in index space.
            Default is (0, 0, 0).
        voxel_sizes (float or 3-tuple or tensor): Size of each voxel. If scalar, same size
            for all dimensions. Can be per-grid or global. Default is 1.0.
        origins (3-tuple or tensor): World-space origin for each grid. Default is (0, 0, 0).
        mask (torch.Tensor, optional): Boolean mask indicating which voxels are active.
            Shape: (num_grids, dense_dims[0], dense_dims[1], dense_dims[2]).
        device (torch.device or str): Device to create the grid on. Default is CPU.

    Returns:
        GridBatch: A new GridBatch with the specified dense structure.
    """
    from ._Cpp import gridbatch_from_dense as _gridbatch_from_dense

    if isinstance(device, str):
        device = _parse_device_string(device)

    if is_Vec3i(dense_dims) and is_Vec3i(ijk_min) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins):
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError(f"Unsupported type for mask: {type(mask)}")
    else:
        raise TypeError(
            "Unsupported types for gridbatch_from_dense(): "
            f"{type(dense_dims)}, {type(ijk_min)}, {type(voxel_sizes)}, {type(origins)}"
        )

    impl = _gridbatch_from_dense(num_grids, dense_dims, ijk_min, voxel_sizes, origins, mask, device)  # type: ignore
    return GridBatch(impl=impl)


def gridbatch_from_ijk(
    ijk: JaggedTensorOrTensor, voxel_sizes: Vec3dBatchOrScalar = 1.0, origins: Vec3dBatch = torch.zeros(3)
) -> GridBatch:
    """
    Create a GridBatch from explicit voxel coordinates.

    Creates a sparse grid batch by specifying the exact integer coordinates of
    active voxels.

    Args:
        ijk (JaggedTensor or torch.Tensor): Integer voxel coordinates for each grid.
            Shape: (total_voxels, 3) with each row being (i, j, k) indices.
        voxel_sizes (float or 3-tuple or tensor): Size of each voxel. If scalar, same size
            for all dimensions. Can be per-grid or global. Default is 1.0.
        origins (3-tuple or tensor): World-space origin for each grid. Default is (0, 0, 0).

    Returns:
        GridBatch: A new GridBatch with active voxels at the specified coordinates.
    """
    from ._Cpp import gridbatch_from_ijk as _gridbatch_from_ijk

    if is_JaggedTensorOrTensor(ijk) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins):
        impl = _gridbatch_from_ijk(ijk, voxel_sizes, origins)  # type: ignore
    else:
        raise TypeError(
            f"Unsupported types for gridbatch_from_ijk(): {type(ijk)}, {type(voxel_sizes)}, {type(origins)}"
        )
    return GridBatch(impl=impl)


def gridbatch_from_mesh(
    vertices: JaggedTensorOrTensor,
    faces: JaggedTensorOrTensor,
    voxel_sizes: Vec3dBatchOrScalar = 1.0,
    origins: Vec3dBatch = torch.zeros(3),
) -> GridBatch:
    """
    Create a GridBatch from triangle meshes.

    Voxelizes triangle meshes by creating active voxels for all voxels that
    intersect with the mesh surface.

    Args:
        vertices (JaggedTensor or torch.Tensor): Vertex positions for each mesh.
            Shape: (total_vertices, 3).
        faces (JaggedTensor or torch.Tensor): Triangle face indices for each mesh.
            Shape: (total_faces, 3) with each row containing vertex indices.
        voxel_sizes (float or 3-tuple or tensor): Size of each voxel. If scalar, same size
            for all dimensions. Can be per-grid or global. Default is 1.0.
        origins (3-tuple or tensor): World-space origin for each grid. Default is (0, 0, 0).

    Returns:
        GridBatch: A new GridBatch with voxels intersecting the mesh surfaces.
    """
    from ._Cpp import gridbatch_from_mesh as _gridbatch_from_mesh

    if (
        is_JaggedTensorOrTensor(vertices)
        and is_JaggedTensorOrTensor(faces)
        and is_Vec3dBatchOrScalar(voxel_sizes)
        and is_Vec3dBatch(origins)
    ):
        impl = _gridbatch_from_mesh(vertices, faces, voxel_sizes, origins)  # type: ignore
    else:
        raise TypeError(
            "Unsupported types for gridbatch_from_mesh(): "
            f"{type(vertices)}, {type(faces)}, {type(voxel_sizes)}, {type(origins)}"
        )

    return GridBatch(impl=impl)


def gridbatch_from_nearest_voxels_to_points(
    points: JaggedTensorOrTensor,
    voxel_sizes: Vec3dBatchOrScalar = 1.0,
    origins: Vec3dBatch = torch.zeros(3),
) -> GridBatch:
    """
    Create a GridBatch from nearest voxels to points.

    Creates active voxels at the locations of the voxel centers that are nearest
    to each input point. This is useful for sparse point cloud representations
    where each point should activate exactly one voxel.

    Args:
        points (JaggedTensor or torch.Tensor): Point positions in world space.
            Shape: (total_points, 3).
        voxel_sizes (float or 3-tuple or tensor): Size of each voxel. If scalar, same size
            for all dimensions. Can be per-grid or global. Default is 1.0.
        origins (3-tuple or tensor): World-space origin for each grid. Default is (0, 0, 0).

    Returns:
        GridBatch: A new GridBatch with active voxels at nearest positions to points.
    """
    from ._Cpp import (
        gridbatch_from_nearest_voxels_to_points as _gridbatch_from_nearest_voxels_to_points,
    )

    if is_JaggedTensorOrTensor(points) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins):
        impl = _gridbatch_from_nearest_voxels_to_points(points, voxel_sizes, origins)  # type: ignore
    else:
        raise TypeError(
            "Unsupported types for gridbatch_from_nearest_voxels_to_points(): "
            f"{type(points)}, {type(voxel_sizes)}, {type(origins)}"
        )

    return GridBatch(impl=impl)


def gridbatch_from_points(
    points: JaggedTensorOrTensor,
    voxel_sizes: Vec3dBatchOrScalar = 1.0,
    origins: Vec3dBatch = torch.zeros(3),
) -> GridBatch:
    """
    Create a GridBatch from point clouds.

    Creates active voxels at all locations containing one or more points.
    Multiple points falling within the same voxel result in a single active voxel.

    Args:
        points (JaggedTensor or torch.Tensor): Point positions in world space.
            Shape: (total_points, 3).
        voxel_sizes (float or 3-tuple or tensor): Size of each voxel. If scalar, same size
            for all dimensions. Can be per-grid or global. Default is 1.0.
        origins (3-tuple or tensor): World-space origin for each grid. Default is (0, 0, 0).

    Returns:
        GridBatch: A new GridBatch with active voxels containing points.
    """
    from ._Cpp import gridbatch_from_points as _gridbatch_from_points

    if is_JaggedTensorOrTensor(points) and is_Vec3dBatchOrScalar(voxel_sizes) and is_Vec3dBatch(origins):
        impl = _gridbatch_from_points(points, voxel_sizes, origins)  # type: ignore
    else:
        raise TypeError(
            f"Unsupported types for gridbatch_from_points(): {type(points)}, {type(voxel_sizes)}, {type(origins)}"
        )

    return GridBatch(impl=impl)


# Load and save functions
@overload
def load(
    path: str,
    *,
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load(
    path: str,
    *,
    indices: list[int],
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load(
    path: str,
    *,
    index: int,
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load(
    path: str,
    *,
    names: list[str],
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


@overload
def load(
    path: str,
    *,
    name: str,
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]: ...


def load(
    path: str,
    *,
    indices: list[int] | None = None,
    index: int | None = None,
    names: list[str] | None = None,
    name: str | None = None,
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = False,
) -> tuple[GridBatch, JaggedTensor, list[str]]:
    """Load a grid batch from a .nvdb file.

    Args:
        path: The path to the .nvdb file to load
        indices: Optional list of indices to load from the file (mutually exclusive with other selectors)
        index: Optional single index to load from the file (mutually exclusive with other selectors)
        names: Optional list of names to load from the file (mutually exclusive with other selectors)
        name: Optional single name to load from the file (mutually exclusive with other selectors)
        device: Which device to load the grid batch on
        verbose: If set to true, print information about the loaded grids

    Returns:
        A tuple (gridbatch, data, names) where gridbatch is a GridBatch containing the loaded
        grids, data is a JaggedTensor containing the data of the grids, and names is a list of
        strings containing the name of each grid
    """
    from ._Cpp import load as _load

    if isinstance(device, str):
        device = _parse_device_string(device)

    # Check that only one selector is provided
    selectors = [indices is not None, index is not None, names is not None, name is not None]
    if sum(selectors) > 1:
        raise ValueError("Only one of indices, index, names, or name can be specified")

    # Call the appropriate overload
    if indices is not None:
        grid_impl, data, names_out = _load(path, indices, device, verbose)
    elif index is not None:
        grid_impl, data, names_out = _load(path, index, device, verbose)
    elif names is not None:
        grid_impl, data, names_out = _load(path, names, device, verbose)
    elif name is not None:
        grid_impl, data, names_out = _load(path, name, device, verbose)
    else:
        # Load all grids
        grid_impl, data, names_out = _load(path, device, verbose)

    # Wrap the GridBatch implementation with the Python wrapper
    return GridBatch(impl=grid_impl), data, names_out


def save(
    path: str,
    grid_batch: GridBatch,
    data: JaggedTensorOrTensor | None = None,
    names: list[str] | str | None = None,
    name: str | None = None,
    compressed: bool = False,
    verbose: bool = False,
) -> None:
    """
    Save a grid batch and optional voxel data to a .nvdb file.

    Saves sparse grids in the NanoVDB format, which can be loaded by other
    applications that support OpenVDB/NanoVDB.

    Args:
        path (str): The file path to save to. Should have .nvdb extension.
        grid_batch (GridBatch): The grid batch to save.
        data (JaggedTensor or torch.Tensor, optional): Voxel data to save with the grids.
            Shape: (total_voxels, channels). If None, only grid structure is saved.
        names (list[str] or str, optional): Names for each grid in the batch.
            If a single string, it's used as the name for all grids.
        name (str, optional): Alternative way to specify a single name for all grids.
            Takes precedence over names parameter.
        compressed (bool): Whether to compress the data using Blosc compression.
            Default is False.
        verbose (bool): Whether to print information about the saved grids.
            Default is False.

    Note:
        The parameters 'names' and 'name' are mutually exclusive ways to specify
        grid names. Use 'name' for a single name applied to all grids, or 'names'
        for individual names per grid.
    """
    from ._Cpp import save as _save

    if data is not None:
        if isinstance(data, torch.Tensor):
            data = JaggedTensor(data)

    # Handle the overloaded signature - if name is provided, use it
    if name is not None:
        _save(path, grid_batch._impl, data, name, compressed, verbose)
    elif names is not None:
        if isinstance(names, str):
            # Handle case where names is actually a single name
            _save(path, grid_batch._impl, data, names, compressed, verbose)
        else:
            # Handle case where names is a list
            _save(path, grid_batch._impl, data, names, compressed, verbose)
    else:
        # Default case with empty names list
        _save(path, grid_batch._impl, data, [], compressed, verbose)
