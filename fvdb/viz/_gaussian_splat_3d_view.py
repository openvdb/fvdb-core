# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .._Cpp import GaussianSplat3dView as GaussianSplat3dViewCpp


class GaussianSplat3dView:
    __PRIVATE__ = object()

    def __init__(self, view: GaussianSplat3dViewCpp, _private: Any = None):
        """
        Create a new `GaussianSplat3dView` from a C++ implementation. This constructor is private and should not be called directly.
        Use `Viewer.register_gaussian_splat3d_view()` instead.

        Args:
            view (GaussianSplat3dViewCpp): The C++ implementation of the Gaussian splat 3D view.
            _private (Any): A private object to prevent direct construction. Must be `GaussianSplat3dView.__PRIVATE__`.
        """
        self._view = view
        if _private is not self.__PRIVATE__:
            raise ValueError(
                "GaussianSplat3dView constructor is private. Use Viewer.register_gaussian_splat3d_view() instead."
            )

    @property
    def tile_size(self) -> int:
        """
        Set the 2D tile size to use when rendering splats. Larger tiles can improve performance, but may
        exhaust shared memory usage on the GPU. In general, tile sizes of 8, 16, or 32 are recommended.

        Returns:
            int: The current tile size.
        """
        return self._view.tile_size

    @tile_size.setter
    def tile_size(self, tile_size: int):
        """
        Set the 2D tile size to use when rendering splats. Larger tiles can improve performance, but may
        exhaust shared memory usage on the GPU. In general, tile sizes of 8, 16, or 32 are recommended.

        Args:
            tile_size (int): The tile size to set.
        """
        if tile_size < 1:
            raise ValueError(f"Tile size must be a positive integer, got {tile_size}")
        self._view.tile_size = tile_size

    @property
    def min_radius_2d(self) -> float:
        """
        Get the minimum radius in pixels below which splats will not be rendered.

        Returns:
            float: The minimum radius in pixels.
        """
        return self._view.min_radius_2d

    @min_radius_2d.setter
    def min_radius_2d(self, radius: float):
        """
        Set the minimum radius in pixels below which splats will not be rendered.

        Args:
            radius (float): The minimum radius in pixels.
        """
        if radius < 0.0:
            raise ValueError(f"Minimum radius must be non-negative, got {radius}")
        self._view.min_radius_2d = radius

    @property
    def eps_2d(self) -> float:
        """
        Get the 2D epsilon value used for rendering splats.

        Returns:
            float: The 2D epsilon value.
        """
        return self._view.eps_2d

    @eps_2d.setter
    def eps_2d(self, eps: float):
        """
        Set the 2D epsilon value used for rendering splats.

        Args:
            eps (float): The 2D epsilon value.
        """
        if eps < 0.0:
            raise ValueError(f"Epsilon must be non-negative, got {eps}")
        self._view.eps_2d = eps

    @property
    def sh_degree_to_use(self) -> int:
        """
        Get the degree of spherical harmonics to use when rendering colors.

        Returns:
            int: The degree of spherical harmonics to use.
        """
        return self._view.sh_degree_to_use

    @sh_degree_to_use.setter
    def sh_degree_to_use(self, degree: int):
        """
        Sets the degree of spherical harmonics to use when rendering colors. If -1, the maximum
        degree supported by the Gaussian splat 3D scene is used.

        Args:
            degree (int): The degree of spherical harmonics to use.
        """
        self._view.sh_degree_to_use = degree

    @property
    def sh_stride_rgb_rgb_rgb(self) -> bool:
        """
        Get whether the spherical harmonics coefficients are stored in interleaved RGB format.

        Returns:
            bool: True if the spherical harmonics coefficients are stored in interleaved RGB format, False otherwise.
        """
        return self._view.sh_stride_rgb_rgb_rgb

    @sh_stride_rgb_rgb_rgb.setter
    def sh_stride_rgb_rgb_rgb(self, value: bool):
        """
        Set whether the spherical harmonics coefficients are stored in interleaved RGB format.

        Args:
            value (bool): True if the spherical harmonics coefficients are stored in interleaved RGB format, False otherwise.
        """
        self._view.sh_stride_rgb_rgb_rgb = value

    @property
    def sh_stride_rrr_ggg_bbb(self) -> bool:
        """
        Get whether the spherical harmonics coefficients are stored in planar RRR GGG BBB format.

        Returns:
            bool: True if the spherical harmonics coefficients are stored in planar RRR GGG BBB format, False otherwise.
        """
        return self._view.sh_stride_rrr_ggg_bbb

    @sh_stride_rrr_ggg_bbb.setter
    def sh_stride_rrr_ggg_bbb(self, value: bool):
        """
        Set whether the spherical harmonics coefficients are stored in planar RRR GGG BBB format.

        Args:
            value (bool): True if the spherical harmonics coefficients are stored in planar RRR GGG BBB format, False otherwise.
        """
        self._view.sh_stride_rrr_ggg_bbb = value

    @property
    def near(self) -> float:
        """
        Gets the near clipping plane distance for rendering. Splats closer to the camera than this distance
        will not be rendered.

        Returns:
            float: The near clipping plane distance.
        """
        return self._view.near

    @near.setter
    def near(self, near: float):
        """
        Sets the near clipping plane distance for rendering. Splats closer to the camera than this distance
        will not be rendered.

        Args:
            near (float): The near clipping plane distance.
        """
        self._view.near = near

    @property
    def far(self) -> float:
        """
        Get the far clipping plane distance for rendering. Splats farther from the camera than this distance
        will not be rendered.

        Returns:
            float: The far clipping plane distance.
        """
        return self._view.far

    @far.setter
    def far(self, far: float):
        """
        Sets the far clipping plane distance for rendering. Splats farther from the camera than this distance
        will not be rendered.

        Args:
            far (float): The far clipping plane distance.
        """
        self._view.far = far
