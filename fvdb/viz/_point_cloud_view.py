# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .._Cpp import GaussianSplat3dView as GaussianSplat3dViewCpp


class PointCloudView:
    __PRIVATE__ = object()

    def __init__(self, view: GaussianSplat3dViewCpp, _private: Any = None):
        """
        Create a new :class:`PointCloudView` from a C++ implementation.

        This constructor is private and should not be called directly.
        Use :meth:`fvdb.viz.Viewer.register_point_cloud_view()` instead.

        Args:
            view (GaussianSplat3dViewCpp): The C++ implementation of the Point Cloud view.
                Note, we're actually hacking this to use the GaussianSplat3dView C++ class
                until we wrap up point clouds in the viewer.
            _private (Any): A private object to prevent direct construction. Must be `PointCloudView.__PRIVATE__`.
        """
        self._view = view
        if _private is not self.__PRIVATE__:
            raise ValueError("PointCloudView constructor is private. Use Viewer.register_point_cloud_view() instead.")

    @property
    def point_size(self) -> float:
        """
        Get the size (in pixels) of points when rendering.

        Returns:
            point_size(float): The current point size.
        """
        return self._view.eps_2d * 2.0  # point size is diameter

    @point_size.setter
    def point_size(self, size: float):
        """
        Set the size (in pixels) of points when rendering.

        Args:
            size (float): The point size to set.
        """
        if size <= 0.0:
            raise ValueError(f"Point size must be a positive float, got {size}")
        self._view.eps_2d = size / 2.0  # point size is diameter
