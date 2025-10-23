# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from fvdb import GaussianSplat3d

from .._Cpp import GaussianSplat3dView as GaussianSplat3dViewCpp
from ._viewer_server import _get_viewer_server_cpp


class PointCloudView:
    __PRIVATE__ = object()

    def _get_view(self) -> GaussianSplat3dViewCpp:
        """
        Get the underlying C++ GaussianSplat3dView instance from the viewer server or raise a :class:`RuntimeError` if it is not registered.

        Returns:
            view (GaussianSplat3dViewCpp): The C++ GaussianSplat3dView instance
        """
        server = _get_viewer_server_cpp()

        if not server.has_gaussian_splat_3d_view(self._name):
            raise RuntimeError(f"GaussianSplat3dView '{self._name}' is not registered with the viewer server.")
        return server.get_gaussian_splat_3d_view(self._name)

    def __init__(
        self,
        scene_name: str,
        name: str,
        positions: torch.Tensor,
        colors: torch.Tensor,
        point_size: float,
        _private: Any = None,
    ):
        """
        Create a new :class:`PointCloudView` or update an existing one within a scene with the given name.

        .. warning::

            This constructor is private and should never be called directly. Use :meth:`fvdb.viz.Scene.add_point_cloud()` instead.

        Args:
            scene_name (str): The name of the scene the view belongs to.
            name (str): The name of the :class:`PointCloudView`.
            positions (torch.Tensor): A tensor of shape ``(N, 3)`` representing the 3D positions of the points.
            colors (torch.Tensor): A tensor of shape ``(N, 3)`` representing the RGB colors of the points.
            point_size (float): The size (in pixels) of the points when rendering.
            _private (Any): A private object to prevent direct construction. Must be :attr:`PointCloudView.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("PointCloudView constructor is private. Use Viewer.register_point_cloud_view() instead.")

        self._name = name
        self._scene_name = scene_name

        server = _get_viewer_server_cpp()

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        means = positions
        quats = torch.zeros((positions.shape[0], 4), dtype=torch.float32)
        quats[:, 0] = 1.0  # identity rotation
        logit_opacities = torch.full((positions.shape[0],), 10.0, dtype=torch.float32)
        log_scales = torch.full((positions.shape[0], 3), -20.0, dtype=torch.float32)  # since scales are exp(log_scale)
        sh0 = _rgb_to_sh(colors)
        shN = torch.zeros((positions.shape[0], 0, 3), dtype=torch.float32)

        gs_impl = GaussianSplat3d.from_tensors(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
        )._impl
        view: GaussianSplat3dViewCpp = server.add_gaussian_splat_3d_view(
            scene_name=scene_name, name=name, gaussian_splat_3d=gs_impl
        )
        view.tile_size = 16
        view.min_radius_2d = 0.0
        view.eps_2d = point_size / 2.0  # point size is diameter
        view.antialias = False
        view.sh_degree_to_use = 0

    @property
    def point_size(self) -> float:
        """
        Get the size (in pixels) of points when rendering.

        Returns:
            size (float): The current point size.
        """
        view = self._get_view()
        return view.eps_2d * 2.0  # point size is diameter

    @point_size.setter
    def point_size(self, size: float):
        """
        Set the size (in pixels) of points when rendering.

        Args:
            size (float): The point size to set.
        """
        if size <= 0.0:
            raise ValueError(f"Point size must be a positive float, got {size}")
        view = self._get_view()
        view.eps_2d = size / 2.0  # point size is diameter
