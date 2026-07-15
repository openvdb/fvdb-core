# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from .._fvdb_cpp import GaussianSplat3dView as GaussianSplat3dViewCpp
from ._gaussian_splat_view_data import ShOrderingMode
from ._viewer_server import _get_viewer_server_cpp


class GaussianSplat3dView:
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
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        sh_degree_to_use: int = -1,
        sh_ordering_mode: ShOrderingMode = ShOrderingMode.RGB_RGB_RGB,
        _private: Any = None,
    ):
        """
        Create a new :class:`GaussianSplat3dView` or update an existing one within a scene with the given name.

        .. warning::

            This constructor is private and should never be called directly. Use :meth:`fvdb.viz.Scene.add_gaussian_splat_3d()` instead.

        Args:
            scene_name (str): The name of the scene the view belongs to.
            name (str): The name of the GaussianSplat3dView.
            means (torch.Tensor): Gaussian means tensor of shape (N, 3).
            quats (torch.Tensor): Gaussian quaternions tensor of shape ``(N, 4)`` in
                ``(w, x, y, z)`` component order.
            log_scales (torch.Tensor): Gaussian log-scales tensor of shape (N, 3).
            logit_opacities (torch.Tensor): Gaussian logit-opacities tensor of shape (N,).
            sh0 (torch.Tensor): Zeroth-order SH coefficients with shape ``(N, 1, D)`` for
                ``"rgb_rgb_rgb"`` ordering or ``(N, D, 1)`` for ``"rrr_ggg_bbb"`` ordering.
            shN (torch.Tensor): Higher-order SH coefficients with shape ``(N, K - 1, D)`` for
                ``"rgb_rgb_rgb"`` ordering or ``(N, D, K - 1)`` for ``"rrr_ggg_bbb"`` ordering.
            tile_size (int): The tile size to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixels to use when rendering splats. Default is 0.0.
            eps_2d (float): The epsilon value to use when rendering splats. Default is 0.3.
            antialias (bool): Whether to use antialiasing when rendering splats. Default is False.
            sh_degree_to_use (int): The degree of spherical harmonics to use when rendering colors.
                If -1, the maximum degree supported by the Gaussian splat 3D scene is used. Default is -1.
            sh_ordering_mode (ShOrderingMode): The spherical harmonics tensor layout to use when rendering colors. Must be
                ``"rgb_rgb_rgb"`` or ``"rrr_ggg_bbb"``. Default is ``"rgb_rgb_rgb"``.
            _private (Any): A private object used by :class:`Scene` to construct the view.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("GaussianSplat3dView constructor is private. Use Scene.add_gaussian_splat_3d() instead.")
        self._scene_name = scene_name
        self._name = name
        server = _get_viewer_server_cpp()
        view = server.add_gaussian_splat_3d_view(
            scene_name=scene_name,
            name=name,
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
        )

        view.tile_size = tile_size
        view.min_radius_2d = min_radius_2d
        view.eps_2d = eps_2d
        view.antialias = antialias
        view.sh_degree_to_use = sh_degree_to_use
        self.sh_ordering_mode = sh_ordering_mode

    @property
    def tile_size(self) -> int:
        """
        Set the 2D tile size to use when rendering splats. Larger tiles can improve performance, but may
        exhaust shared memory usage on the GPU. In general, tile sizes of 8, 16, or 32 are recommended.

        Returns:
            int: The current tile size.
        """
        view = self._get_view()
        return view.tile_size

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
        view = self._get_view()
        view.tile_size = tile_size

    @property
    def min_radius_2d(self) -> float:
        """
        Get the minimum radius in pixels below which splats will not be rendered.

        Returns:
            float: The minimum radius in pixels.
        """
        view = self._get_view()
        return view.min_radius_2d

    @min_radius_2d.setter
    def min_radius_2d(self, radius: float):
        """
        Set the minimum radius in pixels below which splats will not be rendered.

        Args:
            radius (float): The minimum radius in pixels.
        """
        if radius < 0.0:
            raise ValueError(f"Minimum radius must be non-negative, got {radius}")
        view = self._get_view()
        view.min_radius_2d = radius

    @property
    def eps_2d(self) -> float:
        """
        Get the 2D epsilon value used for rendering splats.

        Returns:
            float: The 2D epsilon value.
        """
        view = self._get_view()
        return view.eps_2d

    @eps_2d.setter
    def eps_2d(self, eps: float):
        """
        Set the 2D epsilon value used for rendering splats.

        Args:
            eps (float): The 2D epsilon value.
        """
        if eps < 0.0:
            raise ValueError(f"Epsilon must be non-negative, got {eps}")
        view = self._get_view()
        view.eps_2d = eps

    @property
    def sh_degree_to_use(self) -> int:
        """
        Get the degree of spherical harmonics to use when rendering colors.

        Returns:
            int: The degree of spherical harmonics to use.
        """
        view = self._get_view()
        return view.sh_degree_to_use

    @sh_degree_to_use.setter
    def sh_degree_to_use(self, degree: int):
        """
        Sets the degree of spherical harmonics to use when rendering colors. If -1, the maximum
        degree supported by the Gaussian splat 3D scene is used.

        Args:
            degree (int): The degree of spherical harmonics to use.
        """
        view = self._get_view()
        view.sh_degree_to_use = degree

    @property
    def sh_ordering_mode(self) -> ShOrderingMode:
        """
        Get the spherical harmonics ordering mode used for rendering colors.

        Returns:
            ShOrderingMode: The spherical harmonics tensor layout.
        """
        return self._sh_ordering_mode

    @sh_ordering_mode.setter
    def sh_ordering_mode(self, mode: ShOrderingMode):
        """
        Set the spherical harmonics ordering mode used for rendering colors.

        Args:
            mode (ShOrderingMode): The spherical harmonics tensor layout.
        """
        view = self._get_view()
        try:
            mode = ShOrderingMode(mode)
        except ValueError:
            raise ValueError(f"Invalid spherical harmonics ordering: {mode!r}")
        self._sh_ordering_mode = mode
        view.rgb_rgb_rgb_sh = mode == ShOrderingMode.RGB_RGB_RGB
