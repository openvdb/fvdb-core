# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any

from .._Cpp import GaussianSplat3d as GaussianSplat3dCpp
from .._Cpp import GaussianSplat3dView as GaussianSplat3dViewCpp
from ._viewer_server import _get_viewer_server_cpp


class ShOrderingMode(str, Enum):
    RGB_RGB_RGB = "rgb_rgb_rgb"
    RRR_GGG_BBB = "rrr_ggg_bbb"


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
        gaussian_splat_3d: GaussianSplat3dCpp,
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
            gaussian_splat_3d (GaussianSplat3d): The Gaussian splat 3D scene to add.
            tile_size (int): The tile size to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixels to use when rendering splats. Default is 0.0.
            eps_2d (float): The epsilon value to use when rendering splats. Default is 0.3.
            antialias (bool): Whether to use antialiasing when rendering splats. Default is False.
            sh_degree_to_use (int): The degree of spherical harmonics to use when rendering colors.
                If -1, the maximum degree supported by the Gaussian splat 3D scene is used. Default is -1.
            sh_ordering_mode (ShOrderingMode): The spherical harmonics ordering mode to use when rendering colors.
                Default is :attr:`ShOrderingMode.RGB_RGB_RGB`.
            _private (Any): A private object to prevent direct construction. Must be :attr:`GaussianSplat3dView.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("GaussianSplat3dView constructor is private. Use Scene.add_gaussian_splat_3d() instead.")
        self._scene_name = scene_name
        self._name = name
        server = _get_viewer_server_cpp()
        view = server.add_gaussian_splat_3d_view(name=name, gaussian_splat_3d=gaussian_splat_3d)

        if sh_ordering_mode not in (ShOrderingMode.RGB_RGB_RGB, ShOrderingMode.RRR_GGG_BBB):
            raise ValueError(f"Invalid ShOrderingMode: {sh_ordering_mode}")

        view.tile_size = tile_size
        view.min_radius_2d = min_radius_2d
        view.eps_2d = eps_2d
        view.antialias = antialias
        view.sh_degree_to_use = sh_degree_to_use
        if sh_ordering_mode == ShOrderingMode.RRR_GGG_BBB:
            view.rgb_rgb_rgb_sh = False
        elif sh_ordering_mode == ShOrderingMode.RGB_RGB_RGB:
            view.rgb_rgb_rgb_sh = True
        else:
            raise ValueError(f"Invalid ShOrderingMode: {sh_ordering_mode}")

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
            ShOrderingMode: The spherical harmonics ordering mode.
        """
        view = self._get_view()
        if view.rgb_rgb_rgb_sh:
            return ShOrderingMode.RRR_GGG_BBB
        else:
            return ShOrderingMode.RGB_RGB_RGB

    @sh_ordering_mode.setter
    def sh_ordering_mode(self, mode: ShOrderingMode):
        """
        Set the spherical harmonics ordering mode used for rendering colors.

        Args:
            mode (ShOrderingMode): The spherical harmonics ordering mode.
        """
        view = self._get_view()
        if mode == ShOrderingMode.RRR_GGG_BBB:
            view.rgb_rgb_rgb_sh = False
        elif mode == ShOrderingMode.RGB_RGB_RGB:
            view.rgb_rgb_rgb_sh = True
        else:
            raise ValueError(f"Invalid ShOrderingMode: {mode}")
