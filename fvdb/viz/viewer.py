# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any

import numpy as np
import torch

from .._Cpp import GaussianSplat3d as GaussianSplat3dCpp
from .._Cpp import GaussianSplat3dView as GaussianSplat3dViewCpp
from .._Cpp import Viewer as ViewerCpp
from ..gaussian_splatting import GaussianSplat3d
from ..types import NumericMaxRank1, NumericMaxRank2, to_Mat44f, to_Vec3f


class GaussianSplat3dView:
    __PRIVATE__ = object()

    def __init__(self, view: GaussianSplat3dViewCpp, _private: Any = None):
        self._view = view
        if _private is not self.__PRIVATE__:
            raise ValueError(
                "GaussianSplat3dView constructor is private. Use Viewer.register_gaussian_splat3d_view() instead."
            )

    @property
    def tile_size(self) -> int:
        return self._view.tile_size

    @tile_size.setter
    def tile_size(self, size: int):
        self._view.tile_size = size

    @property
    def min_radius_2d(self) -> float:
        return self._view.min_radius_2d

    @min_radius_2d.setter
    def min_radius_2d(self, radius: float):
        self._view.min_radius_2d = radius

    @property
    def sh_degree_to_use(self) -> int:
        return self._view.sh_degree_to_use

    @sh_degree_to_use.setter
    def sh_degree_to_use(self, degree: int):
        self._view.sh_degree_to_use = degree

    @property
    def near(self) -> float:
        return self._view.near

    @near.setter
    def near(self, near: float):
        self._view.near = near

    @property
    def far(self) -> float:
        return self._view.far

    @far.setter
    def far(self, far: float):
        self._view.far = far


class Viewer:
    def __init__(self, ip_address: str = "127.0.0.1", port: int = 8888, verbose: bool = False):
        """
        Create a new `Viewer` running a server at the specified IP address and port.

        If there is already a viewer server running at the specified address and port,
        this will connect to the existing server instead of starting a new one.

        Args:
            ip_address (str): The IP address to bind the viewer server to. Default is "127.0.0.1"
            port (int): The port to bind the viewer server to. Default is 8888.
        """
        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError(f"Port must be an integer between 0 and 65535, got {port}")

        # TODO: Check that either no application is running on ip_address:port
        # or that it is a viewer server we can connect to.
        self._impl = ViewerCpp(ip_address=ip_address, port=port, verbose=verbose)

    def add_gaussian_splat3d(
        self,
        name: str,
        gaussian_splat_3d: GaussianSplat3d,
        tile_size: int = 16,
        min_radius_2d: float = 0.0,
        eps_2d: float = 0.3,
        antialias: bool = False,
        sh_degree_to_use: int = -1,
    ) -> GaussianSplat3dView:
        """
        Add a `GaussianSplat3d` to the viewer and return a view for it.

        Args:
            name (str): The name of the Gaussian splat 3D scene. This must be unique among all
                scenes added to the viewer.
            gaussian_splat_3d (GaussianSplat3d): The Gaussian splat 3D scene to add.
            tile_size (int): The tile size to use for rendering. Default is 16.
            min_radius_2d (float): The minimum radius in pixels to use when rendering splats. Default is 0.0.
            eps_2d (float): The epsilon value to use when rendering splats. Default is 0.3.
            antialias (bool): Whether to use antialiasing when rendering splats. Default is False.
            sh_degree_to_use (int): The degree of spherical harmonics to use when rendering colors.
                If -1, the maximum degree supported by the Gaussian splat 3D scene is used. Default is -1.

        Returns:
            GaussianSplat3dView: A view for the added Gaussian splat 3D scene.
        """
        gs_impl: GaussianSplat3dCpp = gaussian_splat_3d._impl
        view: GaussianSplat3dViewCpp = self._impl.add_gaussian_splat_3d(name=name, gaussian_splat_3d=gs_impl)
        view.tile_size = tile_size
        view.min_radius_2d = min_radius_2d
        view.eps_2d = eps_2d
        view.antialias = antialias
        if sh_degree_to_use >= 0:
            sh_degree_to_use = max(0, min(gaussian_splat_3d.sh_degree, sh_degree_to_use))
        else:
            sh_degree_to_use = gaussian_splat_3d.sh_degree
        view.sh_degree_to_use = sh_degree_to_use
        return GaussianSplat3dView(view, GaussianSplat3dView.__PRIVATE__)

    @property
    def camera_origin(self) -> torch.Tensor:
        """
        Return center of the camera in world coordinates.

        Returns:
            torch.Tensor: A tensor of shape (3,) representing the camera position in world coordinates.
        """
        ox, oy, oz = self._impl.camera_origin()
        return torch.tensor([ox, oy, oz], dtype=torch.float32)

    @camera_origin.setter
    def camera_origin(self, origin: NumericMaxRank1):
        """
        Set the center of the camera in world coordinates.

        Args:
            origin (NumericMaxRank1): A tensor-like object of shape (3,) representing the camera position in world coordinates.
        """
        origin_vec3f = to_Vec3f(origin).cpu().numpy().tolist()
        self._impl.set_camera_origin(*origin_vec3f)

    @property
    def camera_up_direction(self) -> torch.Tensor:
        """
        Return the up vector of the camera. _i.e._ the direction that is considered 'up' in the camera's view.

        Returns:
            torch.Tensor: A tensor of shape (3,) representing the up vector of the camera.
        """
        ux, uy, uz = self._impl.camera_up_direction()
        return torch.tensor([ux, uy, uz], dtype=torch.float32)

    @camera_up_direction.setter
    def camera_up_direction(self, up: NumericMaxRank1):
        """
        Set the up vector of the camera. _i.e._ the direction that is considered 'up' in the camera's view.

        Args:
            up (NumericMaxRank1): A tensor-like object of shape (3,) representing the up vector of the camera.
        """
        up_vec3f = to_Vec3f(up).cpu().numpy().tolist()
        self._impl.set_camera_up_direction(*up_vec3f)

    @property
    def camera_view_direction(self) -> torch.Tensor:
        """
        Return the view direction of the camera.

        Returns:
            torch.Tensor: A tensor of shape (3,) representing the view direction of the camera.
        """
        dx, dy, dz = self._impl.camera_view_direction()
        return torch.tensor([dx, dy, dz], dtype=torch.float32)

    @camera_view_direction.setter
    def camera_view_direction(self, direction: NumericMaxRank1):
        """
        Set the view direction of the camera.

        Args:
            direction (NumericMaxRank1): A tensor-like object of shape (3,) representing the view direction of the camera.
        """
        dir_vec3f = to_Vec3f(direction).cpu().numpy().tolist()
        self._impl.set_camera_view_direction(*dir_vec3f)

    def set_camera_lookat(
        self,
        camera_origin: NumericMaxRank1,
        lookat_point: NumericMaxRank1,
        up_direction: NumericMaxRank1 = [0.0, 1.0, 0.0],
    ):
        """
        Set the camera pose from a camera origin, a lookat point, and an up direction.

        Args:
            camera_origin (NumericMaxRank1): A tensor-like object of shape (3,) representing the camera position in world coordinates.
            lookat_point (NumericMaxRank1): A tensor-like object of shape (3,) representing the point the camera is looking at.
            up_direction (NumericMaxRank1): A tensor-like object of shape (3,) representing the up direction of the camera.
        """
        camera_origin_vec3f = to_Vec3f(camera_origin).cpu().numpy()
        lookat_point_vec3f = to_Vec3f(lookat_point).cpu().numpy()
        up_direction_vec3f = to_Vec3f(up_direction).cpu().numpy()
        view_direction = lookat_point_vec3f - camera_origin_vec3f
        if np.linalg.norm(view_direction) < 1e-6:
            raise ValueError("Camera origin and lookat point cannot be the same.")
        if np.linalg.norm(up_direction_vec3f) < 1e-6:
            raise ValueError("Up direction cannot be a zero vector.")

        view_direction /= np.linalg.norm(view_direction)
        up_direction_vec3f /= np.linalg.norm(up_direction_vec3f)
        right_direction = np.cross(view_direction, up_direction_vec3f)
        if np.linalg.norm(right_direction) < 1e-6:
            raise ValueError("Up direction cannot be parallel to the view direction.")
        right_direction /= np.linalg.norm(right_direction)
        up_direction_vec3f = np.cross(right_direction, view_direction)
        up_direction_vec3f /= np.linalg.norm(up_direction_vec3f)

        self._impl.set_camera_origin(*camera_origin_vec3f)
        self._impl.set_camera_view_direction(*view_direction)
        self._impl.set_camera_up_direction(*up_direction_vec3f)
