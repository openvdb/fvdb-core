# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging

import numpy as np
import torch

from ..gaussian_splatting import GaussianSplat3d
from ..types import (
    NumericMaxRank1,
    NumericMaxRank2,
    NumericMaxRank3,
    to_Mat33fBatch,
    to_Mat44fBatch,
    to_Vec2fBatch,
    to_Vec3f,
    to_Vec3fBatch,
)
from ._camera_view import CamerasView
from ._gaussian_splat_3d_view import GaussianSplat3dView
from ._point_cloud_view import PointCloudView
from ._viewer_server import _get_viewer_server_cpp


def get_scene(name: str = "fVDB Scene") -> "Scene":
    """
    Get a :class:`fvdb.viz.Scene` by name from the viewer server. If the scene does not exist,
    this function creates a new scene with the given name.

    Args:
        name (str): The name of the scene to get.

    Returns:
        scene (fvdb.viz.Scene): The scene with the given name.
    """
    return Scene(name)


class Scene:
    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        # TODO: Register scene with name in the viewer and use the returned scene ID.
        server = _get_viewer_server_cpp()
        server.add_scene(name)

    @torch.no_grad()
    def add_point_cloud(
        self,
        name: str,
        points: NumericMaxRank2,
        colors: NumericMaxRank2,
        point_size: float,
    ):
        """
        Add a point cloud with colors and world-space radii to the viewer and return a view for it.

        .. note::

            Colors must be in the range ``[0, 1]``.
            You can pass in a single color as a tuple of 3 floats to color all points the same.

        .. note::

            You can pass in a single radius as a float to use the same radius for all points.

        Args:
            name (str): The name of the point cloud added to the viewer. If a point cloud with the same name
                already exists in the viewer, it will be replaced.
            points (NumericMaxRank2): The 3D points of the point cloud as a tensor-like object of shape ``(N, 3)``
                where ``N`` is the number of points.
            colors (NumericMaxRank2): The colors of the points as a tensor-like object of shape ``(N, 3)`` where ``N`` is the number of points.
                Alternatively, you can pass in a single color as a tensor-like object of shape ``(3,)`` to color all points the same.
            point_size (float): The screen-space size (in pixels) of the points when rendering.

        Returns:
            point_cloud_view (GaussianSplat3dView): A view for the point cloud added to the scene.
        """

        points = to_Vec3fBatch(points).cpu()
        colors = to_Vec3fBatch(colors).cpu()
        if colors.shape[0] == 1:
            colors = colors.repeat(points.shape[0], 1)

        if colors.shape[0] != points.shape[0]:
            raise ValueError(
                f"Colors must be a tuple of 3 floats tensor with the same number of elements as points. Got {colors.shape[0]} colors and {points.shape[0]} points."
            )

        if colors.min() < 0.0 or colors.max() > 1.0:
            raise ValueError("Colors must be in the range [0, 1].")

        return PointCloudView(
            scene_name=self._name,
            name=name,
            positions=points,
            colors=colors,
            point_size=point_size,
            _private=PointCloudView.__PRIVATE__,
        )

    @torch.no_grad()
    def add_gaussian_splat_3d(
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
        Add a :class:`fvdb.GaussianSplat3d` to the viewer and return a view for it.

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
            gaussian_splat_3d_view (GaussianSplat3dView): A view for the Gaussian splats added to the scene.
        """
        return GaussianSplat3dView(
            scene_name=self._name,
            name=name,
            gaussian_splat_3d=gaussian_splat_3d._impl,
            tile_size=tile_size,
            min_radius_2d=min_radius_2d,
            eps_2d=eps_2d,
            antialias=antialias,
            sh_degree_to_use=sh_degree_to_use,
            _private=GaussianSplat3dView.__PRIVATE__,
        )

    @torch.no_grad()
    def add_cameras(
        self,
        name: str,
        camera_to_world_matrices: NumericMaxRank3,
        projection_matrices: NumericMaxRank3,
        image_sizes: NumericMaxRank2 | None = None,
        axis_length: float = 0.3,
        axis_thickness: float = 2.0,
        frustum_line_width: float = 2.0,
        frustum_scale: float = 1.0,
        frustum_color: NumericMaxRank1 = (0.5, 0.8, 0.3),
        frustum_near_plane: float = 0,
        frustum_far_plane: float = 0.5,
        enabled: bool = True,
    ) -> CamerasView:
        """
        Add :class:`~fvdb.viz.CamerasView` to this :class:`Scene` and return the added camera view.

        Args:
            name (str): The name of the camera view.
            camera_to_world_matrices (NumericMaxRank3): The 4x4 camera to world transformation matrices (one per camera) encoded
                as a tensor-like object of shape ``(N, 4, 4)`` where ``N`` is the number of cameras.
            projection_matrices (NumericMaxRank3 | None): The 3x3 projection matrices (one per camera) encoded
                as a tensor-like object of shape ``(N, 3, 3)`` where ``N`` is the number of cameras. If ``None``,
                it will use the projection matrix of the scene's main camera.
            image_sizes (NumericMaxRank2 | None): The image sizes as a tensor of shape ``(N, 2)`` where ``N`` is the number of cameras.
                such that ``height_i, width_i = image_sizes[i]`` is the resolution of the ``i``-th camera.
                If ``None``, the image sizes will be inferred from the projection matrices assuming square pixels and
                that the principal point is at the center of the image.
            axis_length (float): The length of the axis lines in the camera frustum view.
            axis_thickness (float): The thickness (in world coordinates) of the axis lines in the camera frustum view.
            frustum_line_width (float): The width (in pixels) of the frustum lines in the camera frustum view.
            frustum_scale (float): The scale factor for the frustum size in the camera frustum view.
            frustum_color (NumericMaxRank1): The color of the frustum lines as a sequence of three floats (R, G, B) in the range [0, 1].
            frustum_near_plane (float): The near clipping plane distance for the frustum in the camera frustum view.
            frustum_far_plane (float): The far clipping plane distance for the frustum in the camera frustum view.
            enabled (bool): If True, the camera view UI is enabled and the cameras will be rendered.
                If False, the camera view UI is disabled and the cameras will not be rendered.
        """
        if camera_to_world_matrices is None or projection_matrices is None:
            raise ValueError("Both camera_to_world_matrices and projection_matrices must be provided.")

        r, g, b = to_Vec3f(frustum_color).cpu().numpy().tolist()
        frustum_color = (r, g, b)
        if any(c < 0.0 or c > 1.0 for c in frustum_color):
            raise ValueError(f"Frustum color must be a sequence of three floats in [0, 1], got {frustum_color}")

        camera_to_world_matrices = to_Mat44fBatch(camera_to_world_matrices)
        projection_matrices = to_Mat33fBatch(projection_matrices)
        image_sizes = to_Vec2fBatch(image_sizes) if image_sizes is not None else torch.Tensor([])
        return CamerasView(
            scene_name=self._name,
            name=name,
            camera_to_world_matrices=camera_to_world_matrices,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
            axis_length=axis_length,
            axis_thickness=axis_thickness,
            frustum_line_width=frustum_line_width,
            frustum_scale=frustum_scale,
            frustum_color=frustum_color,
            frustum_near_plane=frustum_near_plane,
            frustum_far_plane=frustum_far_plane,
            enabled=enabled,
            _private=CamerasView.__PRIVATE__,
        )

    @property
    def camera_orbit_center(self) -> torch.Tensor:
        """
        Return center of the camera orbit in world coordinates.

        .. seealso:: :attr:`camera_orbit_direction`
        .. seealso:: :attr:`camera_orbit_radius`

        .. note::
            The camera itself is positioned at: ``camera_position = orbit_center + orbit_radius * orbit_direction``

        Returns:
            center (torch.Tensor): A tensor of shape ``(3,)`` representing the camera orbit center in world coordinates.
        """
        server = _get_viewer_server_cpp()
        ox, oy, oz = server.camera_orbit_center(self._name)
        return torch.tensor([ox, oy, oz], dtype=torch.float32)

    @camera_orbit_center.setter
    def camera_orbit_center(self, center: NumericMaxRank1):
        """
        Set the center of the camera orbit in world coordinates.

        .. seealso:: :attr:`camera_orbit_direction`
        .. seealso:: :attr:`camera_orbit_radius`

        .. note::
            The camera itself is positioned at: ``camera_position = orbit_center + orbit_radius * orbit_direction``

        Args:
            center (NumericMaxRank1): A tensor-like object of shape ``(3,)`` representing the camera orbit center in world coordinates.
        """
        server = _get_viewer_server_cpp()
        center_vec3f = to_Vec3f(center).cpu().numpy().tolist()
        server.set_camera_orbit_center(self._name, *center_vec3f)

    @property
    def camera_orbit_radius(self) -> float:
        """
        Return the radius of the camera orbit.

        .. seealso:: :attr:`camera_orbit_direction`
        .. seealso:: :attr:`camera_orbit_center`

        .. note::
            The camera itself is positioned at: ``camera_position = orbit_center + orbit_radius * orbit_direction``

        Returns:
            radius (float): The radius of the camera orbit.
        """
        server = _get_viewer_server_cpp()
        return server.camera_orbit_radius(self._name)

    @camera_orbit_radius.setter
    def camera_orbit_radius(self, radius: float):
        """
        Set the radius of the camera orbit.

        .. seealso:: :attr:`camera_orbit_direction`
        .. seealso:: :attr:`camera_orbit_center`

        .. note::
            The camera itself is positioned at: ``camera_position = orbit_center + orbit_radius * orbit_direction``

        Args:
            radius (float): The radius of the camera orbit.
        """
        if radius <= 0.0:
            raise ValueError(f"Radius must be positive, got {radius}")
        server = _get_viewer_server_cpp()
        server.set_camera_orbit_radius(self._name, radius)

    @property
    def camera_orbit_direction(self) -> torch.Tensor:
        """
        Return the direction pointing from the orbit center to the camera position.

        .. seealso:: :attr:`camera_orbit_radius`
        .. seealso:: :attr:`camera_orbit_center`

        .. note::
            The camera itself is positioned at: ``camera_position = orbit_center + orbit_radius * orbit_direction``

        Returns:
            direction (torch.Tensor): A tensor of shape ``(3,)`` representing the direction pointing from the orbit
                center to the camera position.
        """
        server = _get_viewer_server_cpp()
        dx, dy, dz = server.camera_view_direction(self._name)
        return torch.tensor([dx, dy, dz], dtype=torch.float32)

    @camera_orbit_direction.setter
    def camera_orbit_direction(self, direction: NumericMaxRank1):
        """
        Set the direction pointing from the orbit center to the camera position.

        .. seealso:: :attr:`camera_orbit_radius`
        .. seealso:: :attr:`camera_orbit_center`

        .. note::
            The camera itself is positioned at: ``camera_position = orbit_center + orbit_radius * orbit_direction``

        Args:
            direction (NumericMaxRank1): A tensor-like object of shape ``(3,)`` representing the direction pointing from the orbit
                center to the camera position.
        """
        server = _get_viewer_server_cpp()
        dir_vec3f = to_Vec3f(direction).cpu().numpy()
        if np.linalg.norm(dir_vec3f) < 1e-6:
            raise ValueError("Camera orbit direction cannot be a zero vector.")
        dir_vec3f /= np.linalg.norm(dir_vec3f)
        server.set_camera_view_direction(self._name, *dir_vec3f)

    @property
    def camera_up_direction(self) -> torch.Tensor:
        """
        Return the up vector of the camera. *i.e.* the direction that is considered 'up' in the camera's view.

        Returns:
            up (torch.Tensor): A tensor of shape ``(3,)`` representing the up vector of the camera.
        """
        server = _get_viewer_server_cpp()
        ux, uy, uz = server.camera_up_direction(self._name)
        return torch.tensor([ux, uy, uz], dtype=torch.float32)

    @camera_up_direction.setter
    def camera_up_direction(self, up: NumericMaxRank1):
        """
        Set the up vector of the camera. *i.e.* the direction that is considered 'up' in the camera's view.

        Args:
            up (NumericMaxRank1): A tensor-like object of shape ``(3,)`` representing the up vector of the camera.
        """
        server = _get_viewer_server_cpp()
        up_vec3f = to_Vec3f(up).cpu().numpy()
        if np.linalg.norm(up_vec3f) < 1e-6:
            raise ValueError("Camera up direction cannot be a zero vector.")
        up_vec3f /= np.linalg.norm(up_vec3f)
        server.set_camera_up_direction(self._name, *up_vec3f)

    @property
    def camera_near(self) -> float:
        """
        Get the near clipping plane distance for rendering. Objects closer to the camera than this distance
        will not be rendered.

        Returns:
            near (float): The near clipping plane distance.
        """
        server = _get_viewer_server_cpp()
        return server.camera_near(self._name)

    @camera_near.setter
    def camera_near(self, near: float):
        """
        Sets the near clipping plane distance for rendering. Objects closer to the camera than this distance
        will not be rendered.

        Args:
            near (float): The near clipping plane distance.
        """
        server = _get_viewer_server_cpp()
        if near <= 0.0:
            raise ValueError(f"Near clipping plane distance must be positive, got {near}")
        server.set_camera_near(self._name, near)

    @property
    def camera_far(self) -> float:
        """
        Get the far clipping plane distance for rendering. Objects farther from the camera than this distance
        will not be rendered.

        Returns:
            far (float): The far clipping plane distance.
        """
        server = _get_viewer_server_cpp()
        return server.camera_far(self._name)

    @camera_far.setter
    def camera_far(self, far: float):
        """
        Set the far clipping plane distance for rendering. Objects farther from the camera than this distance
        will not be rendered.

        Args:
            far (float): The far clipping plane distance.
        """
        if far <= 0.0:
            raise ValueError(f"Far clipping plane distance must be positive, got {far}")
        server = _get_viewer_server_cpp()
        server.set_camera_far(self._name, far)

    @torch.no_grad()
    def set_camera_lookat(
        self,
        eye: NumericMaxRank1,
        center: NumericMaxRank1,
        up: NumericMaxRank1 = [0.0, 1.0, 0.0],
    ):
        """
        Set the camera pose from a camera origin, a lookat point, and an up direction of this scene's camera.

        Args:
            eye (NumericMaxRank1): A tensor-like object of shape (3,) representing the camera position in world coordinates.
            center (NumericMaxRank1): A tensor-like object of shape (3,) representing the point the camera is looking at.
            up (NumericMaxRank1): A tensor-like object of shape (3,) representing the up direction of the camera.
        """
        server = _get_viewer_server_cpp()
        camera_origin_vec3f = to_Vec3f(eye).cpu().numpy()
        lookat_point_vec3f = to_Vec3f(center).cpu().numpy()
        up_direction_vec3f = to_Vec3f(up).cpu().numpy()
        view_direction_vec3f = lookat_point_vec3f - camera_origin_vec3f
        orbit_radius = float(np.linalg.norm(view_direction_vec3f))

        if orbit_radius < 1e-6:
            raise ValueError("Camera origin and lookat point cannot be the same.")
        if np.linalg.norm(view_direction_vec3f) < 1e-6:
            raise ValueError("Camera origin and lookat point cannot be the same.")
        if np.linalg.norm(up_direction_vec3f) < 1e-6:
            raise ValueError("Up direction cannot be a zero vector.")

        view_direction_vec3f /= np.linalg.norm(view_direction_vec3f)
        up_direction_vec3f /= np.linalg.norm(up_direction_vec3f)

        # Check that the view direction is not parallel to the up direction
        dot_product = np.dot(view_direction_vec3f, up_direction_vec3f)
        if abs(dot_product) > 0.999:
            raise ValueError("View direction and up direction cannot be parallel or anti-parallel.")

        server.set_camera_orbit_center(self._name, *lookat_point_vec3f)
        server.set_camera_view_direction(self._name, *view_direction_vec3f)
        server.set_camera_orbit_radius(self._name, orbit_radius)
        server.set_camera_up_direction(self._name, *up_direction_vec3f)
