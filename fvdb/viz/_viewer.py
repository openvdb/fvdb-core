# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import warnings

import numpy as np
import torch

from .._Cpp import CameraView as CameraViewCpp
from .._Cpp import GaussianSplat3d as GaussianSplat3dCpp
from .._Cpp import GaussianSplat3dView as GaussianSplat3dViewCpp
from .._Cpp import Viewer as ViewerCpp
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
from ._camera_view import CameraView
from ._gaussian_splat_3d_view import GaussianSplat3dView
from ._point_cloud_view import PointCloudView

_running_viewers = dict()


class Viewer:
    def __init__(
        self,
        name: str = "fVDB Viewer",
        port: int = 8080,
        ip_address: str = "127.0.0.1",
        device_id: int = 0,
        verbose: bool = False,
    ):
        """
        Create a `Viewer` with the given name running a server at the specified IP address and port.
        If a viewer was previously created with the same name, this will return a reference to the existing viewer.

        Args:
            name (str): The name of the viewer instance. Default is "fVDB Viewer". This name is used to
                identify the viewer instance in the global registry of running viewers.
            port (int): The port to bind the viewer server to. Default is 8080.
            ip_address (str): The IP address to bind the viewer server to. Default is "127.0.0.1".
            device_id (int): The ID of the device to use for rendering. Default is 0.
            verbose (bool): If True, the viewer will print verbose output to the console. Default is False.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._name = name
        if name in _running_viewers:
            assert isinstance(_running_viewers[name], dict)  # for mypy
            self._impl = _running_viewers[name]["viewer"]
            self._logger.debug(f"Found existing viewer with refcount {_running_viewers[name]['refcount']}")
            _running_viewers[name]["refcount"] += 1
            self._logger.debug(f"Incremented refcount to {_running_viewers[name]['refcount']}")
            if self._impl.ip_address() != ip_address:
                warnings.warn(
                    f"You requested a viewer with name '{name}' at IP address '{ip_address}', but a viewer with that name is already running at IP address '{self._impl.ip_address()}'. Using the existing viewer."
                )
            if self._impl.port() != port:
                warnings.warn(
                    f"You requested a viewer with name '{name}' at port '{port}', but a viewer with that name is already running at port '{self._impl.port()}'. Using the existing viewer."
                )
            self._logger.info(f"Returning existing viewer running on {self.ip_address}:{self.port}...")
            return

        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError(f"Port must be an integer between 0 and 65535, got {port}")

        _running_viewers[name] = {
            "viewer": ViewerCpp(ip_address=ip_address, port=port, device_id=device_id, verbose=verbose),
            "refcount": 1,
        }
        self._impl = _running_viewers[name]["viewer"]

        self._logger.info(f"Viewer running on {self.ip_address}:{self.port}...")

    def __del__(self):
        if not self._name in _running_viewers:
            return
        else:
            _running_viewers[self._name]["refcount"] -= 1
            self._logger.debug(f"Decremented refcount to {_running_viewers[self._name]['refcount']}")
            if _running_viewers[self._name]["refcount"] <= 0:
                self._logger.debug("Refcount is zero, deleting viewer...")
                del _running_viewers[self._name]

    def _show_in_notebook_if_available(self) -> bool:
        """
        If running in a Jupyter notebook, display the viewer in an iframe.

        Returns:
            bool: True if the viewer was displayed in a Jupyter notebook, False otherwise.
        """
        try:
            from IPython import get_ipython
            from IPython.display import IFrame, display

            if get_ipython() is not None:
                display(IFrame(src=f"http://{self.ip_address}:{self.port}", width="100%", height="600px"))
                return True
        except ImportError:
            pass
        return False

    def show(self, force_browser: bool = False):
        """
        Open the viewer in a new browser window or tab, or in an iframe if running in a Jupyter notebook.

        Args:
            force_browser (bool): If True, always open the viewer in a new browser window or tab,
                even if running in a Jupyter notebook. Default is False.
        """
        import webbrowser

        if not force_browser and self._show_in_notebook_if_available():
            return
        webbrowser.open_new_tab(f"http://{self.ip_address}:{self.port}")

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

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        means = points
        quats = torch.zeros((points.shape[0], 4), dtype=torch.float32)
        quats[:, 0] = 1.0  # identity rotation
        logit_opacities = torch.full((points.shape[0],), 10.0, dtype=torch.float32)
        log_scales = torch.full((points.shape[0], 3), -20.0, dtype=torch.float32)  # since scales are exp(log_scale)
        sh0 = _rgb_to_sh(colors)
        shN = torch.zeros((points.shape[0], 0, 3), dtype=torch.float32)

        gs_impl = GaussianSplat3d(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opacities,
            sh0=sh0,
            shN=shN,
        )._impl
        view: GaussianSplat3dViewCpp = self._impl.add_gaussian_splat_3d(name=name, gaussian_splat_3d=gs_impl)
        view.tile_size = 16
        view.min_radius_2d = 0.0
        view.eps_2d = point_size / 2.0  # point size is diameter
        view.antialias = False
        view.sh_degree_to_use = 0
        return PointCloudView(view, PointCloudView.__PRIVATE__)

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
        gs_impl: GaussianSplat3dCpp = gaussian_splat_3d._impl
        view: GaussianSplat3dViewCpp = self._impl.add_gaussian_splat_3d(name=name, gaussian_splat_3d=gs_impl)
        view.tile_size = tile_size
        view.min_radius_2d = min_radius_2d
        view.eps_2d = eps_2d
        view.antialias = antialias
        view.sh_degree_to_use = sh_degree_to_use
        return GaussianSplat3dView(view, GaussianSplat3dView.__PRIVATE__)

    @property
    def port(self) -> int:
        """
        The port the viewer server is listening on.

        Returns:
            int: The port number.
        """
        return self._impl.port()

    @property
    def ip_address(self) -> str:
        """
        The IP address the viewer server is listening on.

        Returns:
            str: The IP address.
        """
        return self._impl.ip_address()

    @torch.no_grad()
    def add_camera_view(
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
    ) -> CameraView:
        """
        Add CameraView to the viewer and return a view.

        Args:
            name (str): The name of the camera view.
            camera_to_world_matrices (NumericMaxRank3): The 4x4 camera to world transformation matrices (one per camera) encoded
                as a tensor-like object of shape (N, 4, 4) where N is the number of cameras.
            projection_matrices (NumericMaxRank3 | None): The 3x3 projection matrices (one per camera) encoded
                as a tensor-like object of shape (N, 3, 3) where N is the number of cameras.
            image_sizes (NumericMaxRank2 | None): The image sizes as a tensor of shape (N, 2) where N is the number of cameras.
                such that height_i, width_i = image_sizes[i] is the resolution of the i-th camera.
                If None, the image sizes will be inferred from the projection matrices assuming square pixels and
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

        frustum_color = to_Vec3f(frustum_color).cpu().numpy()
        if any(c < 0.0 or c > 1.0 for c in frustum_color):
            raise ValueError(f"Frustum color must be a sequence of three floats in [0, 1], got {frustum_color}")

        camera_to_world_matrices = to_Mat44fBatch(camera_to_world_matrices)
        projection_matrices = to_Mat33fBatch(projection_matrices)
        image_sizes = to_Vec2fBatch(image_sizes) if image_sizes is not None else torch.Tensor([])

        view: CameraViewCpp = self._impl.add_camera_view(
            name,
            camera_to_world_matrices,
            projection_matrices,
            image_sizes,
            frustum_near_plane,
            frustum_far_plane,
        )
        view.visible = enabled
        view.axis_length = axis_length
        view.axis_thickness = axis_thickness
        view.frustum_line_width = frustum_line_width
        view.frustum_scale = frustum_scale
        view.frustum_color = (float(frustum_color[0]), float(frustum_color[1]), float(frustum_color[2]))
        return CameraView(view, CameraView.__PRIVATE__)

    @property
    def camera_orbit_center(self) -> torch.Tensor:
        """
        Return center of the camera orbit in world coordinates.

        Returns:
            torch.Tensor: A tensor of shape (3,) representing the camera orbit center in world coordinates.
        """
        ox, oy, oz = self._impl.camera_orbit_center()
        return torch.tensor([ox, oy, oz], dtype=torch.float32)

    @camera_orbit_center.setter
    def camera_orbit_center(self, center: NumericMaxRank1):
        """
        Set the center of the camera orbit in world coordinates.

        Args:
            center (NumericMaxRank1): A tensor-like object of shape (3,) representing the camera orbit center in world coordinates.
        """
        center_vec3f = to_Vec3f(center).cpu().numpy().tolist()
        self._impl.set_camera_orbit_center(*center_vec3f)

    @property
    def camera_orbit_radius(self) -> float:
        """
        Return the radius of the camera orbit.

        Returns:
            float: The radius of the camera orbit.
        """
        return self._impl.camera_orbit_radius()

    @camera_orbit_radius.setter
    def camera_orbit_radius(self, radius: float):
        """
        Set the radius of the camera orbit.

        Args:
            radius (float): The radius of the camera orbit.
        """
        if radius <= 0.0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self._impl.set_camera_orbit_radius(radius)

    @property
    def camera_orbit_direction(self) -> torch.Tensor:
        """
        Return the direction pointing from the orbit center to the camera position.

        Note: The camera itself is positioned at:
            camera_position = orbit_center + orbit_radius * orbit_direction

        Returns:
            torch.Tensor: A tensor of shape (3,) representing the direction pointing from the orbit
                center to the camera position.
        """
        dx, dy, dz = self._impl.camera_view_direction()
        return torch.tensor([dx, dy, dz], dtype=torch.float32)

    @camera_orbit_direction.setter
    def camera_orbit_direction(self, direction: NumericMaxRank1):
        """
        Set the direction pointing from the orbit center to the camera position.

        Note: The camera itself is positioned at:
            camera_position = orbit_center + orbit_radius * orbit_direction

        Args:
            direction (NumericMaxRank1): A tensor-like object of shape (3,) representing the direction pointing from the orbit
                center to the camera position.
        """
        dir_vec3f = to_Vec3f(direction).cpu().numpy()
        if np.linalg.norm(dir_vec3f) < 1e-6:
            raise ValueError("Camera orbit direction cannot be a zero vector.")
        dir_vec3f /= np.linalg.norm(dir_vec3f)
        self._impl.set_camera_view_direction(*dir_vec3f)

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
        up_vec3f = to_Vec3f(up).cpu().numpy()
        if np.linalg.norm(up_vec3f) < 1e-6:
            raise ValueError("Camera up direction cannot be a zero vector.")
        up_vec3f /= np.linalg.norm(up_vec3f)
        self._impl.set_camera_up_direction(*up_vec3f)

    @property
    def camera_near(self) -> float:
        """
        Get the near clipping plane distance for rendering. Splats closer to the camera than this distance
        will not be rendered.

        Returns:
            float: The near clipping plane distance.
        """
        return self._impl.camera_near()

    @camera_near.setter
    def camera_near(self, near: float):
        """
        Sets the near clipping plane distance for rendering. Splats closer to the camera than this distance
        will not be rendered.

        Args:
            near (float): The near clipping plane distance.
        """
        if near <= 0.0:
            raise ValueError(f"Near clipping plane distance must be positive, got {near}")
        self._impl.set_camera_near(near)

    @property
    def camera_far(self) -> float:
        """
        Get the far clipping plane distance for rendering. Splats farther from the camera than this distance
        will not be rendered.

        Returns:
            float: The far clipping plane distance.
        """
        return self._impl.camera_far()

    @camera_far.setter
    def camera_far(self, far: float):
        """
        Set the far clipping plane distance for rendering. Splats farther from the camera than this distance
        will not be rendered.

        Args:
            far (float): The far clipping plane distance.
        """
        if far <= 0.0:
            raise ValueError(f"Far clipping plane distance must be positive, got {far}")
        self._impl.set_camera_far(far)

    @torch.no_grad()
    def set_camera_lookat(
        self,
        eye: NumericMaxRank1,
        center: NumericMaxRank1,
        up: NumericMaxRank1 = [0.0, 1.0, 0.0],
    ):
        """
        Set the camera pose from a camera origin, a lookat point, and an up direction.

        Args:
            eye (NumericMaxRank1): A tensor-like object of shape (3,) representing the camera position in world coordinates.
            center (NumericMaxRank1): A tensor-like object of shape (3,) representing the point the camera is looking at.
            up (NumericMaxRank1): A tensor-like object of shape (3,) representing the up direction of the camera.
        """
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

        self._impl.set_camera_orbit_center(*lookat_point_vec3f)
        self._impl.set_camera_view_direction(*view_direction_vec3f)
        self._impl.set_camera_orbit_radius(orbit_radius)
        self._impl.set_camera_up_direction(*up_direction_vec3f)
