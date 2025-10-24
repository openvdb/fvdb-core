# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any

import torch

from .._Cpp import CameraView as CameraViewCpp
from ..types import NumericMaxRank1, NumericScalarNative, to_Vec3f
from ._viewer_server import _get_viewer_server_cpp


class CamerasView:
    """
    A view for a set of camera frusta and axes in a :class:`fvdb.viz.Scene` with parameters to
    adjust how the cameras are rendered.

    Each camera is represented by its camera-to-world and projection matrices, and drawn as a
    wireframe frustum with orthogonal axes at the camera's origin.
    """

    __PRIVATE__ = object()

    def _get_view(self) -> CameraViewCpp:
        """
        Get the underlying C++ CameraView instance from the viewer server or raise
        a :class:`RuntimeError` if it is not registered.

        Returns:
            view (CameraViewCpp): The C++ CameraView instance
        """
        server = _get_viewer_server_cpp()

        if not server.has_camera_view(self._name):
            raise RuntimeError(f"CameraView '{self._name}' is not registered with the viewer server.")
        return server.get_camera_view(self._name)

    def __init__(
        self,
        scene_name: str,
        name: str,
        camera_to_world_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_sizes: torch.Tensor,
        axis_length: float,
        axis_thickness: float,
        frustum_line_width: float,
        frustum_scale: float,
        frustum_color: tuple[float, float, float],
        frustum_near_plane: float,
        frustum_far_plane: float,
        enabled: bool,
        _private: Any = None,
    ):
        """
        Create a new :class:`CamerasView` or update an existing one within a scene with the given name.

        .. warning::

            This constructor is private and should never be called directly. Use :meth:`fvdb.viz.Scene.add_cameras()` instead.

        Args:
            scene_name (str): The name of the scene the view belongs to.
            name (str): The name of the :class:`CamerasView`.
            camera_to_world_matrices (torch.Tensor): A tensor of shape ``(N, 4, 4)`` representing the camera-to-world matrices for N cameras.
            projection_matrices (torch.Tensor): A tensor of shape ``(N, 4, 4)`` representing the projection matrices for N cameras.
            image_sizes (torch.Tensor): A tensor of shape ``(N, 2)`` representing the image sizes (width, height) for N cameras.
            axis_length (float): The length of the axes drawn at each camera origin in world units.
            axis_thickness (float): The thickness of the axes drawn at each camera origin in pixel units.
            frustum_line_width (float): The line width of the frustum in pixel units.
            frustum_scale (float): The scale factor applied to the frustum visualization.
            frustum_color (tuple[float, float, float]): The color of the frustum lines as an RGB tuple with values in [0, 1].
            frustum_near_plane (float): The near plane distance for the camera frusta.
            frustum_far_plane (float): The far plane distance for the camera frusta.
            enabled (bool): Whether the camera frusta and axes are shown in the viewer.
            _private (Any): A private object to prevent direct construction. Must be :attr:`CamerasView.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("CameraView constructor is private. Use Scene.add_cameras() instead.")

        server = _get_viewer_server_cpp()
        self._scene_name = scene_name
        self._name = name

        view: CameraViewCpp = server.add_camera_view(
            scene_name=scene_name,
            name=name,
            camera_to_world_matrices=camera_to_world_matrices,
            projection_matrices=projection_matrices,
            image_sizes=image_sizes,
            frustum_near_plane=frustum_near_plane,
            frustum_far_plane=frustum_far_plane,
        )

        view.axis_length = axis_length
        view.axis_thickness = axis_thickness
        view.frustum_line_width = frustum_line_width
        view.frustum_scale = frustum_scale
        view.frustum_color = frustum_color
        view.visible = enabled

    @property
    def enabled(self) -> bool:
        """
        Return whether the camera frusta and axes are shown in the scene.

        Returns:
            enabled (bool): ``True`` if the camera frusta and axes are shown in the
                scene, ``False`` otherwise.
        """
        view = self._get_view()
        return view.visible

    @enabled.setter
    def enabled(self, enabled: bool):
        """
        Set whether the camera frusta and axes are shown in the scene.

        Args:
            enabled (bool): ``True`` to show the camera frusta and axes, ``False`` to hide them.
        """
        view = self._get_view()
        view.visible = enabled

    @property
    def axis_length(self) -> float:
        """
        Get the length of the axes drawn at each camera origin in world units.

        Returns:
            length (float): The length of the axes.
        """
        view = self._get_view()
        return view.axis_length

    @axis_length.setter
    def axis_length(self, length: NumericScalarNative):
        """
        Set the length of the axes drawn at each camera origin in world units.

        Args:
            length (NumericScalarNative): The length of the axes.
        """
        view = self._get_view()
        view.axis_length = float(length)

    @property
    def axis_thickness(self) -> float:
        """
        Get the thickness of the axes drawn at each camera origin in pixel units.

        Returns:
            thickness (float): The thickness of the axes.
        """
        view = self._get_view()
        return view.axis_thickness

    @axis_thickness.setter
    def axis_thickness(self, thickness: NumericScalarNative):
        """
        Set the thickness of the axes drawn at each camera origin in pixel units.

        Args:
            thickness (NumericScalarNative): The thickness of the axes.
        """
        view = self._get_view()
        view.axis_thickness = float(thickness)

    @property
    def frustum_line_width(self) -> float:
        """
        Get the line width of the frustum in the camera frustum view.
        """
        view = self._get_view()
        return view.frustum_line_width

    @frustum_line_width.setter
    def frustum_line_width(self, width: NumericScalarNative):
        """
        Set the line width of the frustum in the camera frustum view.

        Args:
            width (NumericScalarNative): The line width of the frustum in world units.
        """
        view = self._get_view()
        view.frustum_line_width = float(width)

    @property
    def frustum_scale(self) -> float:
        """
        Get the scale factor applied to the frustum visualization. Each frustum will have its size
        multiplied by this scale factor when rendered.

        *E.g.* if the frustum has ``near = 0.1``, and ``far = 1.0``, then setting the frustum
        scale to ``2.0`` will render the frustum as if ``near = 0.2`` and ``far = 2.0``.

        Returns:
            scale (float): The scale factor applied to the frustum visualization.
        """
        view = self._get_view()
        return view.frustum_scale

    @frustum_scale.setter
    def frustum_scale(self, scale: NumericScalarNative):
        """
        Set the scale factor applied to the frustum visualization. Each frustum will have its size
        multiplied by this scale factor when rendered.

        *E.g.* if the frustum has ``near = 0.1``, and ``far = 1.0``, then setting the frustum
        scale to ``2.0`` will render the frustum as if ``near = 0.2`` and ``far = 2.0``.

        Args:
            scale (NumericScalarNative): The scale factor to apply to the frustum visualization.
        """
        view = self._get_view()
        view.frustum_scale = float(scale)

    @property
    def frustum_color(self) -> torch.Tensor:
        """
        Get the RGB color of the frustum lines as a tensor of shape ``(3,)`` with
        values in ``[0, 1]``.

        Returns:
            torch.Tensor: The RGB color of the frustum lines.
        """
        view = self._get_view()
        r, g, b = view.frustum_color
        return torch.tensor([r, g, b], dtype=torch.float32)

    @frustum_color.setter
    def frustum_color(self, color: NumericMaxRank1):
        """
        Set the RGB color of the frustum lines. Color values must be in ``[0, 1]``.

        Args:
            color (NumericMaxRank1): A tensor-like object of shape ``(3,)`` representing the
                RGB color of the frustum lines with values in ``[0, 1]``.
        """
        view = self._get_view()
        color_vec3f = to_Vec3f(color).cpu().numpy().tolist()
        if any(c < 0.0 or c > 1.0 for c in color_vec3f):
            raise ValueError(f"Frustum color components must be in [0, 1], got {color_vec3f}")
        view.frustum_color = tuple(color_vec3f)
