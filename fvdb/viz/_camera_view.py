# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from .._Cpp import CameraView as CameraViewCpp


class CameraView:
    """
    A view for a set of camera frusta and axes in a `Viewer` with parameters to adjust how the cameras are rendered.

    Each camera is represented by its camera-to-world and projection matrices, and drawn as a wireframe frustum
    with orthogonal axes at the camera origin.
    """

    __PRIVATE__ = object()

    def __init__(self, view: CameraViewCpp, _private: Any = None):
        """
        Create a new `CameraView` from a C++ implementation. This constructor is private and should not be called directly.
        Use `Viewer.add_camera_view()` instead.
        """
        self._view = view
        if _private is not self.__PRIVATE__:
            raise ValueError("CameraView constructor is private. Use Viewer.add_camera_view() instead.")

    @property
    def visible(self) -> bool:
        """
        Get whether the camera frusta and axes are shown in the viewer.

        Returns:
            bool: True if the camera frusta and axes are visible, False otherwise.
        """
        return self._view.visible

    @visible.setter
    def visible(self, visible: bool):
        """
        Set whether the camera frusta and axes are shown in the viewer.
        Args:
            visible (bool): True to show the camera frusta and axes, False to hide them
        """
        self._view.visible = visible

    @property
    def axis_length(self) -> float:
        """
        Get the length of the axes drawn at each camera origin in world units.

        Returns:
            float: The length of the axes.
        """
        return self._view.axis_length

    @axis_length.setter
    def axis_length(self, length: float):
        """
        Set the length of the axes drawn at each camera origin in world units.
        Args:
            length (float): The length of the axes.
        """
        self._view.axis_length = length

    @property
    def axis_thickness(self) -> float:
        """
        Get the thickness of the axes drawn at each camera origin in world units.

        Returns:
            float: The thickness of the axes.
        """
        return self._view.axis_thickness

    @axis_thickness.setter
    def axis_thickness(self, thickness: float):
        """
        Set the thickness of the axes drawn at each camera origin in world units.

        Args:
            thickness (float): The thickness of the axes.
        """
        self._view.axis_thickness = thickness

    @property
    def axis_scale(self) -> float:
        return self._view.axis_scale

    @axis_scale.setter
    def axis_scale(self, scale: float):
        self._view.axis_scale = scale

    @property
    def frustum_line_width(self) -> float:
        return self._view.frustum_line_width

    @frustum_line_width.setter
    def frustum_line_width(self, width: float):
        self._view.frustum_line_width = width

    @property
    def frustum_scale(self) -> float:
        return self._view.frustum_scale

    @frustum_scale.setter
    def frustum_scale(self, scale: float):
        self._view.frustum_scale = scale
