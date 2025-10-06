# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any

import torch

from .._Cpp import CameraView as CameraViewCpp
from ..types import NumericMaxRank1, NumericScalarNative, to_Vec3f


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
    def axis_length(self, length: NumericScalarNative):
        """
        Set the length of the axes drawn at each camera origin in world units.
        Args:
            length (NumericScalarNative): The length of the axes.
        """
        self._view.axis_length = float(length)

    @property
    def axis_thickness(self) -> float:
        """
        Get the thickness of the axes drawn at each camera origin in world units.

        Returns:
            float: The thickness of the axes.
        """
        return self._view.axis_thickness

    @axis_thickness.setter
    def axis_thickness(self, thickness: NumericScalarNative):
        """
        Set the thickness of the axes drawn at each camera origin in world units.

        Args:
            thickness (NumericScalarNative): The thickness of the axes.
        """
        self._view.axis_thickness = float(thickness)

    @property
    def frustum_line_width(self) -> float:
        """
        Get the line width of the frustum in the camera frustum view.
        """
        return self._view.frustum_line_width

    @frustum_line_width.setter
    def frustum_line_width(self, width: NumericScalarNative):
        """
        Set the line width of the frustum in the camera frustum view.

        Args:
            width (NumericScalarNative): The line width of the frustum in world units.
        """
        self._view.frustum_line_width = float(width)

    @property
    def frustum_scale(self) -> float:
        """
        Get the scale factor applied to the frustum visualization.
        """
        return self._view.frustum_scale

    @frustum_scale.setter
    def frustum_scale(self, scale: NumericScalarNative):
        """
        Set the scale factor applied to the frustum visualization.

        Args:
            scale (NumericScalarNative): The scale factor to apply to the frustum visualization.
        """
        self._view.frustum_scale = float(scale)

    @property
    def frustum_color(self) -> torch.Tensor:
        """
        Get the color of the frustum lines as a tensor of shape (3,) with values in [0, 1].

        Returns:
            torch.Tensor: The color of the frustum lines.
        """
        r, g, b = self._view.frustum_color
        return torch.tensor([r, g, b], dtype=torch.float32)

    @frustum_color.setter
    def frustum_color(self, color: NumericMaxRank1):
        """
        Set the color of the frustum lines.

        Args:
            color (NumericMaxRank1): A tensor-like object of shape (3,) representing the color of the frustum lines
                with values in [0, 1].
        """
        color_vec3f = to_Vec3f(color).cpu().numpy().tolist()
        if any(c < 0.0 or c > 1.0 for c in color_vec3f):
            raise ValueError(f"Frustum color components must be in [0, 1], got {color_vec3f}")
        self._view.frustum_color = tuple(color_vec3f)
