# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any

import torch

from ..types import NumericMaxRank1
from ._viewer_server import _get_viewer_server_cpp


class ImageView:
    """
    A view for an RGBA8 image in a :class:`fvdb.viz.Scene`.

    .. note::

        Images are stored as NanoVDB grids on the C++ side. The ImageView provides
        a Python interface for managing the image and updating its contents.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        scene_name: str,
        name: str,
        width: int,
        height: int,
        _private: Any = None,
    ):
        """
        Create a new :class:`ImageView` within a scene with the given name.

        .. warning::

            This constructor is private and should never be called directly.
            Use :meth:`fvdb.viz.Scene.add_image()` instead.

        Args:
            scene_name (str): The name of the scene the view belongs to.
            name (str): The name of the :class:`ImageView`.
            width (int): The width of the image in pixels.
            height (int): The height of the image in pixels.
            _private (Any): A private object to prevent direct construction.
                Must be :attr:`ImageView.__PRIVATE__`.
        """
        if _private is not self.__PRIVATE__:
            raise ValueError("ImageView constructor is private. Use Scene.add_image() instead.")

        self._scene_name = scene_name
        self._name = name
        self._width = width
        self._height = height

    @property
    def name(self) -> str:
        """
        Get the name of the image view.

        Returns:
            name (str): The name of this image view.
        """
        return self._name

    @property
    def scene_name(self) -> str:
        """
        Get the name of the scene this image view belongs to.

        Returns:
            scene_name (str): The name of the scene.
        """
        return self._scene_name

    @property
    def width(self) -> int:
        """
        Get the width of the image in pixels.

        Returns:
            width (int): The image width.
        """
        return self._width

    @property
    def height(self) -> int:
        """
        Get the height of the image in pixels.

        Returns:
            height (int): The image height.
        """
        return self._height

    def update(self, rgba_image: NumericMaxRank1):
        """
        Update the image data displayed in the viewer.

        Args:
            rgba_image (NumericMaxRank1): A 1D uint8 tensor-like object of size
                ``width * height * 4`` containing packed RGBA values.
                Each pixel is represented by 4 consecutive bytes (R, G, B, A)
                with values in [0, 255].
        """
        # Convert to torch tensor
        rgba_tensor = torch.as_tensor(rgba_image)

        if rgba_tensor.dtype != torch.uint8:
            raise TypeError(f"rgba_image must have dtype torch.uint8, got {rgba_tensor.dtype}")
        if rgba_tensor.dim() != 1:
            raise ValueError(f"rgba_image must be a 1D tensor, got {rgba_tensor.dim()}D")
        if rgba_tensor.numel() != self._width * self._height * 4:
            raise ValueError(
                f"rgba_image must have size width * height * 4 = {self._width * self._height * 4}, "
                f"got {rgba_tensor.numel()}"
            )

        server = _get_viewer_server_cpp()
        server.add_image(self._scene_name, self._name, rgba_tensor, self._width, self._height)

