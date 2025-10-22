# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum


class ProjectionType(str, Enum):
    """
    Enum representing camera projection types. Used in :class:`fvdb.GaussianSplat3d`, and
    :mod:`fvdb.viz`.
    """

    PERSPECTIVE = "perspective"
    """
    Perspective projection type.
    """

    ORTHOGRAPHIC = "orthographic"
    """
    Orthographic projection type.
    """


class ShOrderingMode(str, Enum):
    """
    Enum representing spherical harmonics ordering modes used by Gaussian Splats..
    Spherical harmonics for Gaussian splatting can be stored differently in memory depending on the application. For example,
    PLY files store spherical harmonics in ``RRR_GGG_BBB`` order, while some rendering codes
    (including :class:`fvdb.GaussianSplat3d`) use ``RGB_RGB_RGB`` order.

    This enum defines two common ordering modes:

    - ``RGB_RGB_RGB``: The feature channels are interleaved for each coefficient. *i.e.* The spherical harmonics
      tensor corresponds to a (row-major) contiguous tensor of shape ``[num_coefficients, num_sh_bases, channels]``, where channels=3 for RGB.
    - ``RRR_GGG_BBB``: The feature channels are stored in separate blocks for each coefficient. *i.e.* The spherical harmonics
      tensor corresponds to a (row-major) contiguous tensor of shape ``[num_coefficients, channels, num_sh_bases]``, where channels=3 for RGB.
    """

    RGB_RGB_RGB = "rgb_rgb_rgb"
    """
    The feature channels of spherical harmonics are interleaved for each coefficient. *i.e.* The spherical harmonics
    tensor corresponds to a (row-major) contiguous tensor of shape ``[num_coefficients, num_sh_bases, channels]``, where channels=3 for RGB.
    """

    RRR_GGG_BBB = "rrr_ggg_bbb"
    """
    The feature channels of spherical harmonics are stored in separate blocks for each coefficient. *i.e.* The spherical harmonics
    tensor corresponds to a (row-major) contiguous tensor of shape ``[num_coefficients, channels, num_sh_bases]``, where channels=3 for RGB.
    """
