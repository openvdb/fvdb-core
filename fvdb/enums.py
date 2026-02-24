# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum, IntEnum


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


class RollingShutterType(IntEnum):
    """
    Rolling shutter policy for camera projection / ray generation.

    Rolling shutter models treat different image rows/columns as having different exposure times.
    FVDB uses this to interpolate between per-camera start/end poses when generating rays.
    """

    NONE = 0
    """
    No rolling shutter: the start pose is used for all pixels.
    """

    VERTICAL = 1
    """
    Vertical rolling shutter: exposure time varies with image row (y).
    """

    HORIZONTAL = 2
    """
    Horizontal rolling shutter: exposure time varies with image column (x).
    """


class CameraModel(IntEnum):
    """
    Camera model for projection / ray generation.

    Notes:
    - ``PINHOLE`` and ``ORTHOGRAPHIC`` ignore distortion coefficients.
    - ``OPENCV_*`` variants use pinhole intrinsics plus OpenCV-style distortion. When distortion
      coefficients are provided, FVDB expects a packed layout:

      ``[k1,k2,k3,k4,k5,k6,p1,p2,s1,s2,s3,s4]``

      Unused coefficients for a given model should be set to 0.
    """

    PINHOLE = 0
    """
    Ideal pinhole camera model (no distortion).
    """

    OPENCV_RADTAN_5 = 1
    """
    OpenCV radial-tangential distortion with 5 parameters (k1,k2,p1,p2,k3).
    """

    OPENCV_RATIONAL_8 = 2
    """
    OpenCV rational radial-tangential distortion with 8 parameters (k1..k6,p1,p2).
    """

    OPENCV_RADTAN_THIN_PRISM_9 = 3
    """
    OpenCV radial-tangential + thin-prism distortion with 9 parameters (k1,k2,p1,p2,k3,s1..s4).
    """

    OPENCV_THIN_PRISM_12 = 4
    """
    OpenCV rational radial-tangential + thin-prism distortion with 12 parameters
    (k1..k6,p1,p2,s1..s4).
    """

    ORTHOGRAPHIC = 5
    """
    Orthographic camera model (no distortion).
    """
