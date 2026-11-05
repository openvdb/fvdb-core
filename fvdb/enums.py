# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

try:
    from enum import IntEnum, StrEnum
except ImportError:  # Python 3.10 does not provide enum.StrEnum.
    from enum import Enum, IntEnum

    class StrEnum(str, Enum):
        """Compatibility implementation of :class:`enum.StrEnum` for Python 3.10."""

        __str__ = str.__str__


def _to_cpp_enum(py_enum, cpp_enum, value):
    """Map ``value`` to the ``cpp_enum`` member sharing a name with the matching ``py_enum`` member.

    ``value`` may be a ``py_enum`` member, a plain ``int`` holding one of its values, or a member of
    ``cpp_enum`` itself (pybind11 enums compare unequal to ints, so they are coerced through
    ``int()``). Members of other enums, ``bool`` and non-integers are rejected.
    """
    if isinstance(value, py_enum):
        member = value
    elif isinstance(value, cpp_enum):
        member = py_enum(int(value))
    elif isinstance(value, int) and not isinstance(value, (bool, IntEnum)):
        member = py_enum(value)
    else:
        raise TypeError(
            f"expected {py_enum.__name__}, an int value, or {cpp_enum.__name__}, got {type(value).__name__}"
        )
    return getattr(cpp_enum, member.name)


class ConvolutionTopologyPolicy(StrEnum):
    """Policy controlling the finite output topology of a convolution plan."""

    COMPLETE = "complete"
    """Generate the complete, uncropped structural topology."""

    RESTRICTED = "restricted"
    """Evaluate the convolution relation only on an explicit target grid."""


class ConvolutionTopologyProvenance(StrEnum):
    """How a convolution plan's finite topology was obtained."""

    GENERATED = "generated"
    """Generated from the source grid using the canonical convolution relation."""

    EXPLICIT_TARGET = "explicit_target"
    """Restricted to a target grid supplied by the caller."""

    EXACT_TRANSPOSE = "exact_transpose"
    """Reversed directly from another plan's stored finite edge set."""


class ConvolutionPhasePolicy(StrEnum):
    """Kernel phase convention used by a convolution plan."""

    TORCH_SAME_PHASE = "torch_same_phase"
    """Use PyTorch ``padding=floor((kernel_size - 1) / 2)`` phase."""


class SmoothingMode(IntEnum):
    """
    Laplacian smoothing mode used to de-staircase a signed distance field in
    :meth:`fvdb.Grid.reinitialize_sdf` / :meth:`fvdb.Grid.rebuild_narrow_band` (and their
    :class:`fvdb.GridBatch` counterparts).

    The number of smoothing passes is controlled separately by the ``smooth`` argument; this enum
    selects *which* umbrella-Laplacian flow each pass applies. Smoothing runs after the redistance
    and is followed by a short second redistance (with a freshly computed sign) so ``|grad phi| = 1``
    is restored; unlike a pure redistance it therefore *moves* the zero crossing to the smoothed
    surface. Thin (about one voxel) features can be eroded by :attr:`MEAN_CURVATURE`; prefer
    :attr:`TAUBIN` or ``smooth=0`` for them. Values mirror the C++
    ``fvdb::detail::ops::SmoothingMode`` enum.
    """

    MEAN_CURVATURE = 0
    """
    Mean-curvature flow: each pass moves every voxel toward the average of its 6 face neighbours.
    Effective at removing staircase artifacts but shrinks the surface (volume loss) if over-applied.
    """

    TAUBIN = 1
    """
    Volume-preserving Taubin smoothing: alternates a positive (shrinking) and a slightly larger
    negative (inflating) Laplacian step per pass, de-staircasing with much less volume loss.
    """


class RollingShutterType(IntEnum):
    """
    Rolling shutter policy for Gaussian splat camera projection and ray generation.

    Rolling shutter models treat different image rows or columns as having different exposure
    times, interpolating between per-camera start and end poses. Values mirror the C++
    ``fvdb::detail::ops::RollingShutterType`` enum.
    """

    NONE = 0
    """No rolling shutter: the start pose is used for all pixels."""

    VERTICAL = 1
    """Vertical rolling shutter: exposure time varies with image row (y)."""

    HORIZONTAL = 2
    """Horizontal rolling shutter: exposure time varies with image column (x)."""


class CameraModel(IntEnum):
    """
    Camera model for Gaussian splat projection and ray generation.

    ``PINHOLE`` and ``ORTHOGRAPHIC`` ignore distortion coefficients. The ``OPENCV_*`` variants use
    pinhole intrinsics plus OpenCV-style distortion and expect a packed ``[C, 12]`` coefficient
    tensor laid out as ``[k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]``, with unused entries
    set to zero. Values mirror the C++ ``fvdb::detail::ops::DistortionModel`` enum.
    """

    PINHOLE = 0
    """Ideal pinhole camera (no distortion)."""

    OPENCV_RADTAN_5 = 1
    """OpenCV radial-tangential distortion. Uses the packed slots ``k1, k2, k3, p1, p2``; ``k4..k6`` and ``s1..s4`` must be zero."""

    OPENCV_RATIONAL_8 = 2
    """OpenCV rational radial-tangential distortion. Uses the packed slots ``k1..k6, p1, p2``; ``s1..s4`` must be zero."""

    OPENCV_RADTAN_THIN_PRISM_9 = 3
    """OpenCV radial-tangential plus thin-prism distortion. Uses ``k1, k2, k3, p1, p2, s1..s4``; ``k4..k6`` must be zero."""

    OPENCV_THIN_PRISM_12 = 4
    """OpenCV rational radial-tangential plus thin-prism distortion. Uses all twelve packed slots."""

    ORTHOGRAPHIC = 5
    """Orthographic camera (no distortion)."""
