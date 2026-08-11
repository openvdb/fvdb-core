# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from enum import IntEnum, StrEnum


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
    :meth:`fvdb.Grid.reinitialize_sdf` / :meth:`fvdb.Grid.retopologize_sdf` (and their
    :class:`fvdb.GridBatch` counterparts).

    The number of smoothing passes is controlled separately by the ``smooth`` argument; this enum
    selects *which* umbrella-Laplacian flow each pass applies. Values mirror the C++
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
