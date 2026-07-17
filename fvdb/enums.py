# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from enum import IntEnum


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
