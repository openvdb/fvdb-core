# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Production-facing issue #668 assertions carried across implementation slices."""

import pytest
import torch

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor
from fvdb.convolution_plan import _GatherScatterBackend


def _grid(coordinates, *, voxel_sizes=1.0, origins=(0.0, 0.0, 0.0)) -> GridBatch:
    ijk = torch.tensor(coordinates, dtype=torch.int32)
    return GridBatch.from_ijk(JaggedTensor(ijk), voxel_sizes=voxel_sizes, origins=origins)


def _coordinate_set(grid: GridBatch) -> set[tuple[int, int, int]]:
    return {tuple(coordinate) for coordinate in grid.ijk.jdata.cpu().tolist()}


@pytest.mark.conv_semantics_pending(slice="3a", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 3a")
def test_generated_forward_topology_includes_issue_668_endpoint() -> None:
    fine = _grid([(coordinate, 0, 0) for coordinate in range(16)])
    coarse = fine.conv_grid(kernel_size=(4, 1, 1), stride=(4, 1, 1))
    assert _coordinate_set(coarse) == {(coordinate, 0, 0) for coordinate in range(5)}


@pytest.mark.conv_semantics_pending(slice="3a", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 3a")
def test_generated_transpose_uses_even_torch_phase() -> None:
    coarse = _grid([(0, 0, 0)])
    fine = coarse.conv_transpose_grid(kernel_size=(4, 1, 1), stride=(4, 1, 1))
    assert _coordinate_set(fine) == {(coordinate, 0, 0) for coordinate in (-1, 0, 1, 2)}


@pytest.mark.conv_semantics_pending(slice="3b", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 3b")
def test_generated_forward_grid_uses_convolution_lattice_transform() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=(0.5, 1.0, 2.0), origins=(3.0, -2.0, 7.0))
    coarse = fine.conv_grid(kernel_size=(3, 3, 3), stride=(2, 3, 4))
    torch.testing.assert_close(coarse.voxel_sizes.cpu(), torch.tensor([[1.0, 3.0, 8.0]]))
    torch.testing.assert_close(coarse.origins.cpu(), torch.tensor([[3.0, -2.0, 7.0]]))


@pytest.mark.conv_semantics_pending(slice="3b", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 3b")
def test_incompatible_explicit_transform_fails_before_topology_build() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=1.0)
    incompatible_coarse = _grid([(0, 0, 0)], voxel_sizes=1.0)
    with pytest.raises(ValueError, match="voxel size"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=incompatible_coarse,
        )


@pytest.mark.conv_semantics_pending(slice="5", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 5")
def test_matmul_backend_requires_shared_grid_data() -> None:
    source = _grid([(0, 0, 0), (1, 0, 0)])
    distinct_equal_target = _grid([(0, 0, 0), (1, 0, 0)])
    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=1,
        stride=1,
        source_grid=source,
        target_grid=distinct_equal_target,
    )
    assert isinstance(plan._backend, _GatherScatterBackend)


@pytest.mark.conv_semantics_pending(slice="5", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 5")
def test_dense_backend_rejects_unsupported_geometry() -> None:
    grid = _grid([(0, 0, 0)])
    with pytest.raises(ValueError, match="disabled"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=5,
            stride=1,
            source_grid=grid,
            expert_config={"backend": "dense"},
        )
