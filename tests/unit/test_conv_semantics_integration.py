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


def test_plan_stores_canonical_geometry_and_reports_transform_compatibility() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=(0.5, 1.0, 2.0), origins=(3.0, -2.0, 7.0))
    plan = ConvolutionPlan.from_grid_batch(kernel_size=(4, 3, 2), stride=1, source_grid=fine)

    assert plan.geometry.phase_policy == "torch_same_phase"
    assert plan.geometry.semantics_version == 1
    assert plan.geometry.kernel_size == [4, 3, 2]
    assert plan.geometry.padding_before == [1, 1, 0]
    assert plan.geometry.padding_after == [2, 1, 1]
    assert plan.geometry.dilation == [1, 1, 1]
    assert plan.geometry.registration_offset == [0, 0, 0]
    assert plan.geometry.kernel_volume == 24
    assert plan.transform_compatibility.compatible

    legacy_strided_plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=2, source_grid=fine)
    assert not legacy_strided_plan.transform_compatibility.scale_compatible
    assert not legacy_strided_plan.transform_compatibility.compatible


def test_transform_report_distinguishes_integer_and_fractional_registration() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=1.0, origins=(0.0, 0.0, 0.0))
    integer_phase = _grid([(0, 0, 0)], voxel_sizes=2.0, origins=(1.0, 0.0, 0.0))
    integer_plan = ConvolutionPlan.from_grid_batch(
        kernel_size=3,
        stride=2,
        source_grid=fine,
        target_grid=integer_phase,
    )
    assert integer_plan.transform_compatibility.scale_compatible
    assert integer_plan.transform_compatibility.registration_integer
    assert not integer_plan.transform_compatibility.registration_zero
    assert not integer_plan.transform_compatibility.compatible

    fractional_phase = _grid([(0, 0, 0)], voxel_sizes=2.0, origins=(0.5, 0.0, 0.0))
    fractional_plan = ConvolutionPlan.from_grid_batch(
        kernel_size=3,
        stride=2,
        source_grid=fine,
        target_grid=fractional_phase,
    )
    assert fractional_plan.transform_compatibility.scale_compatible
    assert not fractional_plan.transform_compatibility.registration_integer
    assert not fractional_plan.transform_compatibility.compatible


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
