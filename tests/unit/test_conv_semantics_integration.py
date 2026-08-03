# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Production-facing issue #668 assertions carried across implementation slices."""

import warnings

import pytest
import torch

from fvdb import ConvolutionPlan, GridBatch, JaggedTensor, _fvdb_cpp
from fvdb.convolution_plan import _GatherScatterBackend
from fvdb.utils.tests.convolution_semantics_oracle import (
    ConvolutionRelation,
    forward_degrees,
    forward_support,
    relation_edges,
    transpose_support,
)


def _grid(coordinates, *, voxel_sizes=1.0, origins=(0.0, 0.0, 0.0), device="cpu") -> GridBatch:
    ijk = torch.tensor(coordinates, dtype=torch.int32, device=device)
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

    strided_plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=2, source_grid=fine)
    assert strided_plan.transform_compatibility.compatible
    torch.testing.assert_close(strided_plan.target_grid_batch.voxel_sizes.cpu(), torch.tensor([[1.0, 2.0, 4.0]]))
    torch.testing.assert_close(strided_plan.target_grid_batch.origins.cpu(), fine.origins.cpu())


def test_explicit_transform_rejects_integer_and_fractional_registration() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=1.0, origins=(0.0, 0.0, 0.0))
    integer_phase = _grid([(0, 0, 0)], voxel_sizes=2.0, origins=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="nonzero integer.*a=0"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=integer_phase,
        )

    fractional_phase = _grid([(0, 0, 0)], voxel_sizes=2.0, origins=(0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="fractional.*a=0"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=fractional_phase,
        )


def test_generated_forward_topology_includes_issue_668_endpoint() -> None:
    fine = _grid([(coordinate, 0, 0) for coordinate in range(16)])
    coarse = fine.conv_grid(kernel_size=(4, 1, 1), stride=(4, 1, 1))
    assert _coordinate_set(coarse) == {(coordinate, 0, 0) for coordinate in range(5)}


def test_generated_transpose_uses_even_torch_phase() -> None:
    coarse = _grid([(0, 0, 0)])
    fine = coarse.conv_transpose_grid(kernel_size=(4, 1, 1), stride=(4, 1, 1))
    assert _coordinate_set(fine) == {(coordinate, 0, 0) for coordinate in (-1, 0, 1, 2)}


_UNIFORM_GEOMETRIES = [
    ((kernel, kernel, kernel), (stride, stride, stride)) for kernel in range(1, 7) for stride in range(1, 6)
]
_MIXED_GEOMETRIES = [((2, 3, 4), (1, 2, 3)), ((5, 2, 3), (4, 2, 1)), ((3, 4, 2), (2, 3, 4))]


@pytest.mark.parametrize(("kernel_size", "stride"), _UNIFORM_GEOMETRIES + _MIXED_GEOMETRIES)
def test_generated_topologies_match_independent_relation_cpu(kernel_size, stride) -> None:
    coordinates = [(-4, -1, 0), (-1, 0, 1), (0, 2, -3), (3, -2, 4), (5, 1, -1)]
    relation = ConvolutionRelation(kernel_size, stride)
    fine = _grid(coordinates)

    coarse = fine.conv_grid(kernel_size=kernel_size, stride=stride)
    assert _coordinate_set(coarse) == forward_support(coordinates, relation)

    generated_fine = coarse.conv_transpose_grid(kernel_size=kernel_size, stride=stride)
    assert _coordinate_set(generated_fine) == transpose_support(forward_support(coordinates, relation), relation)

    participating = {edge.fine for edge in relation_edges(coordinates, relation)}
    assert participating.issubset(_coordinate_set(generated_fine))


@pytest.mark.parametrize(("kernel_size", "stride"), _UNIFORM_GEOMETRIES + _MIXED_GEOMETRIES)
def test_generated_forward_all_one_values_equal_independent_degrees_cpu(kernel_size, stride) -> None:
    coordinates = [(-3, 0, 0), (-1, 1, 0), (0, -1, 1), (2, 0, -1), (5, 2, 1)]
    relation = ConvolutionRelation(kernel_size, stride)
    fine = _grid(coordinates)
    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=kernel_size,
        stride=stride,
        source_grid=fine,
        acknowledge_incomplete_coverage=True,
    )
    features = JaggedTensor(torch.ones((len(coordinates), 1), dtype=torch.float64))
    weights = torch.ones((1, 1, *kernel_size), dtype=torch.float64)
    execution_weights = weights[:, :, 0, 0, 0] if kernel_size == stride == (1, 1, 1) else weights
    values = plan.execute(features, execution_weights).jdata[:, 0].cpu()
    degrees = forward_degrees(coordinates, relation)
    expected = torch.tensor(
        [degrees[tuple(coordinate)] for coordinate in plan.target_grid_batch.ijk.jdata.cpu().tolist()]
    )
    torch.testing.assert_close(values, expected.to(dtype=values.dtype), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("kernel_size", "stride"),
    [
        ((2, 2, 2), (2, 2, 2)),
        ((3, 3, 3), (3, 3, 3)),
        ((4, 4, 4), (4, 4, 4)),
        ((3, 3, 3), (4, 4, 4)),
        ((4, 4, 4), (3, 3, 3)),
        ((1, 1, 1), (3, 2, 1)),
        ((2, 3, 4), (3, 2, 4)),
    ],
)
def test_generated_topology_and_all_one_degrees_cuda(kernel_size, stride) -> None:
    coordinates = [(-4, -1, 0), (-1, 0, 1), (0, 2, -3), (3, -2, 4), (5, 1, -1)]
    relation = ConvolutionRelation(kernel_size, stride)
    fine = _grid(coordinates, device="cuda")
    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=kernel_size,
        stride=stride,
        source_grid=fine,
        acknowledge_incomplete_coverage=True,
    )
    assert _coordinate_set(plan.target_grid_batch) == forward_support(coordinates, relation)
    torch.testing.assert_close(plan.target_grid_batch.voxel_sizes.cpu(), torch.tensor([stride], dtype=torch.float32))
    torch.testing.assert_close(plan.target_grid_batch.origins.cpu(), fine.origins.cpu())

    features = JaggedTensor(torch.ones((len(coordinates), 1), dtype=torch.float64, device="cuda"))
    weights = torch.ones((1, 1, *kernel_size), dtype=torch.float64, device="cuda")
    values = plan.execute(features, weights).jdata[:, 0].cpu()
    degrees = forward_degrees(coordinates, relation)
    expected = torch.tensor(
        [degrees[tuple(coordinate)] for coordinate in plan.target_grid_batch.ijk.jdata.cpu().tolist()]
    )
    torch.testing.assert_close(values, expected.to(dtype=values.dtype), rtol=0, atol=0)

    generated_fine = plan.target_grid_batch.conv_transpose_grid(kernel_size=kernel_size, stride=stride)
    assert _coordinate_set(generated_fine) == transpose_support(forward_support(coordinates, relation), relation)
    torch.testing.assert_close(generated_fine.voxel_sizes.cpu(), fine.voxel_sizes.cpu())
    torch.testing.assert_close(generated_fine.origins.cpu(), fine.origins.cpu())


def test_k1_s1_generated_grid_preserves_public_and_data_identity() -> None:
    fine = _grid([(-1, 0, 0), (0, 0, 0), (1, 0, 0)])
    generated = fine.conv_grid(kernel_size=1, stride=1)
    assert generated is fine
    assert generated.is_same(fine)


def test_k1_strided_forward_is_residue_sampling_not_floor_coarsening() -> None:
    fine = _grid([(-3, 0, 0), (-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    coarse = fine.conv_grid(kernel_size=(1, 1, 1), stride=(3, 1, 1))
    assert _coordinate_set(coarse) == {(-1, 0, 0), (0, 0, 0), (1, 0, 0)}

    uncovered = _grid([(1, 0, 0)])
    empty = uncovered.conv_grid(kernel_size=(1, 1, 1), stride=(2, 1, 1))
    assert empty.total_voxels == 0


def test_topology_policy_is_symmetric_and_reports_exact_output_coverage() -> None:
    fine = _grid([(0, 0, 0)])
    generated = ConvolutionPlan.from_grid_batch(kernel_size=(3, 1, 1), stride=1, source_grid=fine)
    assert generated.topology_policy == "full_support"
    assert generated.coverage_report is not None
    assert generated.coverage_report.output_zero_count == 0

    explicit = _grid([(0, 0, 0), (5, 0, 0)])
    restricted = ConvolutionPlan.from_grid_batch(
        kernel_size=(3, 1, 1), stride=1, source_grid=fine, target_grid=explicit
    )
    assert restricted.topology_policy == "restricted"
    assert restricted.coverage_report is not None
    assert restricted.coverage_report.output_zero_count == 1
    assert restricted.coverage_report.output_degree_histogram == ((0, 1), (1, 1))

    with pytest.raises(ValueError, match="zero-degree output"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=(3, 1, 1),
            stride=1,
            source_grid=fine,
            target_grid=explicit,
            strict_output_coverage=True,
        )

    coarse = _grid([(0, 0, 0)])
    transposed = ConvolutionPlan.from_grid_batch_transposed(
        kernel_size=(2, 1, 1),
        stride=(3, 1, 1),
        source_grid=coarse,
        acknowledge_incomplete_coverage=True,
    )
    assert transposed.topology_policy == "full_support"
    assert _coordinate_set(transposed.target_grid_batch) == {(0, 0, 0), (1, 0, 0)}
    assert transposed.coverage_report is not None
    assert transposed.coverage_report.output_zero_count == 0


def test_topology_policy_rejects_inconsistent_factory_arguments() -> None:
    source = _grid([(0, 0, 0)])
    target = _grid([(0, 0, 0)])
    with pytest.raises(ValueError, match="full_support.*target_grid=None"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=1,
            source_grid=source,
            target_grid=target,
            topology_policy="full_support",
        )
    with pytest.raises(ValueError, match="restricted.*explicit target_grid"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=1,
            source_grid=source,
            topology_policy="restricted",
        )


def test_incomplete_residue_warning_is_proactive_and_acknowledgeable() -> None:
    fine = _grid([(0, 0, 0)])
    with pytest.warns(UserWarning, match="uncovered stride residues"):
        ConvolutionPlan.from_grid_batch(kernel_size=(1, 1, 1), stride=(6, 1, 1), source_grid=fine)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ConvolutionPlan.from_grid_batch(
            kernel_size=(1, 1, 1),
            stride=(7, 1, 1),
            source_grid=fine,
            acknowledge_incomplete_coverage=True,
        )
    assert not caught


def test_issue_668_16_cubed_production_round_trip() -> None:
    coordinates = list(torch.cartesian_prod(*(torch.arange(16, dtype=torch.int32) for _ in range(3))).tolist())
    fine = _grid(coordinates)
    coarse = fine.conv_grid(kernel_size=4, stride=4)
    assert coarse.total_voxels == 5**3
    round_trip = coarse.conv_transpose_grid(kernel_size=4, stride=4)
    assert set(map(tuple, coordinates)).issubset(_coordinate_set(round_trip))


def test_forward_builder_exposes_exact_staging_accounting() -> None:
    coordinates = [(-4, 0, 0), (-1, 0, 0), (0, 0, 0), (3, 0, 0), (4, 0, 0)]
    fine = _grid(coordinates)

    fine.conv_grid(kernel_size=3, stride=3)
    direct = _fvdb_cpp.last_conv_grid_resource_stats()
    assert direct["input_voxel_count"] == len(coordinates)
    assert direct["kernel_volume"] == 27
    assert direct["valid_emission_count"] == len(coordinates)
    assert direct["used_direct_projection"] is True
    assert direct["peak_requested_bytes"] == direct["emission_requested_bytes"] == 16 * len(coordinates)

    relation = ConvolutionRelation((3, 1, 1), (4, 1, 1))
    fine.conv_grid(kernel_size=relation.kernel_size, stride=relation.stride)
    staged = _fvdb_cpp.last_conv_grid_resource_stats()
    expected_emissions = len(relation_edges(coordinates, relation))
    assert staged["valid_emission_count"] == expected_emissions
    assert staged["valid_emission_count"] < staged["input_voxel_count"] * staged["kernel_volume"]
    assert staged["used_direct_projection"] is False
    assert staged["peak_requested_bytes"] == max(
        staged["count_requested_bytes"] + staged["prefix_requested_bytes"],
        staged["prefix_requested_bytes"] + staged["emission_requested_bytes"],
    )


def test_generated_forward_grid_uses_convolution_lattice_transform() -> None:
    ijk = JaggedTensor(
        [
            torch.tensor([[-2, 0, 1], [1, 2, -1]], dtype=torch.int32),
            torch.tensor([[0, -1, 3]], dtype=torch.int32),
        ]
    )
    voxel_sizes = torch.tensor([[0.5, 1.0, 2.0], [1.25, 0.25, 0.75]])
    origins = torch.tensor([[3.0, -2.0, 7.0], [-4.0, 5.0, 0.5]])
    stride = (2, 3, 4)
    fine = GridBatch.from_ijk(ijk, voxel_sizes=voxel_sizes, origins=origins)
    coarse = fine.conv_grid(kernel_size=(3, 4, 2), stride=stride)
    torch.testing.assert_close(coarse.voxel_sizes.cpu(), voxel_sizes * torch.tensor(stride))
    torch.testing.assert_close(coarse.origins.cpu(), origins)

    restored = coarse.conv_transpose_grid(kernel_size=(3, 4, 2), stride=stride)
    torch.testing.assert_close(restored.voxel_sizes.cpu(), voxel_sizes)
    torch.testing.assert_close(restored.origins.cpu(), origins)

    coarse_coordinates = JaggedTensor([torch.tensor([[1.0, -2.0, 3.0]]), torch.tensor([[-1.0, 4.0, 2.0]])])
    fine_coordinates = coarse_coordinates * torch.tensor(stride)
    torch.testing.assert_close(
        coarse.voxel_to_world(coarse_coordinates).jdata.cpu(),
        fine.voxel_to_world(fine_coordinates).jdata.cpu(),
    )


def test_valid_explicit_forward_and_transpose_transforms_are_accepted() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=(0.5, 1.0, 2.0), origins=(3.0, -2.0, 7.0))
    coarse = _grid([(0, 0, 0)], voxel_sizes=(1.0, 3.0, 8.0), origins=(3.0, -2.0, 7.0))
    forward = ConvolutionPlan.from_grid_batch(
        kernel_size=(3, 4, 2),
        stride=(2, 3, 4),
        source_grid=fine,
        target_grid=coarse,
        acknowledge_incomplete_coverage=True,
    )
    transposed = ConvolutionPlan.from_grid_batch_transposed(
        kernel_size=(3, 4, 2),
        stride=(2, 3, 4),
        source_grid=coarse,
        target_grid=fine,
        acknowledge_incomplete_coverage=True,
    )
    assert forward.transform_compatibility.compatible
    assert transposed.transform_compatibility.compatible


def test_incompatible_explicit_transform_fails_before_topology_build(monkeypatch) -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=1.0)
    incompatible_coarse = _grid([(0, 0, 0)], voxel_sizes=1.0)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("topology construction ran before transform validation")

    monkeypatch.setattr(ConvolutionPlan, "_build_backend", staticmethod(fail_if_called))
    with pytest.raises(ValueError, match="voxel size"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=incompatible_coarse,
        )


def test_explicit_transform_rejects_batch_mismatch_and_coarsening_contract() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=1.0)
    batched_target = GridBatch.from_ijk(
        JaggedTensor(
            [
                torch.tensor([[0, 0, 0]], dtype=torch.int32),
                torch.tensor([[0, 0, 0]], dtype=torch.int32),
            ]
        ),
        voxel_sizes=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="same batch size"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=batched_target,
        )

    with pytest.raises(ValueError, match="fractional.*coarsened_grid"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=fine.coarsened_grid(2),
        )
    with pytest.raises(ValueError, match="nonzero integer.*coarsened_grid"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=3,
            source_grid=fine,
            target_grid=fine.coarsened_grid(3),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_explicit_transform_rejects_device_mismatch() -> None:
    fine = _grid([(0, 0, 0)], voxel_sizes=1.0)
    coarse = _grid([(0, 0, 0)], voxel_sizes=2.0, device="cuda")
    with pytest.raises(ValueError, match="same device"):
        ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=2,
            source_grid=fine,
            target_grid=coarse,
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
