# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for canonical sparse-convolution semantics."""

import subprocess
import sys
import textwrap
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from parameterized import parameterized

import fvdb.convolution_plan as convolution_plan_module
from fvdb import (
    ConvolutionCoverageWarning,
    ConvolutionPhasePolicy,
    ConvolutionPlan,
    ConvolutionTopologyPolicy,
    ConvolutionTopologyProvenance,
    Grid,
    GridBatch,
    JaggedTensor,
    _fvdb_cpp,
)
from fvdb.convolution_plan import _GatherScatterBackend, _MatmulBackend, _PredGatherIGemmBackend
from fvdb.utils.tests.convolution_semantics_oracle import (
    ConvolutionRelation,
    dense_forward_oracle,
    dense_transpose_oracle,
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


def _dense_rows(result, coordinates) -> torch.Tensor:
    return torch.stack([result.value_at(tuple(coordinate)) for coordinate in coordinates])


def _gather_scatter_topology(plan: ConvolutionPlan) -> _fvdb_cpp.GatherScatterDefaultTopology:
    assert isinstance(plan._backend, _GatherScatterBackend)
    return plan._backend.topology


def _topology_edges(topology: _fvdb_cpp.GatherScatterDefaultTopology) -> set[tuple[int, int, int]]:
    gather = topology.gather_indices.cpu()
    scatter = topology.scatter_indices.cpu()
    offsets = topology.offsets.cpu()
    return {
        (tap, int(gather[edge].item()), int(scatter[edge].item()))
        for tap in range(topology.kernel_volume)
        for edge in range(int(offsets[tap].item()), int(offsets[tap + 1].item()))
    }


@pytest.fixture(autouse=True)
def _reset_coverage_warning_state():
    convolution_plan_module._WARNED_INCOMPLETE_COVERAGE_GEOMETRIES.clear()
    yield
    convolution_plan_module._WARNED_INCOMPLETE_COVERAGE_GEOMETRIES.clear()


class TestConvSemanticsIntegration(unittest.TestCase):
    def test_plan_stores_canonical_geometry_and_reports_transform_compatibility(self) -> None:
        fine = _grid([(0, 0, 0)], voxel_sizes=(0.5, 1.0, 2.0), origins=(3.0, -2.0, 7.0))
        plan = ConvolutionPlan.from_grid_batch(kernel_size=(4, 3, 2), stride=1, source_grid=fine)

        assert plan.geometry.phase_policy == "torch_same_phase"
        assert plan.phase_policy is ConvolutionPhasePolicy.TORCH_SAME_PHASE
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

    def test_explicit_transform_rejects_integer_and_fractional_registration(self) -> None:
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

    def test_generated_forward_topology_includes_issue_668_endpoint(self) -> None:
        fine = _grid([(coordinate, 0, 0) for coordinate in range(16)])
        coarse = fine.conv_grid(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        assert _coordinate_set(coarse) == {(coordinate, 0, 0) for coordinate in range(5)}

    def test_generated_transpose_uses_even_torch_phase(self) -> None:
        coarse = _grid([(0, 0, 0)])
        fine = coarse.conv_transpose_grid(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        assert _coordinate_set(fine) == {(coordinate, 0, 0) for coordinate in (-1, 0, 1, 2)}

    _UNIFORM_GEOMETRIES = [
        ((kernel, kernel, kernel), (stride, stride, stride)) for kernel in range(1, 7) for stride in range(1, 6)
    ]
    _MIXED_GEOMETRIES = [((2, 3, 4), (1, 2, 3)), ((5, 2, 3), (4, 2, 1)), ((3, 4, 2), (2, 3, 4))]

    @parameterized.expand(_UNIFORM_GEOMETRIES + _MIXED_GEOMETRIES)
    def test_generated_topologies_match_independent_relation_cpu(self, kernel_size, stride) -> None:
        coordinates = [(-4, -1, 0), (-1, 0, 1), (0, 2, -3), (3, -2, 4), (5, 1, -1)]
        relation = ConvolutionRelation(kernel_size, stride)
        fine = _grid(coordinates)

        coarse = fine.conv_grid(kernel_size=kernel_size, stride=stride)
        assert _coordinate_set(coarse) == forward_support(coordinates, relation)

        generated_fine = coarse.conv_transpose_grid(kernel_size=kernel_size, stride=stride)
        assert _coordinate_set(generated_fine) == transpose_support(forward_support(coordinates, relation), relation)

        participating = {edge.fine for edge in relation_edges(coordinates, relation)}
        assert participating.issubset(_coordinate_set(generated_fine))

    @parameterized.expand(_UNIFORM_GEOMETRIES + _MIXED_GEOMETRIES)
    def test_generated_forward_all_one_values_equal_independent_degrees_cpu(self, kernel_size, stride) -> None:
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

    @parameterized.expand(
        [
            (device, transposed, kernel_size, stride)
            for device in ("cpu", "cuda")
            for transposed in (False, True)
            for kernel_size, stride in (
                ((4, 2, 2), (4, 2, 2)),
                ((2, 3, 4), (1, 2, 3)),
                ((2, 1, 3), (3, 2, 4)),
            )
        ]
    )
    def test_generated_values_and_gradients_match_dense_oracle(self, device, transposed, kernel_size, stride) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        relation = ConvolutionRelation(kernel_size, stride)
        coordinates = sorted(
            {relation.fine_from_coarse(coarse, tap) for coarse in ((0, 0, 0), (-2, 1, -1)) for tap in relation.taps()}
        )
        assert {edge.tap for edge in relation_edges(coordinates, relation)} == set(relation.taps())
        source = _grid(coordinates, device=device)
        source_coordinates = [tuple(coordinate) for coordinate in source.ijk.jdata.cpu().tolist()]
        factory = ConvolutionPlan.from_grid_batch_transposed if transposed else ConvolutionPlan.from_grid_batch
        plan = factory(
            kernel_size=kernel_size,
            stride=stride,
            source_grid=source,
            acknowledge_incomplete_coverage=True,
        )
        target_coordinates = [tuple(coordinate) for coordinate in plan.target_grid_batch.ijk.jdata.cpu().tolist()]

        generator = torch.Generator().manual_seed(668 + int(transposed))
        base_features = torch.randn((source.total_voxels, 2), generator=generator, dtype=torch.float64).to(device)
        weight_count = 3 * 2 * kernel_size[0] * kernel_size[1] * kernel_size[2]
        base_weights = (
            torch.arange(1, weight_count + 1, dtype=torch.float64)
            .reshape(3, 2, *kernel_size)
            .div(weight_count)
            .to(device)
        )
        production_features = base_features.detach().clone().requires_grad_()
        production_weights = base_weights.detach().clone().requires_grad_()
        oracle_features = base_features.detach().clone().requires_grad_()
        oracle_weights = base_weights.detach().clone().requires_grad_()

        production_values = plan.execute(production_features, production_weights)
        oracle = (
            dense_transpose_oracle(source_coordinates, oracle_features, oracle_weights, relation)
            if transposed
            else dense_forward_oracle(source_coordinates, oracle_features, oracle_weights, relation)
        )
        oracle_values = _dense_rows(oracle, target_coordinates)
        torch.testing.assert_close(production_values, oracle_values, rtol=1.0e-11, atol=1.0e-11)

        probe = torch.arange(1, production_values.numel() + 1, dtype=torch.float64, device=device).reshape_as(
            production_values
        )
        production_gradients = torch.autograd.grad(
            torch.sum(production_values * probe), (production_features, production_weights)
        )
        oracle_gradients = torch.autograd.grad(torch.sum(oracle_values * probe), (oracle_features, oracle_weights))
        torch.testing.assert_close(production_gradients[0], oracle_gradients[0], rtol=1.0e-11, atol=1.0e-11)
        torch.testing.assert_close(production_gradients[1], oracle_gradients[1], rtol=1.0e-11, atol=1.0e-11)

    @parameterized.expand(
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
    def test_generated_topology_and_all_one_degrees_cuda(self, kernel_size, stride) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
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
        torch.testing.assert_close(
            plan.target_grid_batch.voxel_sizes.cpu(), torch.tensor([stride], dtype=torch.float32)
        )
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

    def test_generated_cuda_topology_matches_independent_relation_for_multi_grid_batch(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        relation = ConvolutionRelation((1, 1, 1), (2, 3, 4))
        coordinate_batches = [
            [(-4, -3, 0), (0, 0, 0), (2, 3, 4)],
            [(-1, 0, 0), (0, 1, 0), (0, 0, 2)],
            [(-2, -3, -4), (1, 3, 4), (4, 6, 8)],
        ]
        fine = GridBatch.from_ijk(
            JaggedTensor(
                [torch.tensor(coordinates, dtype=torch.int32, device="cuda") for coordinates in coordinate_batches]
            )
        )

        coarse = fine.conv_grid(kernel_size=relation.kernel_size, stride=relation.stride)
        expected_coarse_batches = [forward_support(coordinates, relation) for coordinates in coordinate_batches]
        assert expected_coarse_batches[1] == set()
        for batch_index, expected in enumerate(expected_coarse_batches):
            assert {tuple(coordinate) for coordinate in coarse.ijk[batch_index].jdata.cpu().tolist()} == expected

        generated_fine = coarse.conv_transpose_grid(kernel_size=relation.kernel_size, stride=relation.stride)
        for batch_index, coarse_coordinates in enumerate(expected_coarse_batches):
            expected = transpose_support(coarse_coordinates, relation)
            assert {
                tuple(coordinate) for coordinate in generated_fine.ijk[batch_index].jdata.cpu().tolist()
            } == expected

    @parameterized.expand(
        [
            ((4, 4, 4), (1, 1, 1)),
            ((2, 2, 2), (2, 2, 2)),
            ((1, 2, 2), (1, 2, 2)),
            ((3, 3, 3), (2, 2, 2)),
        ],
    )
    def test_generated_cuda_nanovdb_paths_match_independent_relation_for_empty_batch_item(
        self, kernel_size, stride
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        relation = ConvolutionRelation(kernel_size, stride)
        coordinate_batches = [
            [(-5, -2, 0), (-1, 0, 1), (3, 4, -2)],
            [],
            [(-4, -3, -2), (0, 0, 0), (2, 5, 3)],
        ]
        fine = GridBatch.from_ijk(
            JaggedTensor(
                [
                    torch.tensor(coordinates, dtype=torch.int32, device="cuda").reshape(-1, 3)
                    for coordinates in coordinate_batches
                ]
            )
        )

        coarse = fine.conv_grid(kernel_size=kernel_size, stride=stride)
        expected_coarse_batches = [forward_support(coordinates, relation) for coordinates in coordinate_batches]
        for batch_index, expected in enumerate(expected_coarse_batches):
            assert {tuple(coordinate) for coordinate in coarse.ijk[batch_index].jdata.cpu().tolist()} == expected

        generated_fine = coarse.conv_transpose_grid(kernel_size=kernel_size, stride=stride)
        for batch_index, coarse_coordinates in enumerate(expected_coarse_batches):
            expected = transpose_support(coarse_coordinates, relation)
            assert {
                tuple(coordinate) for coordinate in generated_fine.ijk[batch_index].jdata.cpu().tolist()
            } == expected

    def test_k1_s1_generated_grid_preserves_public_and_data_identity(self) -> None:
        fine = _grid([(-1, 0, 0), (0, 0, 0), (1, 0, 0)])
        generated_forward = fine.conv_grid(kernel_size=1, stride=1)
        generated_transpose = fine.conv_transpose_grid(kernel_size=1, stride=1)
        assert generated_forward is fine
        assert generated_transpose is fine
        assert generated_forward.is_same(fine)

        single = Grid.from_ijk(torch.tensor([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.int32))
        assert single.conv_grid(kernel_size=1, stride=1) is single
        assert single.conv_transpose_grid(kernel_size=1, stride=1) is single

    def test_generated_transpose_k1_s1_uses_matmul_and_compact_weights(self) -> None:
        source = _grid([(-1, 0, 0), (0, 0, 0), (3, 0, 0)])
        plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=1,
            stride=1,
            source_grid=source,
            channel_pairs=((2, 3),),
        )
        assert plan.target_grid_batch is source
        assert isinstance(plan._backend, _MatmulBackend)

        features = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-4.0, 2.0]], dtype=torch.float64)
        weights = torch.tensor([[1.0, 2.0], [-3.0, 0.5], [4.0, -2.0]], dtype=torch.float64)
        torch.testing.assert_close(plan.execute(features, weights), features @ weights.transpose(0, 1))

    def test_k1_strided_forward_is_residue_sampling_not_floor_coarsening(self) -> None:
        fine = _grid([(-3, 0, 0), (-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        coarse = fine.conv_grid(kernel_size=(1, 1, 1), stride=(3, 1, 1))
        assert _coordinate_set(coarse) == {(-1, 0, 0), (0, 0, 0), (1, 0, 0)}

        uncovered = _grid([(1, 0, 0)])
        empty = uncovered.conv_grid(kernel_size=(1, 1, 1), stride=(2, 1, 1))
        assert empty.total_voxels == 0

    def test_topology_policy_is_symmetric_and_reports_exact_output_coverage(self) -> None:
        fine = _grid([(0, 0, 0)])
        generated = ConvolutionPlan.from_grid_batch(kernel_size=(3, 1, 1), stride=1, source_grid=fine)
        assert generated.topology_policy is ConvolutionTopologyPolicy.COMPLETE
        assert generated.topology_provenance is ConvolutionTopologyProvenance.GENERATED
        assert generated.coverage_report is not None
        assert generated.coverage_report.output_zero_count == 0

        explicit = _grid([(0, 0, 0), (5, 0, 0)])
        restricted = ConvolutionPlan.from_grid_batch(
            kernel_size=(3, 1, 1), stride=1, source_grid=fine, target_grid=explicit
        )
        assert restricted.topology_policy is ConvolutionTopologyPolicy.RESTRICTED
        assert restricted.topology_provenance is ConvolutionTopologyProvenance.EXPLICIT_TARGET
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
        assert transposed.topology_policy is ConvolutionTopologyPolicy.COMPLETE
        assert _coordinate_set(transposed.target_grid_batch) == {(0, 0, 0), (1, 0, 0)}
        assert transposed.coverage_report is not None
        assert transposed.coverage_report.output_zero_count == 0

    def test_coverage_report_is_lazy_and_shared_by_exact_transposes(self) -> None:
        coverage_calls = 0
        original_coverage_report = convolution_plan_module._coverage_report

        def counted_coverage_report(*args, **kwargs):
            nonlocal coverage_calls
            coverage_calls += 1
            return original_coverage_report(*args, **kwargs)

        with patch.object(convolution_plan_module, "_coverage_report", counted_coverage_report):
            source = _grid([(-4, 0, 0), (-1, 0, 0), (0, 0, 0), (3, 0, 0), (8, 0, 0)])
            plan = ConvolutionPlan.from_grid_batch(kernel_size=(4, 1, 1), stride=(2, 1, 1), source_grid=source)
            transposed = ConvolutionPlan.from_plan_transposed(plan)
            twice_transposed = ConvolutionPlan.from_plan_transposed(transposed)
            assert coverage_calls == 0

            transposed_report = transposed.coverage_report
            assert transposed_report is not None
            assert coverage_calls == 1
            report = plan.coverage_report
            assert report is not None
            assert coverage_calls == 1
            assert transposed_report.input_degree_histogram == report.output_degree_histogram
            assert transposed_report.output_degree_histogram == report.input_degree_histogram
            assert twice_transposed.coverage_report is report
            assert coverage_calls == 1

    def test_topology_policy_rejects_inconsistent_factory_arguments(self) -> None:
        source = _grid([(0, 0, 0)])
        target = _grid([(0, 0, 0)])
        with pytest.raises(ValueError, match="COMPLETE.*target_grid=None"):
            ConvolutionPlan.from_grid_batch(
                kernel_size=3,
                stride=1,
                source_grid=source,
                target_grid=target,
                topology_policy=ConvolutionTopologyPolicy.COMPLETE,
            )
        with pytest.raises(ValueError, match="RESTRICTED.*explicit target_grid"):
            ConvolutionPlan.from_grid_batch(
                kernel_size=3,
                stride=1,
                source_grid=source,
                topology_policy=ConvolutionTopologyPolicy.RESTRICTED,
            )

        with pytest.raises(TypeError, match="ConvolutionTopologyPolicy"):
            ConvolutionPlan.from_grid_batch(
                kernel_size=3,
                stride=1,
                source_grid=source,
                topology_policy="complete",  # type: ignore[arg-type]
            )

    def test_incomplete_residue_warning_is_proactive_and_acknowledgeable(self) -> None:
        fine = _grid([(0, 0, 0)])
        with pytest.warns(ConvolutionCoverageWarning, match="uncovered stride residues"):
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

    def test_issue_668_16_cubed_production_round_trip(self) -> None:
        coordinates = list(torch.cartesian_prod(*(torch.arange(16, dtype=torch.int32) for _ in range(3))).tolist())
        fine = _grid(coordinates)
        coarse = fine.conv_grid(kernel_size=4, stride=4)
        assert coarse.total_voxels == 5**3
        round_trip = coarse.conv_transpose_grid(kernel_size=4, stride=4)
        assert set(map(tuple, coordinates)).issubset(_coordinate_set(round_trip))

    def test_forward_builder_exposes_exact_staging_accounting(self) -> None:
        coordinates = [(-4, 0, 0), (-1, 0, 0), (0, 0, 0), (3, 0, 0), (4, 0, 0)]
        fine = _grid(coordinates)

        fine.conv_grid(kernel_size=3, stride=3)
        direct = _fvdb_cpp.last_conv_grid_resource_stats()
        assert direct.input_voxel_count == len(coordinates)
        assert direct.kernel_volume == 27
        assert direct.valid_emission_count == len(coordinates)
        assert direct.used_direct_projection is True
        assert direct.peak_requested_bytes == direct.emission_requested_bytes == 16 * len(coordinates)

        relation = ConvolutionRelation((3, 1, 1), (4, 1, 1))
        fine.conv_grid(kernel_size=relation.kernel_size, stride=relation.stride)
        staged = _fvdb_cpp.last_conv_grid_resource_stats()
        expected_emissions = len(relation_edges(coordinates, relation))
        assert staged.valid_emission_count == expected_emissions
        assert staged.valid_emission_count < staged.input_voxel_count * staged.kernel_volume
        assert staged.used_direct_projection is False
        assert staged.peak_requested_bytes == max(
            staged.count_requested_bytes + staged.prefix_requested_bytes,
            staged.prefix_requested_bytes + staged.emission_requested_bytes,
        )

    def test_forward_builder_reports_nanovdb_paths_without_coordinate_staging(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        coordinates = [(-4, 0, 0), (-1, 0, 0), (0, 0, 0), (3, 0, 0), (4, 0, 0)]
        fine = _grid(coordinates, device="cuda")

        fine.conv_grid(kernel_size=2, stride=2)
        coarsened = _fvdb_cpp.last_conv_grid_resource_stats()
        assert coarsened.used_direct_projection is True
        assert coarsened.valid_emission_count == len(coordinates)
        assert coarsened.peak_requested_bytes == coarsened.emission_requested_bytes == 0

        fine.conv_grid(kernel_size=4, stride=1)
        padded = _fvdb_cpp.last_conv_grid_resource_stats()
        assert padded.input_voxel_count == len(coordinates)
        assert padded.kernel_volume == 64
        assert padded.valid_emission_count == 64 * len(coordinates)
        assert padded.used_direct_projection is False
        assert padded.count_requested_bytes == padded.prefix_requested_bytes == 0
        assert padded.peak_requested_bytes == padded.emission_requested_bytes == 0

    def test_generated_transpose_reports_allocation_failure_with_staging_context(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = _grid([(0, 0, 0)], device="cuda")
        with pytest.raises(
            torch.OutOfMemoryError,
            match=r"Coordinate staging alone requires .* input voxels \* .* kernel taps",
        ):
            source.conv_transpose_grid(kernel_size=(1_000_000_000, 100_000_000, 1), stride=1)

    def test_generated_transpose_uses_inactive_split_cuda_cache(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        script = textwrap.dedent(
            """
            import gc

            import torch

            import fvdb

            MiB = 1024**2
            GiB = 1024**3
            if torch.cuda.memory.get_allocator_backend() != "native":
                print("requires the native CUDA caching allocator")
                raise SystemExit(77)

            source = fvdb.GridBatch.from_ijk(
                fvdb.JaggedTensor([torch.zeros((1, 3), dtype=torch.int32, device="cuda")])
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_stats()
            total_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            target_limit_bytes = baseline["reserved_bytes.all.current"] + 196 * MiB
            if target_limit_bytes >= total_bytes:
                print("device is too small for the isolated split-cache witness")
                raise SystemExit(77)

            memory_fraction = target_limit_bytes / total_bytes
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            whole_block = torch.empty(128 * MiB, dtype=torch.uint8, device="cuda")
            del whole_block
            gc.collect()
            torch.cuda.synchronize()
            live_block = torch.empty(64 * MiB, dtype=torch.uint8, device="cuda")

            stats = torch.cuda.memory_stats()
            reserved_bytes = stats["reserved_bytes.all.current"]
            active_bytes = stats["active_bytes.all.current"]
            inactive_split_bytes = stats["inactive_split_bytes.all.current"]
            driver_free_bytes, _ = torch.cuda.mem_get_info()
            allocator_limit_bytes = int(total_bytes * memory_fraction)
            reservation_allowance_bytes = max(allocator_limit_bytes - reserved_bytes, 0)
            new_reservation_bytes = min(driver_free_bytes, reservation_allowance_bytes)
            desired_headroom_bytes = min(GiB, max(64 * MiB, allocator_limit_bytes // 20))
            headroom_bytes = min(desired_headroom_bytes, allocator_limit_bytes // 2)
            old_cache_bytes = max(reserved_bytes - active_bytes - inactive_split_bytes, 0)
            old_available_bytes = min(total_bytes, old_cache_bytes + new_reservation_bytes)
            old_safe_bytes = max(old_available_bytes - headroom_bytes, 0)
            reusable_cache_bytes = max(reserved_bytes - active_bytes, 0)
            allocator_available_bytes = min(total_bytes, reusable_cache_bytes + new_reservation_bytes)
            allocator_safe_bytes = max(allocator_available_bytes - headroom_bytes, 0)

            kernel_size = 80
            staging_bytes = 16 * kernel_size**3
            if not (
                inactive_split_bytes >= staging_bytes
                and old_safe_bytes < staging_bytes <= allocator_safe_bytes
            ):
                print(
                    "allocator did not form the required witness: "
                    f"inactive_split={inactive_split_bytes}, old_safe={old_safe_bytes}, "
                    f"allocator_safe={allocator_safe_bytes}, request={staging_bytes}"
                )
                raise SystemExit(77)

            result = source.conv_transpose_grid(kernel_size=kernel_size, stride=2)
            torch.cuda.synchronize()
            assert result.total_voxels == kernel_size**3
            assert live_block.numel() == 64 * MiB
            print(
                "split cache reused: "
                f"inactive_split={inactive_split_bytes}, old_safe={old_safe_bytes}, request={staging_bytes}"
            )
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode == 77:
            pytest.skip(completed.stdout.strip())
        assert (
            completed.returncode == 0
        ), f"split-cache subprocess failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        assert "split cache reused" in completed.stdout

    def test_generated_forward_grid_uses_convolution_lattice_transform(self) -> None:
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

    def test_valid_explicit_forward_and_transpose_transforms_are_accepted(self) -> None:
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

    def test_incompatible_explicit_transform_fails_before_topology_build(self) -> None:
        fine = _grid([(0, 0, 0)], voxel_sizes=1.0)
        incompatible_coarse = _grid([(0, 0, 0)], voxel_sizes=1.0)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("topology construction ran before transform validation")

        with patch.object(ConvolutionPlan, "_build_backend", staticmethod(fail_if_called)):
            with pytest.raises(ValueError, match="voxel size"):
                ConvolutionPlan.from_grid_batch(
                    kernel_size=3,
                    stride=2,
                    source_grid=fine,
                    target_grid=incompatible_coarse,
                )

    def test_explicit_transform_rejects_batch_mismatch_and_coarsening_contract(self) -> None:
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

    def test_explicit_transform_rejects_device_mismatch(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        fine = _grid([(0, 0, 0)], voxel_sizes=1.0)
        coarse = _grid([(0, 0, 0)], voxel_sizes=2.0, device="cuda")
        with pytest.raises(ValueError, match="same device"):
            ConvolutionPlan.from_grid_batch(
                kernel_size=3,
                stride=2,
                source_grid=fine,
                target_grid=coarse,
            )

    @parameterized.expand([(False,), (True,)])
    def test_plan_transpose_is_constant_time_exact_array_reversal(self, starts_transposed) -> None:
        fine = _grid([(-3, 0, 0), (0, 0, 0), (2, 0, 0), (9, 0, 0)])
        coarse = _grid([(-2, 0, 0), (0, 0, 0), (4, 0, 0), (7, 0, 0)])
        factory = ConvolutionPlan.from_grid_batch_transposed if starts_transposed else ConvolutionPlan.from_grid_batch
        plan = factory(
            kernel_size=(4, 1, 1),
            stride=1,
            source_grid=coarse if starts_transposed else fine,
            target_grid=fine if starts_transposed else coarse,
            channel_pairs=((2, 3), (4, 5)),
        )
        topology = _gather_scatter_topology(plan)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("from_plan_transposed rebuilt topology")

        with patch.object(ConvolutionPlan, "_build_backend", staticmethod(fail_if_called)):
            reversed_plan = ConvolutionPlan.from_plan_transposed(plan)
        reversed_topology = _gather_scatter_topology(reversed_plan)

        assert torch.equal(reversed_topology.gather_indices, topology.scatter_indices)
        assert torch.equal(reversed_topology.scatter_indices, topology.gather_indices)
        assert torch.equal(reversed_topology.offsets, topology.offsets)
        assert reversed_topology.gather_indices.data_ptr() == topology.scatter_indices.data_ptr()
        assert reversed_topology.scatter_indices.data_ptr() == topology.gather_indices.data_ptr()
        assert reversed_topology.offsets.data_ptr() == topology.offsets.data_ptr()
        assert reversed_topology.feature_total_voxels == topology.output_total_voxels
        assert reversed_topology.output_total_voxels == topology.feature_total_voxels
        assert reversed_topology.total_pairs == topology.total_pairs
        assert reversed_topology.is_transposed is not topology.is_transposed
        assert reversed_plan.source_grid_batch is plan.target_grid_batch
        assert reversed_plan.target_grid_batch is plan.source_grid_batch
        assert reversed_plan.topology_policy is ConvolutionTopologyPolicy.RESTRICTED
        assert reversed_plan.topology_provenance is ConvolutionTopologyProvenance.EXACT_TRANSPOSE
        assert reversed_plan._channel_pairs == ((3, 2), (5, 4))

    @parameterized.expand([(False,), (True,)])
    def test_independent_explicit_builder_matches_reversed_edge_set(self, starts_transposed) -> None:
        fine = _grid([(-3, 0, 0), (0, 0, 0), (2, 0, 0), (9, 0, 0)])
        coarse = _grid([(-2, 0, 0), (0, 0, 0), (4, 0, 0), (7, 0, 0)])
        if starts_transposed:
            plan = ConvolutionPlan.from_grid_batch_transposed(
                kernel_size=(4, 1, 1), stride=1, source_grid=coarse, target_grid=fine
            )
            independent = ConvolutionPlan.from_grid_batch(
                kernel_size=(4, 1, 1), stride=1, source_grid=fine, target_grid=coarse
            )
        else:
            plan = ConvolutionPlan.from_grid_batch(
                kernel_size=(4, 1, 1), stride=1, source_grid=fine, target_grid=coarse
            )
            independent = ConvolutionPlan.from_grid_batch_transposed(
                kernel_size=(4, 1, 1), stride=1, source_grid=coarse, target_grid=fine
            )

        reversed_plan = ConvolutionPlan.from_plan_transposed(plan)
        assert _topology_edges(_gather_scatter_topology(reversed_plan)) == _topology_edges(
            _gather_scatter_topology(independent)
        )
        assert independent.topology_provenance is ConvolutionTopologyProvenance.EXPLICIT_TARGET

    def test_double_plan_transpose_recovers_exact_topology_and_domains(self) -> None:
        source = _grid([(-4, 0, 0), (-1, 0, 0), (0, 0, 0), (3, 0, 0), (8, 0, 0)])
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=(4, 1, 1), stride=(2, 1, 1), source_grid=source, channel_pairs=((2, 3),)
        )
        topology = _gather_scatter_topology(plan)

        transposed = ConvolutionPlan.from_plan_transposed(plan)
        twice_transposed = ConvolutionPlan.from_plan_transposed(transposed)
        twice_topology = _gather_scatter_topology(twice_transposed)

        assert plan.topology_policy is ConvolutionTopologyPolicy.COMPLETE
        assert plan.topology_provenance is ConvolutionTopologyProvenance.GENERATED
        assert transposed.topology_policy is ConvolutionTopologyPolicy.RESTRICTED
        assert transposed.topology_provenance is ConvolutionTopologyProvenance.EXACT_TRANSPOSE
        assert twice_transposed.topology_policy is ConvolutionTopologyPolicy.RESTRICTED
        assert twice_transposed.topology_provenance is ConvolutionTopologyProvenance.EXACT_TRANSPOSE
        assert twice_transposed.source_grid_batch is plan.source_grid_batch
        assert twice_transposed.target_grid_batch is plan.target_grid_batch
        assert twice_topology.gather_indices.data_ptr() == topology.gather_indices.data_ptr()
        assert twice_topology.scatter_indices.data_ptr() == topology.scatter_indices.data_ptr()
        assert twice_topology.offsets.data_ptr() == topology.offsets.data_ptr()
        assert twice_topology.is_transposed == topology.is_transposed
        assert twice_transposed.valid_usage(2, 3, (4, 1, 1), (2, 1, 1), transposed=False)

    def test_exact_transpose_preserves_zero_degree_rows_and_columns(self) -> None:
        fine = _grid([(0, 0, 0), (1, 0, 0), (10, 0, 0)])
        coarse = _grid([(0, 0, 0), (7, 0, 0), (8, 0, 0)])
        restricted = ConvolutionPlan.from_grid_batch(
            kernel_size=(3, 1, 1), stride=1, source_grid=fine, target_grid=coarse
        )
        assert restricted.coverage_report is not None
        assert restricted.coverage_report.input_zero_count == 1
        assert restricted.coverage_report.output_zero_count == 2

        reversed_plan = ConvolutionPlan.from_plan_transposed(restricted)
        assert reversed_plan.coverage_report is not None
        assert reversed_plan.coverage_report.input_zero_count == 2
        assert reversed_plan.coverage_report.output_zero_count == 1
        assert (
            reversed_plan.coverage_report.input_degree_histogram == restricted.coverage_report.output_degree_histogram
        )
        assert (
            reversed_plan.coverage_report.output_degree_histogram == restricted.coverage_report.input_degree_histogram
        )

        residue_source = _grid([(0, 0, 0), (1, 0, 0)])
        complete_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=1,
            stride=(2, 1, 1),
            source_grid=residue_source,
            acknowledge_incomplete_coverage=True,
        )
        assert complete_plan.coverage_report is not None
        assert complete_plan.coverage_report.input_zero_count == 1
        assert complete_plan.coverage_report.output_zero_count == 0
        reversed_complete = ConvolutionPlan.from_plan_transposed(complete_plan)
        assert reversed_complete.topology_policy is ConvolutionTopologyPolicy.RESTRICTED
        assert reversed_complete.coverage_report is not None
        assert reversed_complete.coverage_report.input_zero_count == 0
        assert reversed_complete.coverage_report.output_zero_count == 1

    def test_pred_gather_plan_transpose_uses_reversed_fallback_topology(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = _grid([(-2, 0, 0), (0, 0, 0), (3, 0, 0)], device="cuda")
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=1,
            source_grid=source,
            channel_pairs=((32, 64),),
            expert_config={"backend": "pred_gather_igemm"},
        )
        assert isinstance(plan._backend, _PredGatherIGemmBackend)

        reversed_plan = ConvolutionPlan.from_plan_transposed(plan)
        assert isinstance(reversed_plan._backend, _GatherScatterBackend)
        assert (
            reversed_plan._backend.topology.gather_indices.data_ptr()
            == plan._backend.gs_topology.scatter_indices.data_ptr()
        )
        assert (
            reversed_plan._backend.topology.scatter_indices.data_ptr()
            == plan._backend.gs_topology.gather_indices.data_ptr()
        )
        assert reversed_plan.valid_usage(64, 32, 3, 1, transposed=True)

    @parameterized.expand([(False,), (True,)])
    def test_weighted_plan_transpose_dot_product_and_both_backward_paths(self, starts_transposed) -> None:
        fine = _grid([(-3, 0, 0), (0, 0, 0), (2, 0, 0), (9, 0, 0)])
        coarse = _grid([(-2, 0, 0), (0, 0, 0), (4, 0, 0), (7, 0, 0)])
        factory = ConvolutionPlan.from_grid_batch_transposed if starts_transposed else ConvolutionPlan.from_grid_batch
        plan = factory(
            kernel_size=(4, 1, 1),
            stride=1,
            source_grid=coarse if starts_transposed else fine,
            target_grid=fine if starts_transposed else coarse,
            channel_pairs=((2, 3),),
        )
        reversed_plan = ConvolutionPlan.from_plan_transposed(plan)

        generator = torch.Generator().manual_seed(668 + int(starts_transposed))
        features = torch.randn((plan.source_grid_batch.total_voxels, 2), generator=generator, dtype=torch.float64)
        features.requires_grad_(True)
        weights = torch.randn((3, 2, 4, 1, 1), generator=generator, dtype=torch.float64)
        weights.requires_grad_(True)
        dual = torch.randn((plan.target_grid_batch.total_voxels, 3), generator=generator, dtype=torch.float64)

        output = plan.execute(features, weights)
        transpose_weights = weights.detach().transpose(0, 1).contiguous()
        transposed_output = reversed_plan.execute(dual, transpose_weights)
        lhs = torch.sum(output * dual)
        rhs = torch.sum(features * transposed_output)
        torch.testing.assert_close(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)

        grad_features, grad_weights = torch.autograd.grad(lhs, (features, weights))
        torch.testing.assert_close(grad_features, transposed_output, rtol=1.0e-12, atol=1.0e-12)

        dual_for_backward = dual.detach().clone().requires_grad_(True)
        transpose_weights_for_backward = transpose_weights.detach().clone().requires_grad_(True)
        transposed_for_backward = reversed_plan.execute(dual_for_backward, transpose_weights_for_backward)
        reverse_loss = torch.sum(transposed_for_backward * features.detach())
        grad_dual, grad_transpose_weights = torch.autograd.grad(
            reverse_loss, (dual_for_backward, transpose_weights_for_backward)
        )
        torch.testing.assert_close(grad_dual, output.detach(), rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(
            grad_transpose_weights,
            grad_weights.transpose(0, 1).contiguous(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_matmul_backend_requires_shared_grid_data(self) -> None:
        source = _grid([(0, 0, 0), (1, 0, 0)])
        distinct_equal_target = _grid([(0, 0, 0), (1, 0, 0)])
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=1,
            stride=1,
            source_grid=source,
            target_grid=distinct_equal_target,
        )
        assert isinstance(plan._backend, _GatherScatterBackend)
        assert plan.target_grid_batch is distinct_equal_target

        features = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float64)
        weight_matrix = torch.tensor([[1.0, 2.0], [-3.0, 0.5], [4.0, -2.0]], dtype=torch.float64)
        weights = weight_matrix[:, :, None, None, None]
        torch.testing.assert_close(plan.execute(features, weights), features @ weight_matrix.transpose(0, 1))

    def test_generated_k1_s1_preserves_identity_and_matmul_fast_path(self) -> None:
        source = _grid([(-1, 0, 0), (0, 0, 0), (3, 0, 0)])
        plan = ConvolutionPlan.from_grid_batch(kernel_size=1, stride=1, source_grid=source)

        assert plan.target_grid_batch is source
        assert plan.target_grid_batch.data.is_same(source.data)
        assert isinstance(plan._backend, _MatmulBackend)
        assert plan.topology_policy is ConvolutionTopologyPolicy.COMPLETE

    @parameterized.expand([("compact",), ("public",)])
    def test_matmul_accepts_2d_and_public_5d_weights_for_flat_and_jagged_data(self, weight_layout) -> None:
        source = _grid([(-1, 0, 0), (0, 0, 0), (3, 0, 0)])
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=1,
            stride=1,
            source_grid=source,
            channel_pairs=((2, 3),),
        )
        features = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-4.0, 2.0]], dtype=torch.float64)
        weight_matrix = torch.tensor([[1.0, 2.0], [-3.0, 0.5], [4.0, -2.0]], dtype=torch.float64)
        weights = weight_matrix if weight_layout == "compact" else weight_matrix[:, :, None, None, None]
        expected = features @ weight_matrix.transpose(0, 1)

        flat_output = plan.execute(features, weights)
        jagged_output = plan.execute(JaggedTensor(features), weights)
        assert isinstance(flat_output, torch.Tensor)
        assert isinstance(jagged_output, JaggedTensor)
        torch.testing.assert_close(flat_output, expected)
        torch.testing.assert_close(jagged_output.jdata, expected)

    @parameterized.expand([((3, 2, 1),), ((3, 2, 1, 1, 2),)])
    def test_matmul_rejects_noncanonical_weight_shapes(self, weight_shape) -> None:
        source = _grid([(0, 0, 0)])
        plan = ConvolutionPlan.from_grid_batch(kernel_size=1, stride=1, source_grid=source)
        with pytest.raises(ValueError, match=r"\[C_out, C_in\].*\[C_out, C_in, 1, 1, 1\]"):
            plan.execute(torch.ones((1, 2)), torch.ones(weight_shape))

    @parameterized.expand(
        [
            (factory_name, kernel_size)
            for factory_name in ("from_grid_batch", "from_grid_batch_transposed")
            for kernel_size in (1, 3)
        ]
    )
    def test_dense_backend_is_disabled_for_all_factories_and_identity_geometry(self, factory_name, kernel_size) -> None:
        source = _grid([(0, 0, 0)])
        factory = getattr(ConvolutionPlan, factory_name)
        with pytest.raises(ValueError, match="dense convolution backend is disabled"):
            factory(
                kernel_size=kernel_size,
                stride=1,
                source_grid=source,
                expert_config={"backend": "dense"},
            )

    def test_dense_backend_is_disabled_in_private_backend_selection(self) -> None:
        source = _grid([(0, 0, 0)])
        with pytest.raises(ValueError, match="dense convolution backend is disabled"):
            ConvolutionPlan._build_backend(
                source,
                source,
                torch.tensor([1, 1, 1], dtype=torch.int32),
                torch.tensor([1, 1, 1], dtype=torch.int32),
                (),
                {"backend": "dense"},
            )

    @parameterized.expand([(kernel_size, stride) for kernel_size in (3, 5, 7) for stride in (1, 2)])
    def test_pred_gather_admits_only_pinned_phase_safe_geometries(self, kernel_size, stride) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = _grid([(-1, 0, 0), (0, 0, 0)], device="cuda")
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=stride,
            source_grid=source,
            channel_pairs=((32, 64),),
            expert_config={"backend": "pred_gather_igemm"},
        )
        assert isinstance(plan._backend, _PredGatherIGemmBackend)
        assert plan._backend.kernel_size == kernel_size
        assert plan._backend.stride == stride

    @parameterized.expand(
        [
            (1, 1, ((32, 64),), "uniform kernel sizes 3, 5, 7"),
            (4, 1, ((32, 64),), "uniform kernel sizes 3, 5, 7"),
            (9, 1, ((32, 64),), "uniform kernel sizes 3, 5, 7"),
            ((3, 3, 5), 1, ((32, 64),), "uniform kernel sizes 3, 5, 7"),
            (3, 3, ((32, 64),), "uniform strides 1, 2"),
            (3, (1, 2, 1), ((32, 64),), "uniform strides 1, 2"),
            (3, 1, ((16, 64),), "channel counts divisible by 32"),
            (3, 1, ((32, 48),), "channel counts divisible by 32"),
        ],
    )
    def test_pred_gather_rejects_geometry_or_channels_outside_pinned_boundary(
        self, kernel_size, stride, channel_pairs, match
    ) -> None:
        source = _grid([(0, 0, 0)])
        with pytest.raises(ValueError, match=match):
            ConvolutionPlan.from_grid_batch(
                kernel_size=kernel_size,
                stride=stride,
                source_grid=source,
                channel_pairs=channel_pairs,
                expert_config={"backend": "pred_gather_igemm"},
            )

    @parameterized.expand([(16, 64), (32, 48)])
    def test_pred_gather_unknown_channel_plan_fails_closed_at_usage_and_execution(
        self, in_channels, out_channels
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = _grid([(0, 0, 0)], device="cuda")
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=1,
            source_grid=source,
            expert_config={"backend": "pred_gather_igemm"},
        )
        assert plan.valid_usage(32, 64, 3, 1, transposed=False)
        assert not plan.valid_usage(in_channels, out_channels, 3, 1, transposed=False)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("native PredGatherIGemm dispatch received unsupported channels")

        features = torch.ones((source.total_voxels, in_channels), dtype=torch.float32, device="cuda")
        weights = torch.ones((out_channels, in_channels, 3, 3, 3), dtype=torch.float32, device="cuda")
        with patch.object(_fvdb_cpp, "pred_gather_igemm_conv", fail_if_called):
            with pytest.raises(ValueError, match="channel counts divisible by 32"):
                plan.execute(features, weights)

    def test_pred_gather_rejects_transposed_factory_without_building_target(self) -> None:
        source = _grid([(0, 0, 0)])

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("target topology was built before backend admission")

        with patch.object(GridBatch, "conv_transpose_grid", fail_if_called):
            with pytest.raises(ValueError, match="does not support transposed convolution"):
                ConvolutionPlan.from_grid_batch_transposed(
                    kernel_size=3,
                    stride=1,
                    source_grid=source,
                    channel_pairs=((32, 64),),
                    expert_config={"backend": "pred_gather_igemm"},
                )

    def test_pred_gather_rejects_cpu_grid_without_building_target(self) -> None:
        source = _grid([(0, 0, 0)])

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("target topology was built before backend admission")

        with patch.object(GridBatch, "conv_grid", fail_if_called):
            with pytest.raises(ValueError, match="grids on CUDA"):
                ConvolutionPlan.from_grid_batch(
                    kernel_size=3,
                    stride=1,
                    source_grid=source,
                    channel_pairs=((32, 64),),
                    expert_config={"backend": "pred_gather_igemm"},
                )

    def test_pred_gather_rejects_multi_grid_batch_without_building_target(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = GridBatch.from_ijk(
            JaggedTensor(
                [
                    torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda"),
                    torch.tensor([[1, 0, 0]], dtype=torch.int32, device="cuda"),
                ]
            )
        )

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("target topology was built before backend admission")

        with patch.object(GridBatch, "conv_grid", fail_if_called):
            with pytest.raises(ValueError, match="only batch size 1"):
                ConvolutionPlan.from_grid_batch(
                    kernel_size=3,
                    stride=1,
                    source_grid=source,
                    channel_pairs=((32, 64),),
                    expert_config={"backend": "pred_gather_igemm"},
                )
