# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for canonical sparse-convolution semantics."""

from itertools import product
import unittest

import pytest
import torch
from parameterized import parameterized

from fvdb.utils.tests.convolution_utils import (
    compute_conv_grid_topology_ground_truth,
    compute_conv_transpose_topology_ground_truth,
)
from fvdb.utils.tests.convolution_semantics_oracle import (
    MAX_DENSE_ORACLE_SPATIAL_SITES,
    ConvolutionRelation,
    DenseOraclePreflightError,
    ceil_div,
    dense_forward_oracle,
    dense_transpose_oracle,
    floor_div,
    forward_degrees,
    forward_support,
    relation_edges,
    transpose_support,
)


class TestConvSemantics(unittest.TestCase):
    @parameterized.expand(product(range(1, 7), range(1, 6)))
    def test_scalar_relation_matrix_covers_signed_residues(self, kernel: int, stride: int) -> None:
        relation = ConvolutionRelation((kernel, 1, 1), (stride, 1, 1))
        fine = [(coordinate, 0, 0) for coordinate in (-stride, -1, 0, stride - 1, stride)]
        edges = relation_edges(fine, relation)

        assert len(edges) == len(set(edges))
        assert all(edge.fine == relation.fine_from_coarse(edge.coarse, edge.tap) for edge in edges)
        assert {edge.fine for edge in edges}.issubset(transpose_support(forward_support(fine, relation), relation))

    @parameterized.expand([(kernel,) for kernel in range(1, 7)])
    def test_componentwise_kernel_equals_stride_projection(self, kernel: int) -> None:
        relation = ConvolutionRelation((kernel, kernel, kernel), (kernel, kernel, kernel))
        for fine in product((-kernel, -1, 0, kernel - 1, kernel), repeat=3):
            edges = relation_edges([fine], relation)
            assert len(edges) == 1
            edge = edges[0]
            expected_coarse = tuple((fine[axis] + relation.p_before[axis]) // kernel for axis in range(3))
            assert edge.coarse == expected_coarse
            assert relation.fine_from_coarse(edge.coarse, edge.tap) == fine

    def test_signed_division_and_even_torch_phase(self) -> None:
        assert floor_div(-5, 4) == -2
        assert ceil_div(-5, 4) == -1
        assert floor_div(5, 4) == 1
        assert ceil_div(5, 4) == 2
        expected_offsets = {
            1: (0,),
            2: (0, 1),
            3: (-1, 0, 1),
            4: (-1, 0, 1, 2),
            5: (-2, -1, 0, 1, 2),
            6: (-2, -1, 0, 1, 2, 3),
        }
        for kernel, expected in expected_offsets.items():
            relation = ConvolutionRelation((kernel, 1, 1), (1, 1, 1))
            assert tuple(relation.offset((tap, 0, 0))[0] for tap in range(kernel)) == expected

    def test_issue_668_one_dimensional_endpoint_counts(self) -> None:
        relation = ConvolutionRelation((4, 1, 1), (4, 1, 1))
        fine = [(coordinate, 0, 0) for coordinate in range(16)]
        assert forward_degrees(fine, relation) == {
            (0, 0, 0): 3,
            (1, 0, 0): 4,
            (2, 0, 0): 4,
            (3, 0, 0): 4,
            (4, 0, 0): 1,
        }

    def test_issue_668_16_cubed_round_trip_structural_regression(self) -> None:
        relation = ConvolutionRelation((4, 4, 4), (4, 4, 4))
        fine = list(product(range(16), repeat=3))
        coarse = forward_support(fine, relation)
        assert len(coarse) == 5**3
        assert set(fine).issubset(transpose_support(coarse, relation))
        assert all(degree > 0 for degree in forward_degrees(fine, relation).values())

    @parameterized.expand(
        [((4, 1, 1), (4, 1, 1)), ((4, 3, 2), (3, 2, 1)), ((3, 3, 3), (4, 4, 4))],
    )
    def test_legacy_topology_helpers_follow_canonical_relation(self, kernel, stride) -> None:
        fine = [(-5, -1, 0), (-1, 0, 1), (0, 2, -3), (3, -2, 4), (8, 1, -1)]
        relation = ConvolutionRelation(kernel, stride)
        fine_tensor = torch.tensor(fine, dtype=torch.int32)

        helper_forward = compute_conv_grid_topology_ground_truth(
            fine_tensor, kernel, stride, torch.device("cpu"), torch.float64
        )
        assert {tuple(row) for row in helper_forward.tolist()} == forward_support(fine, relation)

        coarse = sorted(forward_support(fine, relation))
        helper_transpose = compute_conv_transpose_topology_ground_truth(
            torch.tensor(coarse, dtype=torch.int32), kernel, stride, torch.device("cpu")
        )
        assert {tuple(row) for row in helper_transpose.tolist()} == transpose_support(coarse, relation)

    def test_legacy_topology_helpers_preserve_empty_coordinate_shape(self) -> None:
        empty = torch.empty((0, 3), dtype=torch.int32)
        assert compute_conv_grid_topology_ground_truth(
            empty, (4, 3, 2), (3, 2, 1), torch.device("cpu"), torch.float64
        ).shape == (0, 3)
        assert compute_conv_transpose_topology_ground_truth(empty, (4, 3, 2), (3, 2, 1), torch.device("cpu")).shape == (
            0,
            3,
        )

    @parameterized.expand(
        [((1, 1, 1), (1, 1, 1)), ((2, 3, 4), (1, 2, 3)), ((5, 2, 3), (4, 2, 1)), ((3, 3, 3), (4, 4, 4))],
    )
    def test_scalar_edges_support_and_round_trip_for_mixed_signed_geometry(self, kernel, stride) -> None:
        relation = ConvolutionRelation(kernel, stride)
        fine = [(-4, -1, 0), (-1, 0, 1), (0, 2, -3), (3, -2, 4), (5, 1, -1)]
        edges = relation_edges(fine, relation)
        assert len(edges) == len(set(edges))
        assert all(edge.fine == relation.fine_from_coarse(edge.coarse, edge.tap) for edge in edges)
        support = forward_support(fine, relation)
        participating = {edge.fine for edge in edges}
        assert participating.issubset(transpose_support(support, relation))

    def test_dense_forward_counts_and_sparse_scalar_degrees_agree(self) -> None:
        relation = ConvolutionRelation((4, 2, 3), (3, 2, 1))
        fine = [(-2, 0, 1), (0, 1, -1), (3, -1, 2), (4, 2, 0)]
        features = torch.ones((len(fine), 1), dtype=torch.float64)
        weights = torch.ones((1, 1, 4, 2, 3), dtype=torch.float64)
        dense = dense_forward_oracle(fine, features, weights, relation)
        assert {
            coordinate for coordinate, degree in forward_degrees(fine, relation).items() if degree > 0
        } == forward_support(fine, relation)
        for coordinate, degree in forward_degrees(fine, relation).items():
            assert dense.value_at(coordinate).item() == degree

    def test_dense_forward_values_and_gradients_match_scalar_relation(self) -> None:
        relation = ConvolutionRelation((2, 3, 1), (2, 1, 1))
        fine = [(-2, 0, 0), (-1, 1, 0), (0, -1, 0), (2, 0, 0)]
        coarse = sorted(forward_support(fine, relation))
        fine_to_index = {coordinate: index for index, coordinate in enumerate(fine)}

        features = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0], [4.0, -0.25]], dtype=torch.float64, requires_grad=True
        )
        weights = torch.arange(1, 1 + 3 * 2 * 2 * 3, dtype=torch.float64).reshape(3, 2, 2, 3, 1)
        weights = (weights / 17.0).requires_grad_()

        dense = dense_forward_oracle(fine, features, weights, relation)
        dense_values = torch.stack([dense.value_at(coordinate) for coordinate in coarse])

        scalar_rows = []
        for coordinate in coarse:
            terms = []
            for tap in relation.taps():
                fine_coordinate = relation.fine_from_coarse(coordinate, tap)
                if fine_coordinate in fine_to_index:
                    terms.append(weights[(slice(None), slice(None), *tap)] @ features[fine_to_index[fine_coordinate]])
            scalar_rows.append(torch.stack(terms).sum(dim=0))
        scalar_values = torch.stack(scalar_rows)
        torch.testing.assert_close(dense_values, scalar_values, rtol=1e-12, atol=1e-12)

        probe = torch.arange(1, 1 + dense_values.numel(), dtype=torch.float64).reshape_as(dense_values)
        dense_gradients = torch.autograd.grad((dense_values * probe).sum(), (features, weights), retain_graph=True)
        scalar_gradients = torch.autograd.grad((scalar_values * probe).sum(), (features, weights))
        torch.testing.assert_close(dense_gradients[0], scalar_gradients[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dense_gradients[1], scalar_gradients[1], rtol=1e-12, atol=1e-12)

    def test_dense_transpose_support_and_asymmetric_channel_adapter(self) -> None:
        relation = ConvolutionRelation((2, 1, 1), (2, 1, 1))
        coarse = [(-1, 0, 0), (2, 0, 0)]
        features = torch.tensor([[5.0, 7.0], [11.0, 13.0]], dtype=torch.float64)
        weights = torch.zeros((2, 2, 2, 1, 1), dtype=torch.float64)
        weights[:, :, 0, 0, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weights[:, :, 1, 0, 0] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        dense = dense_transpose_oracle(coarse, features, weights, relation)
        assert dense.value_at((-2, 0, 0)).tolist() == [19.0, 43.0]
        assert set(transpose_support(coarse, relation)) == {
            coordinate
            for coordinate in product(range(-2, 6), range(0, 1), range(0, 1))
            if not torch.equal(dense.value_at(coordinate), torch.zeros(2, dtype=torch.float64))
        }

    def test_dense_oracles_satisfy_asymmetric_channel_adjoint_identity(self) -> None:
        relation = ConvolutionRelation((3, 2, 1), (2, 1, 1))
        fine = [(-2, 0, 0), (0, 1, 0), (3, -1, 0), (4, 0, 0)]
        features = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-1.0, 5.0], [2.0, 7.0]], dtype=torch.float64)
        weights = torch.arange(1, 1 + 3 * 2 * 3 * 2, dtype=torch.float64).reshape(3, 2, 3, 2, 1)
        forward = dense_forward_oracle(fine, features, weights, relation)
        coarse = [
            tuple(forward.origin[axis] + local[axis] for axis in range(3))
            for local in product(*(range(forward.values.shape[axis + 2]) for axis in range(3)))
        ]
        coarse_features = forward.values[0].permute(1, 2, 3, 0).reshape(-1, weights.shape[0])
        cotangent = torch.arange(1, coarse_features.numel() + 1, dtype=torch.float64).reshape_as(coarse_features)
        transpose = dense_transpose_oracle(coarse, cotangent, weights.transpose(0, 1).contiguous(), relation)
        lhs = torch.sum(coarse_features * cotangent)
        rhs = torch.sum(torch.stack([transpose.value_at(coordinate) for coordinate in fine]) * features)
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=1e-10)

    def test_explicit_restriction_and_zero_feature_structure(self) -> None:
        relation = ConvolutionRelation((3, 1, 1), (2, 1, 1))
        fine = [(-1, 0, 0), (0, 0, 0), (2, 0, 0)]
        support = forward_support(fine, relation)
        restricted = relation_edges(fine, relation, coarse_coordinates=[next(iter(support))])
        assert {edge.coarse for edge in restricted} == {next(iter(support))}
        assert forward_support(fine, relation) == forward_support(list(fine), relation)

    def test_dense_oracle_preflight_fails_before_allocation(self) -> None:
        relation = ConvolutionRelation((1, 1, 1), (1, 1, 1))
        far_coordinate = (MAX_DENSE_ORACLE_SPATIAL_SITES + 1, 0, 0)
        with pytest.raises(DenseOraclePreflightError, match="preflight"):
            dense_forward_oracle(
                [(0, 0, 0), far_coordinate],
                torch.ones((2, 1), dtype=torch.float64),
                torch.ones((1, 1, 1, 1, 1), dtype=torch.float64),
                relation,
            )
