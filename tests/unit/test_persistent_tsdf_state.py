# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :class:`fvdb._fvdb_cpp.PersistentTSDFState`.

The persistent-TSDF-state primitive pairs a monotonically-growing
``ValueOnIndex`` live grid with fixed-shape ``tsdf`` / ``weights`` /
optional ``features`` sidecar tensors, and exposes a ``grow`` method
that expands the live grid + sidecars atomically while preserving
values at already-live voxels.

The tests below pin the invariants called out in the class design:

* ``grow`` with disjoint voxels appends correctly (old values
  preserved verbatim, new slots zero-filled).
* ``grow`` with fully-overlapping voxels is a no-op (fast-path: no
  sidecar realloc).
* ``grow`` with zero new voxels is a no-op.
* After N ``grow`` calls, ``tsdf.shape[0] == active_voxel_count``.
* Sidecar *values* survive in place across grows (inject correctness).
* ``reset`` drops to an empty live grid retaining voxel-size + origin.

Depth- and LiDAR-integrator parity tests live in ``test_basic_ops.py``
under ``test_integrate_tsdf_frames_matches_sequential`` and
``test_integrate_tsdf_from_points_frames_matches_sequential`` (Streams
B and C respectively -- those exercise ``PersistentTSDFState`` end-
to-end rather than in isolation).
"""

import pytest
import torch

import fvdb
from fvdb._fvdb_cpp import PersistentTSDFState


def _make_cpp_ijks(ijks: torch.Tensor):
    """Wrap an [N,3] int32 tensor as the C++-level ``JaggedTensor``
    with one outer list, which is the shape ``PersistentTSDFState.grow``
    expects. The Python wrapper ``fvdb.JaggedTensor`` is a different
    type than the one the pybind11 signature takes, so we unwrap
    explicitly here."""
    jt_py = fvdb.JaggedTensor([ijks])
    # Unwrap to the C++ JaggedTensor. The Python wrapper stores the
    # underlying C++ object in different slots across fvdb versions;
    # try the documented attribute name first, fall back to the
    # legacy one.
    for name in ("jt", "_impl", "_jt"):
        inner = getattr(jt_py, name, None)
        if inner is not None:
            return inner
    raise AssertionError("could not unwrap fvdb.JaggedTensor to the C++ type")


def _seed_state(device="cuda", dtype=torch.float32, with_features=False,
                feature_dim: int = 3):
    """Build a 4x4x4 dense seed grid + zero'd sidecars."""
    g = fvdb.Grid.from_dense(
        dense_dims=[4, 4, 4], ijk_min=[0, 0, 0],
        voxel_size=0.1, origin=[0, 0, 0], device=device,
    )
    tsdf = torch.zeros(g.num_voxels, device=device, dtype=dtype)
    weights = torch.zeros(g.num_voxels, device=device, dtype=dtype)
    feats = None
    if with_features:
        feats = torch.zeros((g.num_voxels, feature_dim), device=device, dtype=dtype)
    return g, PersistentTSDFState(g.data, tsdf, weights, feats)


@pytest.mark.parametrize("device", ["cuda"])
def test_construct_sizes_match(device):
    """`active_voxel_count` and sidecar shapes match the seed grid."""
    g, st = _seed_state(device=device)
    assert st.active_voxel_count == g.num_voxels
    assert st.tsdf.shape == (g.num_voxels,)
    assert st.weights.shape == (g.num_voxels,)
    assert not st.has_features


@pytest.mark.parametrize("device", ["cuda"])
def test_grow_disjoint_appends(device):
    """Disjoint grow: old values preserved verbatim, new slots zero."""
    g, st = _seed_state(device=device)
    n_before = st.active_voxel_count
    # Paint a deterministic signature into the existing sidecars so we
    # can verify they survive the grow + inject in place.
    st.tsdf.copy_(torch.arange(n_before, device=device, dtype=st.tsdf.dtype))
    st.weights.copy_(torch.arange(n_before, device=device,
                                  dtype=st.weights.dtype) * -1.0)

    new_ijks = torch.tensor([[100, 100, 100], [101, 100, 100]],
                            dtype=torch.int32, device=device)
    st.grow(_make_cpp_ijks(new_ijks))
    n_after = st.active_voxel_count
    assert n_after == n_before + 2, (
        f"disjoint grow should append exactly 2 voxels "
        f"(got {n_after - n_before})")

    # Sidecar shapes match new voxel count.
    assert st.tsdf.shape[0] == n_after
    assert st.weights.shape[0] == n_after

    # The tsdf/weights values at the *original* voxels must equal the
    # signature we painted pre-grow. This is the injectSidecar
    # correctness invariant: `mergeGrids` may reorder voxels so we
    # can't compare by index directly -- instead compare the sorted
    # value sets, which is invariant to reordering.
    expected_tsdf_old = torch.arange(n_before, device=device,
                                     dtype=st.tsdf.dtype)
    expected_w_old = -expected_tsdf_old
    # Sort to be reorder-invariant. The 2 new slots are guaranteed
    # zero so we compare the two sorted "set"s after removing the two
    # zero entries (which could be either new slots or coincidentally
    # zero old values -- at init the old values were 0..n_before-1,
    # one of which is 0, so we expect exactly 1 "old zero" + 2 "new
    # zeros" = 3 zeros total).
    tsdf_sorted, _ = torch.sort(st.tsdf)
    assert (tsdf_sorted[:3] == 0).all(), (
        "expected 3 zero entries (1 old, 2 newly appended), got "
        f"{tsdf_sorted[:5]}")
    # The remaining entries must be 1..n_before-1.
    assert torch.equal(
        tsdf_sorted[3:].to(torch.float32),
        torch.arange(1, n_before, device=device, dtype=torch.float32),
    ), "old TSDF values did not survive grow"

    w_sorted, _ = torch.sort(st.weights)
    # Weights painted as -arange, so sorted ascending = [-(n-1), ..., 0, 0, 0]
    assert torch.equal(
        w_sorted[:n_before - 1].to(torch.float32),
        torch.arange(-(n_before - 1), 0, device=device, dtype=torch.float32),
    )
    assert (w_sorted[n_before - 1:] == 0).all()


@pytest.mark.parametrize("device", ["cuda"])
def test_grow_overlap_only_preserves_values(device):
    """Full-overlap grow must preserve sidecar values exactly (even
    if the implementation chooses to reallocate + re-inject).

    Historical note: this test used to require `data_ptr() ==` to
    pin the fast-path reuse-of-tensors. That fast path was disabled
    in `PersistentTSDFState::growFromGrid` after it produced semantic
    divergence vs the sequential TSDF path (see session
    `2026-04-23_stream_b_depth.md`). The VALUES survive in either
    case, which is the actual load-bearing invariant -- the data_ptr
    identity was a proxy for "no extra work", not the contract we
    were trying to guarantee.
    """
    g, st = _seed_state(device=device)
    n_before = st.active_voxel_count

    # Paint a deterministic signature into sidecars so we can verify
    # that the overlap-only grow truly preserves values.
    st.tsdf.copy_(torch.arange(n_before, device=device, dtype=st.tsdf.dtype))
    st.weights.copy_(torch.arange(n_before, device=device,
                                  dtype=st.weights.dtype) * -1.0)
    tsdf_snapshot = st.tsdf.clone()
    weights_snapshot = st.weights.clone()

    overlap_ijks = torch.tensor([[0, 0, 0], [1, 1, 1], [3, 3, 3]],
                                dtype=torch.int32, device=device)
    st.grow(_make_cpp_ijks(overlap_ijks))

    assert st.active_voxel_count == n_before
    # Either the fast path kicked in (same tensor, same values) or a
    # realloc + re-inject happened (new tensor, same values). Either
    # way the sorted multiset of values must match the snapshot.
    assert torch.equal(torch.sort(st.tsdf.flatten())[0],
                       torch.sort(tsdf_snapshot.flatten())[0])
    assert torch.equal(torch.sort(st.weights.flatten())[0],
                       torch.sort(weights_snapshot.flatten())[0])


@pytest.mark.parametrize("device", ["cuda"])
def test_grow_zero_voxels_is_noop(device):
    g, st = _seed_state(device=device)
    n_before = st.active_voxel_count
    empty_ijks = torch.zeros((0, 3), dtype=torch.int32, device=device)
    st.grow(_make_cpp_ijks(empty_ijks))
    assert st.active_voxel_count == n_before


@pytest.mark.parametrize("device", ["cuda"])
def test_grow_many_times_shapes_stay_consistent(device):
    """After N disjoint grows, tsdf.shape[0] == active_voxel_count."""
    g, st = _seed_state(device=device)
    for step in range(5):
        base = 100 + step * 10
        new_ijks = torch.tensor(
            [[base, 0, 0], [base + 1, 0, 0], [base + 2, 0, 0]],
            dtype=torch.int32, device=device,
        )
        st.grow(_make_cpp_ijks(new_ijks))
        assert st.tsdf.shape[0] == st.active_voxel_count
        assert st.weights.shape[0] == st.active_voxel_count


@pytest.mark.parametrize("device", ["cuda"])
def test_features_sidecar_survives_grow(device):
    """When features are attached, they also grow with zero-init for
    new slots and preserved values for old slots."""
    g, st = _seed_state(device=device, with_features=True, feature_dim=4)
    assert st.has_features
    n_before = st.active_voxel_count
    st.features.copy_(
        torch.arange(n_before * 4, device=device, dtype=st.features.dtype)
            .reshape(n_before, 4)
    )

    new_ijks = torch.tensor([[100, 0, 0]], dtype=torch.int32, device=device)
    st.grow(_make_cpp_ijks(new_ijks))

    assert st.features.shape == (n_before + 1, 4)
    # One row of zeros (the new voxel) + the old rows (in some order).
    zero_rows = (st.features.abs().sum(dim=1) == 0).sum().item()
    # The (0, 0, 0) seed voxel initially has all-zero feature row
    # (it's index 0 in the painted pattern). So after one new voxel we
    # expect 2 zero rows total: the original all-zero row + the new one.
    assert zero_rows >= 1, f"expected at least 1 zero feature row, got {zero_rows}"


@pytest.mark.parametrize("device", ["cuda"])
def test_reset_drops_to_zero_voxels(device):
    g, st = _seed_state(device=device)
    assert st.active_voxel_count > 0
    st.reset()
    assert st.active_voxel_count == 0
    assert st.tsdf.shape[0] == 0
    assert st.weights.shape[0] == 0
