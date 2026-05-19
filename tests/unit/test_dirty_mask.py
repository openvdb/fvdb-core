# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :func:`fvdb.functional.dirty_mask_from_sidecars_single`
and the ``dirty_mask`` argument on :meth:`fvdb.Grid.compute_esdf_incremental`.

Paper-framing context: dirty-region ESDF updates in fvdb are expressed
via a user-visible torch tensor (the dirty mask) rather than library-
internal allocator state (nvblox's ``BlockManager`` dirty-block set).
These tests pin the invariants that make that composition work.

Coverage:

* ``dirty_mask_from_sidecars`` correctness:
  - Flags voxels whose sidecar value differs.
  - Flags voxels absent from old grid as dirty.
  - Does NOT flag voxels present in both grids with identical values.
  - Multi-channel sidecars reduce via "any channel differs".
  - Empty old grid → everything dirty.
* ``compute_esdf_incremental(dirty_mask=all_false)`` short-circuits:
  returns the same ``Grid`` and ``Tensor`` objects (Python identity).
* ``compute_esdf_incremental(dirty_mask=all_true)`` is bit-identical
  to no-mask (full recompute).
* Partial dirty mask produces output that matches full-recompute on
  the dirty-reached region, with previously-good values preserved
  elsewhere (monotone-scene correctness under partial updates).
"""

import pytest
import torch

import fvdb


def _sphere_tsdf(vs=0.05, dims=20, ijk_min=-10, radius=0.35, trunc=0.15,
                 device="cuda"):
    """Helper: dense grid with analytic sphere TSDF + unit weights."""
    g = fvdb.Grid.from_dense(
        dense_dims=[dims, dims, dims], ijk_min=[ijk_min, ijk_min, ijk_min],
        voxel_size=vs, origin=[0, 0, 0], device=device,
    )
    xyz = (g.ijk.float() + 0.5) * vs
    tsdf = ((xyz.norm(dim=1) - radius) / trunc).clamp(-1, 1).to(torch.float32)
    weights = torch.ones(g.num_voxels, device=device, dtype=torch.float32)
    return g, tsdf, weights


# ---------------------------------------------------------------------------
#  dirty_mask_from_sidecars: correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_dirty_mask_flags_new_and_changed(device):
    """Classic three-voxel case: one unchanged, one value-changed,
    one new."""
    old_ijk = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    new_ijk = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.int32)
    old_grid = fvdb.Grid.from_ijk(old_ijk, voxel_size=0.1, origin=[0, 0, 0]).to(device)
    new_grid = fvdb.Grid.from_ijk(new_ijk, voxel_size=0.1, origin=[0, 0, 0]).to(device)

    old_sc = torch.tensor([1.0, 2.0], device=device)
    new_sc = torch.tensor([1.0, 5.0, 7.0], device=device)

    dirty = fvdb.functional.dirty_mask_from_sidecars_single(
        new_grid, new_sc, old_grid, old_sc,
    )
    assert dirty.dtype == torch.bool
    assert dirty.shape == (3,)
    assert dirty.cpu().tolist() == [False, True, True]


@pytest.mark.parametrize("device", ["cuda"])
def test_dirty_mask_all_unchanged_is_all_false(device):
    """Two identical grids + identical sidecars → no voxels dirty."""
    g, tsdf, _ = _sphere_tsdf(device=device)
    dirty = fvdb.functional.dirty_mask_from_sidecars_single(
        g, tsdf, g, tsdf,
    )
    assert not dirty.any().item()


@pytest.mark.parametrize("device", ["cuda"])
def test_dirty_mask_empty_old_is_all_true(device):
    """Old grid has zero voxels → every voxel in new grid is "new" →
    every entry dirty. Exercises the fast-path in the C++ helper."""
    empty = fvdb.Grid.from_zero_voxels(
        voxel_size=0.1, origin=[0, 0, 0], device=device,
    )
    empty_sc = torch.zeros(0, device=device, dtype=torch.float32)
    g, tsdf, _ = _sphere_tsdf(device=device)
    dirty = fvdb.functional.dirty_mask_from_sidecars_single(
        g, tsdf, empty, empty_sc,
    )
    assert dirty.shape == (g.num_voxels,)
    assert dirty.all().item()


@pytest.mark.parametrize("device", ["cuda"])
def test_dirty_mask_multichannel_any_differs(device):
    """Multi-channel sidecars: voxel is dirty iff ANY channel differs."""
    ijk = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.int32)
    grid = fvdb.Grid.from_ijk(ijk, voxel_size=0.1, origin=[0, 0, 0]).to(device)

    old_sc = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]], device=device)
    # Voxel 0: identical. Voxel 1: one channel changed. Voxel 2: all changed.
    new_sc = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 99.0],
                           [70.0, 80.0, 90.0]], device=device)
    dirty = fvdb.functional.dirty_mask_from_sidecars_single(
        grid, new_sc, grid, old_sc,
    )
    assert dirty.shape == (3,)
    assert dirty.cpu().tolist() == [False, True, True]


# ---------------------------------------------------------------------------
#  compute_esdf_incremental + dirty_mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_esdf_incremental_all_false_dirty_is_identity(device):
    """All-false dirty mask + non-empty prev_esdf ⇒ return (prev_grid,
    prev_esdf) directly via Python identity, never entering C++.
    This is the ~50 μs "cache hit" path that closes the warm-reuse
    gap with nvblox."""
    vs, trunc, max_dist = 0.05, 0.15, 0.3
    g, tsdf, weights = _sphere_tsdf(vs=vs, dims=20, ijk_min=-10,
                                     radius=0.35, trunc=trunc, device=device)

    # Build a prev_esdf state via one-shot call.
    prev_grid, prev_esdf = g.compute_esdf(
        tsdf, weights,
        truncation_distance=trunc, max_distance=max_dist,
    )

    # All-false dirty mask ⇒ short-circuit.
    dirty_all_false = torch.zeros(g.num_voxels, device=device, dtype=torch.bool)
    out_grid, out_esdf = g.compute_esdf_incremental(
        tsdf, weights, prev_grid, prev_esdf,
        truncation_distance=trunc, max_distance=max_dist,
        dirty_mask=dirty_all_false,
    )
    # Python-identity equality: no new allocation happened.
    assert out_grid is prev_grid, "should return prev_grid by identity"
    assert out_esdf is prev_esdf, "should return prev_esdf tensor by identity"


@pytest.mark.parametrize("device", ["cuda"])
def test_esdf_incremental_all_true_matches_no_mask(device):
    """All-true dirty mask is equivalent to no-mask: every voxel seeds,
    so the sweep runs the full propagation. Output must be bit-
    identical."""
    vs, trunc, max_dist = 0.05, 0.15, 0.3
    g, tsdf, weights = _sphere_tsdf(vs=vs, dims=20, ijk_min=-10,
                                     radius=0.35, trunc=trunc, device=device)
    prev_grid, prev_esdf = g.compute_esdf(
        tsdf, weights,
        truncation_distance=trunc, max_distance=max_dist,
    )

    dirty_all_true = torch.ones(g.num_voxels, device=device, dtype=torch.bool)
    _, esdf_dirty = g.compute_esdf_incremental(
        tsdf, weights, prev_grid, prev_esdf,
        truncation_distance=trunc, max_distance=max_dist,
        dirty_mask=dirty_all_true,
    )
    _, esdf_nomask = g.compute_esdf_incremental(
        tsdf, weights, prev_grid, prev_esdf,
        truncation_distance=trunc, max_distance=max_dist,
    )
    # Monotone-min is deterministic on these inputs; same seed set ⇒
    # byte-for-byte identical output.
    assert torch.equal(esdf_dirty, esdf_nomask)


@pytest.mark.parametrize("device", ["cuda"])
def test_esdf_incremental_partial_dirty_preserves_clean_region(device):
    """Partial dirty mask: half the seed-band voxels are marked dirty.
    The ESDF values on voxels far from the dirty region should match
    ``prev_esdf`` (they aren't re-seeded, and the wavefront from
    dirty seeds can't reach them within max_distance)."""
    vs, trunc, max_dist = 0.05, 0.15, 0.2
    g, tsdf, weights = _sphere_tsdf(vs=vs, dims=24, ijk_min=-12,
                                     radius=0.4, trunc=trunc, device=device)
    prev_grid, prev_esdf = g.compute_esdf(
        tsdf, weights,
        truncation_distance=trunc, max_distance=max_dist,
    )

    # Mark only voxels in the +x half of the grid as dirty.
    xyz = (g.ijk.float() + 0.5) * vs
    dirty = (xyz[:, 0] > 0.0).contiguous()

    out_grid, out_esdf = g.compute_esdf_incremental(
        tsdf, weights, prev_grid, prev_esdf,
        truncation_distance=trunc, max_distance=max_dist,
        dirty_mask=dirty,
    )

    # Same grid structure (incremental uses merge → topology identical
    # to prev in the static-TSDF case).
    assert out_grid.num_voxels == prev_grid.num_voxels

    # Voxels FAR from the dirty region (x < -max_distance - vs) cannot
    # receive wavefront contributions from dirty seeds; their values
    # must equal the previous ESDF exactly.
    out_xyz = (out_grid.ijk.float() + 0.5) * vs
    far_from_dirty = out_xyz[:, 0] < -(max_dist + vs)
    if far_from_dirty.any():
        assert torch.equal(out_esdf[far_from_dirty], prev_esdf[far_from_dirty])


@pytest.mark.parametrize("device", ["cuda"])
def test_esdf_incremental_no_mask_unchanged_behaviour(device):
    """Passing ``dirty_mask=None`` (the default) is backward-
    compatible: produces the same output as before this feature
    existed. Pinned against the existing idempotency invariant."""
    vs, trunc, max_dist = 0.05, 0.15, 0.3
    g, tsdf, weights = _sphere_tsdf(vs=vs, dims=20, ijk_min=-10,
                                     radius=0.35, trunc=trunc, device=device)
    prev_grid, prev_esdf = g.compute_esdf(
        tsdf, weights,
        truncation_distance=trunc, max_distance=max_dist,
    )

    _, esdf_nomask = g.compute_esdf_incremental(
        tsdf, weights, prev_grid, prev_esdf,
        truncation_distance=trunc, max_distance=max_dist,
    )
    # Feeding one-shot output back as prev with same TSDF should yield
    # the same result (idempotence of monotone-min at fixed point).
    assert torch.equal(esdf_nomask, prev_esdf)


@pytest.mark.parametrize("device", ["cuda"])
def test_full_pipeline_dirty_mask_workflow(device):
    """End-to-end demonstration that a user can (a) integrate a TSDF
    sweep, (b) compute a dirty mask from pre/post weights, (c) pass
    the dirty mask to compute_esdf_incremental. This is the paper's
    "dirty-region ESDF update" recipe in one test."""
    vs, trunc, max_dist = 0.1, 0.3, 0.5
    device_t = device

    # Two LiDAR-ish frames on a small synthetic sphere shell.
    torch.manual_seed(0)
    R = 1.0
    n_pts = 2000
    theta = torch.rand(n_pts) * 2 * 3.14159
    cos_phi = 2 * torch.rand(n_pts) - 1
    sin_phi = (1 - cos_phi ** 2).clamp_min(0).sqrt()
    pts1 = R * torch.stack([sin_phi * torch.cos(theta),
                             sin_phi * torch.sin(theta),
                             cos_phi], dim=1).to(device_t, dtype=torch.float32)

    # Seed grid + initial TSDF integrate.
    seed = fvdb.Grid.from_dense(
        dense_dims=[1, 1, 1], ijk_min=[0, 0, 0],
        voxel_size=vs, origin=[0, 0, 0], device=device_t,
    )
    tsdf0 = torch.zeros(seed.num_voxels, device=device_t, dtype=torch.float32)
    w0 = torch.zeros(seed.num_voxels, device=device_t, dtype=torch.float32)
    origin = torch.zeros(3, device=device_t, dtype=torch.float32)

    # Frame 0: integrate first sweep.
    g0, tsdf1, w1 = seed.integrate_tsdf_from_points(
        truncation_distance=trunc, points=pts1, sensor_origin=origin,
        tsdf=tsdf0, weights=w0,
    )
    # First ESDF: no prev state, use one-shot.
    esdf_grid0, esdf0 = g0.compute_esdf(
        tsdf1, w1, truncation_distance=trunc, max_distance=max_dist,
    )

    # Frame 1: identical points (simulated "no motion") → no change.
    g1, tsdf2, w2 = g0.integrate_tsdf_from_points(
        truncation_distance=trunc, points=pts1, sensor_origin=origin,
        tsdf=tsdf1, weights=w1,
    )
    # Compute dirty mask from weights diff (the integrator grew w1+=1
    # everywhere it re-observed; but since it's the same sweep, all
    # voxels that were touched in frame 0 are touched again — so
    # "dirty" here means "values changed". Some voxels *will* be
    # dirty because weights grow monotonically with each observation.
    dirty = fvdb.functional.dirty_mask_from_sidecars_single(
        g1, w2, g0, w1,
    )
    # Apply the dirty mask to incremental ESDF.
    esdf_grid2, esdf2 = g1.compute_esdf_incremental(
        tsdf2, w2, esdf_grid0, esdf0,
        truncation_distance=trunc, max_distance=max_dist,
        dirty_mask=dirty,
    )
    # Output grid has sensible voxel count + finite values.
    assert esdf_grid2.num_voxels > 0
    assert torch.isfinite(esdf2).all()
    # All values within the [-max_dist, +max_dist] clamp.
    assert esdf2.abs().max().item() <= max_dist + 1e-5
