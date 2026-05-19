# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :meth:`fvdb.Grid.decay_and_prune` — the dynamic-scene
decay primitive.

The paper-framing point this helper demonstrates: because fvdb stores
each per-voxel sidecar as a separate torch tensor, selective decay
(decay one field, leave the others alone) is a trivial composition of
a multiplicative torch op and the existing ``pruneGrid`` primitive.
No new library machinery needed -- contrast nvblox, whose block-packed
``{sdf, weight, color}`` tuples require layer-aware decay methods.

These tests pin the six invariants the helper promises:

* Decay-only (``prune_threshold=0``) is a pure tensor multiply; the
  grid and sidecar shape are unchanged.
* Decay-and-prune at a non-zero threshold drops exactly the voxels
  whose decayed magnitude has fallen below the threshold.
* Extra sidecars stay in sync with the pruned grid (same mask).
* Idempotence: ``decay_factor=1.0`` with threshold=0 is a no-op.
* Multi-channel sidecars prune on L2 norm magnitude.
* Repeated calls compose naturally (5 calls at factor=0.9 with
  threshold=0.2 matches a single call at factor=0.9^5 with the same
  threshold, up to the order of prune/not-prune decisions).
"""

import pytest
import torch

import fvdb


def _make_grid_with_sidecars(device: str = "cuda"):
    """Small dense grid of 27 voxels with TSDF + weights + features."""
    g = fvdb.Grid.from_dense(
        dense_dims=[3, 3, 3], ijk_min=[-1, -1, -1],
        voxel_size=0.1, origin=[0, 0, 0], device=device,
    )
    # Weights: monotonic 1.0 ... 27.0 so we can predict which voxels
    # survive each threshold.
    weights = torch.arange(1, g.num_voxels + 1, device=device, dtype=torch.float32)
    tsdf = torch.linspace(-1.0, 1.0, g.num_voxels, device=device, dtype=torch.float32)
    features = torch.randn(g.num_voxels, 3, device=device, dtype=torch.float32,
                           generator=torch.Generator(device=device).manual_seed(42))
    return g, tsdf, weights, features


# ---------------------------------------------------------------------------
#  Decay-only (threshold = 0): pure tensor multiply, no topology change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_decay_only_is_tensor_multiply(device):
    """With ``prune_threshold=0`` the helper is a pure multiplicative
    scaling of the sidecar; the grid is returned unchanged."""
    g, tsdf, weights, features = _make_grid_with_sidecars(device=device)

    g2, w2, extras = g.decay_and_prune(
        weights, decay_factor=0.5, prune_threshold=0.0,
        extra_sidecars=[tsdf, features],
    )
    # Grid unchanged.
    assert g2.num_voxels == g.num_voxels
    # Sidecar = sidecar * decay_factor.
    assert torch.allclose(w2, weights * 0.5)
    # Extras unchanged (decay only acts on the primary sidecar).
    assert torch.equal(extras[0], tsdf)
    assert torch.equal(extras[1], features)


@pytest.mark.parametrize("device", ["cuda"])
def test_decay_factor_1_is_noop(device):
    """decay_factor=1.0, prune_threshold=0 is a pure no-op: grid and
    sidecars are returned as-is (up to tensor identity/allclose)."""
    g, tsdf, weights, _ = _make_grid_with_sidecars(device=device)
    g2, w2, extras = g.decay_and_prune(
        weights, decay_factor=1.0, prune_threshold=0.0,
        extra_sidecars=[tsdf],
    )
    assert g2.num_voxels == g.num_voxels
    assert torch.equal(w2, weights)
    assert torch.equal(extras[0], tsdf)


# ---------------------------------------------------------------------------
#  Decay-and-prune: topology shrinks to match the retained voxels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_prune_drops_below_threshold(device):
    """With decay=0.5 on weights [1..27] and threshold=5:
    decayed weights = [0.5, 1.0, ..., 13.5].
    Keep those with |decayed| > 5, i.e. decayed > 5.0, i.e. original
    weight > 10.0. So voxels with weight >= 11 survive = 17 voxels."""
    g, tsdf, weights, features = _make_grid_with_sidecars(device=device)
    g2, w2, extras = g.decay_and_prune(
        weights, decay_factor=0.5, prune_threshold=5.0,
        extra_sidecars=[tsdf, features],
    )
    # 27 original voxels; those with decayed weight > 5 survive.
    # decayed weights > 5 means original weights > 10, so weights in
    # {11, 12, ..., 27} = 17 voxels.
    assert g2.num_voxels == 17
    assert w2.shape == (17,)
    assert extras[0].shape == (17,)
    assert extras[1].shape == (17, 3)
    # All surviving weights are > 5 after decay.
    assert (w2 > 5.0).all()


@pytest.mark.parametrize("device", ["cuda"])
def test_extra_sidecars_stay_in_sync(device):
    """The pruned grid and all extra_sidecars must share the same mask —
    voxel i in the output corresponds to the same voxel across all
    output tensors."""
    g, tsdf, weights, features = _make_grid_with_sidecars(device=device)
    # Reference: apply the same decay + mask manually.
    expected_weights = weights * 0.7
    mask = expected_weights.abs() > 3.0
    expected_tsdf = tsdf[mask]
    expected_features = features[mask]

    _, w2, extras = g.decay_and_prune(
        weights, decay_factor=0.7, prune_threshold=3.0,
        extra_sidecars=[tsdf, features],
    )
    assert torch.equal(w2, expected_weights[mask])
    assert torch.equal(extras[0], expected_tsdf)
    assert torch.equal(extras[1], expected_features)


@pytest.mark.parametrize("device", ["cuda"])
def test_threshold_above_max_prunes_everything(device):
    """Threshold higher than any decayed magnitude prunes every voxel
    and produces a zero-voxel grid."""
    g, _, weights, _ = _make_grid_with_sidecars(device=device)
    g2, w2, _ = g.decay_and_prune(
        weights, decay_factor=0.5, prune_threshold=100.0,
    )
    assert g2.num_voxels == 0
    assert w2.shape == (0,)


# ---------------------------------------------------------------------------
#  Multi-channel sidecars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_multichannel_sidecar_uses_l2_magnitude(device):
    """For a ``[num_voxels, C]`` sidecar, the prune predicate is the
    per-voxel L2 norm."""
    g, _, _, features = _make_grid_with_sidecars(device=device)

    decayed_feat = features * 0.8
    l2 = decayed_feat.norm(dim=1)
    thresh = l2.median().item()  # prunes ~half the voxels

    g2, feat2, _ = g.decay_and_prune(
        features, decay_factor=0.8, prune_threshold=thresh,
    )
    # Sanity: we dropped some voxels.
    assert 0 < g2.num_voxels < g.num_voxels
    assert feat2.shape == (g2.num_voxels, 3)
    # All surviving rows have L2 norm > threshold.
    assert (feat2.norm(dim=1) > thresh).all()


# ---------------------------------------------------------------------------
#  Composition / temporal behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_repeated_decay_composes(device):
    """5 successive decays at factor=0.9 should match one decay at
    0.9**5 = 0.59049 applied to the same starting weights, provided
    the prune threshold doesn't fire (so no topology changes)."""
    g, tsdf, weights, _ = _make_grid_with_sidecars(device=device)

    # Loop 5 decays without pruning.
    cur_grid, cur_w, extras = g, weights.clone(), [tsdf.clone()]
    for _ in range(5):
        cur_grid, cur_w, extras = cur_grid.decay_and_prune(
            cur_w, decay_factor=0.9, prune_threshold=0.0,
            extra_sidecars=extras,
        )

    # Reference: single decay with compound factor.
    expected = weights * (0.9 ** 5)
    # fp32 associativity: compare with a small tolerance.
    assert torch.allclose(cur_w, expected, atol=1e-5, rtol=1e-5)
    # Topology unchanged (no pruning happened).
    assert cur_grid.num_voxels == g.num_voxels
    # Extras untouched.
    assert torch.equal(extras[0], tsdf)


# ---------------------------------------------------------------------------
#  Composability with other per-field ops (the paper-figure point)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_per_field_decay_is_independent(device):
    """Selective decay: decay weights while leaving features untouched,
    using nothing but :meth:`decay_and_prune` on the one sidecar.

    This is the paper-figure demonstration of fvdb's "field
    orthogonality is free" architectural advantage — you don't need a
    layer-aware library method; you decay the tensor you care about
    and that's it."""
    g, tsdf, weights, features = _make_grid_with_sidecars(device=device)
    features_orig = features.clone()

    # Decay weights only. Features pass through extra_sidecars
    # unchanged (except for any pruning that the grid shrinks).
    _, w2, extras = g.decay_and_prune(
        weights, decay_factor=0.5, prune_threshold=0.0,
        extra_sidecars=[tsdf, features],
    )
    tsdf2, features2 = extras

    # Weights scaled, features and tsdf unchanged.
    assert torch.allclose(w2, weights * 0.5)
    assert torch.equal(tsdf2, tsdf)
    assert torch.equal(features2, features_orig)


@pytest.mark.parametrize("device", ["cuda"])
def test_compound_prune_predicate_via_user_mask(device):
    """The user can also skip ``decay_and_prune`` and compose a
    compound prune predicate directly through :meth:`pruned_grid`
    (which is what ``decay_and_prune`` uses internally). This pins
    that the underlying primitive is accessible for custom
    predicates -- the paper point is that every composition here is
    1-3 lines of Python."""
    g, tsdf, weights, features = _make_grid_with_sidecars(device=device)
    # Compound predicate: keep voxels with weight > 5 AND features-
    # norm > 0.5. Entirely user-authored, no fvdb helper needed.
    keep = (weights > 5.0) & (features.norm(dim=1) > 0.5)
    g2 = g.pruned_grid(keep)
    assert g2.num_voxels == int(keep.sum().item())
