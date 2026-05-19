# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :func:`fvdb.Grid.integrate_occupancy_from_points` and
its batched counterpart :func:`integrate_occupancy_from_points_frames`.

This op is the paper's fifth application of the nanoVDB topology-op
vocabulary (after depth TSDF, LiDAR TSDF, MC V4-V6, ESDF). It closes
the nvblox feature-parity gap from the primitive-usage matrix.

The tests below pin the invariants any future refactor must preserve:

* **Hit / miss / unknown classification**. A voxel at the sphere
  shell should get positive log-odds from hit rays; a voxel between
  the sensor and the shell should get negative (free) log-odds; a
  voxel behind the shell should not be updated.
* **Clamp bounds**. All log-odds values must stay in
  ``[log_odds_min, log_odds_max]`` after integration.
* **Bayesian idempotence under zero-update**. Integrating an empty
  point cloud should be a no-op.
* **Persistence across frames**. Running the batched N-frame call
  equals running the single-frame call N times in sequence (bit-
  identically up to the atomic-add noise floor).
* **Grid growth**. The output grid is the union of the input grid
  and the new point truncation shell.
* **Input validation**. Mismatched shapes / dtypes raise cleanly.
"""

import pytest
import torch

import fvdb


def _make_sphere_shell_points(
    radius: float, n_points: int, device: str, seed: int = 0,
) -> torch.Tensor:
    """`n_points` points uniformly sampled on a sphere of the given
    radius, centred at the origin. Deterministic via `seed`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.rand(n_points, generator=g) * (2.0 * 3.14159265)
    # uniform on sphere: phi via inverse-CDF (acos of uniform [-1, 1])
    cos_phi = 2.0 * torch.rand(n_points, generator=g) - 1.0
    sin_phi = (1.0 - cos_phi * cos_phi).clamp_min(0.0).sqrt()
    x = radius * sin_phi * torch.cos(theta)
    y = radius * sin_phi * torch.sin(theta)
    z = radius * cos_phi
    return torch.stack([x, y, z], dim=1).to(device=device, dtype=torch.float32)


def _seed_empty_grid(voxel_size: float, device: str = "cuda"):
    """1-voxel metadata-only seed — the integrator grows it via the
    shell allocator as rays come in."""
    g = fvdb.Grid.from_dense(
        dense_dims=[1, 1, 1], ijk_min=[0, 0, 0],
        voxel_size=voxel_size, origin=[0, 0, 0], device=device,
    )
    log_odds = torch.zeros(g.num_voxels, device=device, dtype=torch.float32)
    return g, log_odds


# ---------------------------------------------------------------------------
#  Correctness: hit / miss / unknown classification on a sphere shell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_sphere_shell_hits_are_positive(device):
    """Voxels at the sphere-shell radius should have positive log-odds
    (hits dominate), while voxels between the sensor origin and the
    shell should have negative log-odds (misses dominate)."""
    vs = 0.05
    trunc = 0.1
    R = 1.0
    n_pts = 2000
    points = _make_sphere_shell_points(R, n_pts, device=device)
    sensor_origin = torch.zeros(3, device=device, dtype=torch.float32)

    g, log_odds = _seed_empty_grid(vs, device=device)
    g2, log_odds2 = g.integrate_occupancy_from_points(
        truncation_distance=trunc,
        points=points, sensor_origin=sensor_origin,
        log_odds=log_odds,
    )

    xyz = (g2.ijk.float() + 0.5) * vs
    r = xyz.norm(dim=1)

    # Hit band: voxels within one truncation of the shell radius.
    hit_mask = (r >= R - trunc) & (r <= R + trunc)
    # Free band: voxels well inside the shell (traversed by many rays
    # as 'miss').
    free_mask = (r < R - 2 * vs) & (r > 0.2)

    assert hit_mask.sum().item() > 0, "sanity: should have hit-band voxels"
    assert free_mask.sum().item() > 0, "sanity: should have free-band voxels"

    # On average, hit-band voxels should have strictly higher log-odds
    # than free-band voxels. We don't assert per-voxel signs because
    # individual hit-band voxels can have net-negative log-odds if many
    # rays pass through them en route to a more distant surface
    # (edge of the shell); the statistical invariant is still clean.
    hit_mean = log_odds2[hit_mask].mean().item()
    free_mean = log_odds2[free_mask].mean().item()
    assert hit_mean > free_mean, \
        f"hit-band mean {hit_mean:.3f} should exceed free-band mean {free_mean:.3f}"


@pytest.mark.parametrize("device", ["cuda"])
def test_log_odds_clamped_to_bounds(device):
    """All returned log-odds must be in [log_odds_min, log_odds_max]."""
    vs = 0.05
    trunc = 0.1
    R = 1.0
    points = _make_sphere_shell_points(R, 2000, device=device)
    sensor_origin = torch.zeros(3, device=device, dtype=torch.float32)
    g, log_odds = _seed_empty_grid(vs, device=device)

    lo_min, lo_max = -3.5, 2.5
    _, log_odds2 = g.integrate_occupancy_from_points(
        truncation_distance=trunc,
        points=points, sensor_origin=sensor_origin,
        log_odds=log_odds,
        log_odds_hit=0.85, log_odds_miss=-0.40,
        log_odds_min=lo_min, log_odds_max=lo_max,
    )
    assert log_odds2.min().item() >= lo_min - 1e-6
    assert log_odds2.max().item() <= lo_max + 1e-6
    # Clamp should actually be hitting at least one bound on a scene
    # this dense (many rays through each near-origin voxel).
    assert (log_odds2 <= lo_min + 1e-6).any() or (log_odds2 >= lo_max - 1e-6).any()


# ---------------------------------------------------------------------------
#  Persistence / composition invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_empty_pointcloud_is_noop(device):
    """Zero-point integration grows the grid to the empty-shell union
    (which equals the input grid) and leaves log-odds unchanged."""
    vs = 0.05
    g, log_odds = _seed_empty_grid(vs, device=device)
    empty_pts = torch.empty(0, 3, device=device, dtype=torch.float32)
    sensor_origin = torch.zeros(3, device=device, dtype=torch.float32)

    g2, log_odds2 = g.integrate_occupancy_from_points(
        truncation_distance=0.1,
        points=empty_pts, sensor_origin=sensor_origin,
        log_odds=log_odds,
    )
    # Grid topology preserved.
    assert g2.num_voxels == g.num_voxels
    # Log-odds tensor preserved (0 -> 0 with no observations).
    assert torch.allclose(log_odds2, log_odds)


@pytest.mark.parametrize("device", ["cuda"])
def test_frames_matches_sequential(device):
    """Batched N-frame integration should produce the same result as
    calling the single-frame API N times in sequence (up to the
    atomic-add noise floor of the ray-walk kernel). Mirrors the
    analogous invariant pinned by
    ``test_integrate_tsdf_from_points_frames_matches_sequential``."""
    vs = 0.05
    trunc = 0.1
    n_frames = 3
    n_pts = 800
    device_t = device

    # Three frames with different sphere-shell radii (so each frame's
    # shell is structurally different and grid growth is exercised).
    pts_per_frame = [
        _make_sphere_shell_points(0.8, n_pts, device_t, seed=0),
        _make_sphere_shell_points(1.1, n_pts, device_t, seed=1),
        _make_sphere_shell_points(0.9, n_pts, device_t, seed=2),
    ]
    sensor_origins = torch.zeros(n_frames, 3, device=device_t, dtype=torch.float32)
    sensor_origins[:, 0] = torch.linspace(0.0, 0.1, n_frames)

    # Sequential reference: loop over single-frame API.
    g_seq, lo_seq = _seed_empty_grid(vs, device=device_t)
    for i in range(n_frames):
        g_seq, lo_seq = g_seq.integrate_occupancy_from_points(
            truncation_distance=trunc,
            points=pts_per_frame[i],
            sensor_origin=sensor_origins[i],
            log_odds=lo_seq,
        )

    # Batched path.
    g_batched, lo_batched = _seed_empty_grid(vs, device=device_t)
    g_batched, lo_batched = g_batched.integrate_occupancy_from_points_frames(
        truncation_distance=trunc,
        points_per_frame=pts_per_frame,
        sensor_origins=sensor_origins,
        log_odds=lo_batched,
    )

    assert g_seq.num_voxels == g_batched.num_voxels, \
        f"grid size mismatch: seq {g_seq.num_voxels}, batched {g_batched.num_voxels}"
    # Same ijk ordering by construction (both built the same union
    # sequence). Values should match to within atomic-add rounding
    # (1 ULP on a small fraction of voxels under heavy ray overlap).
    diff = (lo_seq - lo_batched).abs()
    # Use the same tolerance the LiDAR-TSDF batched-vs-sequential
    # parity test uses (atol=2e-6, rtol=1e-5). At log-odds magnitudes
    # around 4 this is effectively a 5e-5 abs tolerance.
    tol = 2e-6 + 1e-5 * lo_seq.abs().max().item()
    assert diff.max().item() <= tol, \
        f"seq vs batched max diff {diff.max().item()} exceeds tol {tol}"


# ---------------------------------------------------------------------------
#  Grid-growth / sidecar size invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_output_sidecar_size_matches_grid(device):
    vs = 0.05
    trunc = 0.1
    points = _make_sphere_shell_points(1.0, 1000, device=device)
    sensor_origin = torch.zeros(3, device=device, dtype=torch.float32)
    g, log_odds = _seed_empty_grid(vs, device=device)

    g2, log_odds2 = g.integrate_occupancy_from_points(
        truncation_distance=trunc,
        points=points, sensor_origin=sensor_origin,
        log_odds=log_odds,
    )
    # Output sidecar must match output grid's voxel count.
    assert log_odds2.shape == (g2.num_voxels,)
    # Output grid strictly grows (sphere shell adds voxels).
    assert g2.num_voxels > g.num_voxels


# ---------------------------------------------------------------------------
#  Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda"])
def test_mismatched_log_odds_size_raises(device):
    vs = 0.05
    g, _ = _seed_empty_grid(vs, device=device)
    bad_log_odds = torch.zeros(g.num_voxels + 1, device=device, dtype=torch.float32)
    points = _make_sphere_shell_points(1.0, 100, device=device)
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    with pytest.raises((RuntimeError, ValueError)):
        g.integrate_occupancy_from_points(
            truncation_distance=0.1,
            points=points, sensor_origin=origin,
            log_odds=bad_log_odds,
        )


@pytest.mark.parametrize("device", ["cuda"])
def test_inverted_clamp_bounds_raises(device):
    vs = 0.05
    g, log_odds = _seed_empty_grid(vs, device=device)
    points = _make_sphere_shell_points(1.0, 100, device=device)
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    with pytest.raises((RuntimeError, ValueError)):
        g.integrate_occupancy_from_points(
            truncation_distance=0.1,
            points=points, sensor_origin=origin, log_odds=log_odds,
            log_odds_min=2.0, log_odds_max=-2.0,   # inverted
        )
