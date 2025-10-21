# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import functools
import itertools
import math
import site
import tempfile
from pathlib import Path
from typing import Any, Sequence, overload

import git
import git.repo
import torch
from fvdb.types import (
    DeviceIdentifier,
    NumericMaxRank1,
    NumericMaxRank2,
    NumericScalar,
    ValueConstraint,
    resolve_device,
    to_GenericScalar,
    to_Vec3i,
    to_Vec3iBatch,
    to_Vec3iBatchBroadcastable,
)
from git.exc import InvalidGitRepositoryError
from parameterized import parameterized

from fvdb import JaggedTensor

from .grid_utils import (
    make_dense_grid_and_point_data,
    make_dense_grid_batch_and_jagged_point_data,
    make_grid_and_point_data,
    make_grid_batch_and_jagged_point_data,
)
from .gsplat_utils import (
    create_uniform_grid_points_at_depth,
    generate_center_frame_point_at_depth,
    generate_random_4x4_xform,
)

git_tag_for_data = "main"


def set_testing_git_tag(git_tag):
    global git_tag_for_data
    git_tag_for_data = git_tag


def _is_editable_install() -> bool:
    # check we're not in a site package
    module_path = Path(__file__).resolve()
    for site_path in site.getsitepackages():
        if str(module_path).startswith(site_path):
            return False
    # check if we're in the source directory
    module_dir = module_path.parent.parent.parent.parent
    return (module_dir / "setup.py").is_file()


def _get_local_repo_path(repo_name: str) -> Path:
    """Get the local path where a git repository should be cloned.

    Args:
        repo_name: The name of the repository (e.g., 'fvdb_example_data', 'fvdb_test_data')

    Returns:
        Path to the local repository directory
    """
    if _is_editable_install():
        external_dir = Path(__file__).resolve().parent.parent.parent.parent / "external"
        if not external_dir.exists():
            external_dir.mkdir()
        local_repo_path = external_dir
    else:
        local_repo_path = Path(tempfile.gettempdir())

    local_repo_path = local_repo_path / repo_name
    return local_repo_path


def _clone_git_repo(git_url: str, git_tag: str, repo_name: str) -> tuple[Path, git.repo.Repo]:
    """Generic function to clone and checkout a git repository.

    Args:
        git_url: URL of the git repository to clone
        git_tag: Git tag or commit hash to checkout
        repo_name: Name for the local repository directory

    Returns:
        Tuple of (repo_path, repo) where repo_path is the Path to the cloned repo
        and repo is the git.repo.Repo object
    """

    def is_git_repo(repo_path: str) -> bool:
        is_repo = False
        try:
            _ = git.repo.Repo(repo_path)
            is_repo = True
        except InvalidGitRepositoryError:
            is_repo = False

        return is_repo

    repo_path = _get_local_repo_path(repo_name)

    if repo_path.exists() and repo_path.is_dir():
        if is_git_repo(str(repo_path)):
            repo = git.repo.Repo(repo_path)
        else:
            raise ValueError(f"A path {repo_path} exists but is not a git repo")
    else:
        repo = git.repo.Repo.clone_from(git_url, repo_path)
    repo.remotes.origin.fetch(tags=True)
    repo.git.checkout(git_tag)

    return repo_path, repo


def _clone_fvdb_test_data() -> tuple[Path, git.repo.Repo]:
    """Clone the fvdb-test-data repository for unit tests."""
    global git_tag_for_data
    git_url = "https://github.com/voxel-foundation/fvdb-test-data.git"
    return _clone_git_repo(git_url, git_tag_for_data, "fvdb_test_data")


def _clone_fvdb_example_data() -> tuple[Path, git.repo.Repo]:
    """Clone the fvdb-example-data repository for examples and documentation."""
    git_tag = "613c3a4e220eb45b9ae0271dca4808ab484ee134"
    git_url = "https://github.com/voxel-foundation/fvdb-example-data.git"
    return _clone_git_repo(git_url, git_tag, "fvdb_example_data")


def get_fvdb_test_data_path() -> Path:
    repo_path, _ = _clone_fvdb_test_data()
    return repo_path / "unit_tests"


def get_fvdb_example_data_path() -> Path:
    """Get the path to the cloned fvdb-example-data repository."""
    repo_path, _ = _clone_fvdb_example_data()
    return repo_path


# Hack parameterized to use the function name and the expand parameters as the test name
expand_tests = functools.partial(
    parameterized.expand,
    name_func=lambda f, n, p: f'{f.__name__}_{parameterized.to_safe_name("_".join(str(x) for x in p.args))}',
)


def probabilistic_test(
    iterations,
    pass_percentage: float = 80,
    conditional_args: list[list] | None = None,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the condition argument is present and matches the condition value
            do_repeat = True
            if conditional_args is None:
                do_repeat = False
            else:
                for a, condition_values in enumerate(conditional_args):
                    if args[a + 1] in condition_values:
                        continue
                    else:
                        do_repeat = False
                        break
            if do_repeat:
                passed = 0
                for _ in range(iterations):
                    try:
                        func(*args, **kwargs)
                        passed += 1
                    except AssertionError:
                        pass
                pass_rate = (passed / iterations) * 100
                assert pass_rate >= pass_percentage, f"Test passed only {pass_rate:.2f}% of the time"
            else:
                # If condition is not met, just run the function once
                return func(*args, **kwargs)

        return wrapper

    return decorator


def dtype_to_atol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 1e-1
    if dtype == torch.float16:
        return 1e-1
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float64:
        return 1e-5
    raise TypeError("dtype must be a valid torch floating type")


def generate_chebyshev_spaced_ijk(
    num_candidates: int,
    volume_shape: NumericMaxRank1,
    min_separation: NumericMaxRank1,
    *,
    avoid_borders: bool = False,
    dtype: torch.dtype = torch.int32,
    device: DeviceIdentifier | None = None,
) -> torch.Tensor:
    """
    Generates a set of 3D integer coordinates ("voxels") that are well-separated.

    The function uses a greedy sequential sampling strategy. It generates a number
    of random candidate points and accepts a point only if its Chebyshev distance
    (L-infinity norm) to all previously accepted points is greater than or equal
    to `min_separation`.

    This is particularly useful for generating non-interfering test locations
    for operations with a cubic footprint, such as a standard 3D convolution,
    where `min_separation` would typically be the kernel size.

    If `avoid_borders` is True, the function will avoid generating points that
    are too close to the borders of the volume. It will use half the min_separation
    as the border size.

    Args:
        num_candidates (int): The number of random candidate points to generate
            and test. The final number of points returned will be less than or
            equal to this value.
        volume_shape (NumericMaxRank1): The (I, J, K) dimensions of the
            volume from which to sample points.
        min_separation (NumericMaxRank1): The minimum required separation between
            any two points, measured by Chebyshev distance.
        avoid_borders (bool): Whether to avoid generating points that are too close to the borders of the volume.
            If True, the function will use half the min_separation as the border size.
        dtype (torch.dtype): The data type of the coordinates.
        device (DeviceIdentifier): The device to generate the coordinates on.

    Returns:
        torch.Tensor: A list of accepted (i, j, k) coordinates.
    """
    device = resolve_device(device)
    volume_shape = to_Vec3i(volume_shape, value_constraint=ValueConstraint.POSITIVE)
    min_separation = to_Vec3i(min_separation, value_constraint=ValueConstraint.POSITIVE)

    num_candidates = int(num_candidates)
    I, J, K = volume_shape.tolist()

    if avoid_borders:
        border_size = min_separation // 2
        Bi, Bj, Bk = border_size.tolist()
    else:
        Bi, Bj, Bk = 0, 0, 0

    # Generate tensor of random coordinates within the volume
    candidates = torch.stack(
        [
            Bi + torch.randint(0, I - 2 * Bi, (num_candidates,), dtype=dtype, device="cpu"),
            Bj + torch.randint(0, J - 2 * Bj, (num_candidates,), dtype=dtype, device="cpu"),
            Bk + torch.randint(0, K - 2 * Bk, (num_candidates,), dtype=dtype, device="cpu"),
        ],
        dim=1,
    )

    kept_points = torch.empty((num_candidates, 3), dtype=dtype, device="cpu")
    kept_points[0] = candidates[0]
    num_kept = 1

    for point_idx in range(1, num_candidates):
        test_point = candidates[point_idx]

        # Check if the test point is far enough from all previously kept points
        if torch.all(torch.abs(test_point - kept_points[:num_kept]) >= min_separation):
            kept_points[num_kept] = test_point
            num_kept += 1

    return kept_points[:num_kept].contiguous().to(device)


def generate_chebyshev_spaced_ijk_batch(
    batch_size: int,
    num_candidates: int,
    volume_shapes: NumericMaxRank2,
    min_separations: NumericMaxRank2,
    *,
    avoid_borders: bool = False,
    dtype: torch.dtype = torch.int32,
    device: DeviceIdentifier | None = None,
) -> JaggedTensor:
    """
    Generates batches of well-separated 3D integer coordinates.

    This is the batch version of `generate_chebyshev_spaced_ijk`. It generates
    a separate set of Chebyshev-spaced points for each item in the batch, where
    each batch item can have its own volume shape and minimum separation
    requirements.

    Args:
        batch_size (int): The number of batches to generate.
        num_candidates (int): The number of random candidate points to generate
            and test for each batch item. The final number of points per batch
            will be less than or equal to this value.
        volume_shapes (NumericMaxRank2): The (I, J, K) dimensions for each
            batch item. Can be a single shape broadcasted to all batches or
            a different shape per batch.
        min_separations (NumericMaxRank2): The minimum required separation
            between points for each batch item, measured by Chebyshev distance.
            Can be a single value broadcasted to all batches or different per batch.
        avoid_borders (bool): Whether to avoid generating points that are too close to the borders of the volume.
            If True, the function will use half the min_separation as the border size.
        dtype (torch.dtype): The data type of the coordinates.
        device (DeviceIdentifier): The device to generate the coordinates on.

    Returns:
        JaggedTensor: A jagged tensor containing the accepted (i, j, k) coordinates
            for each batch item. Each batch may have a different number of points.
    """
    volume_shapes = to_Vec3iBatchBroadcastable(volume_shapes, value_constraint=ValueConstraint.POSITIVE)
    min_separations = to_Vec3iBatchBroadcastable(min_separations, value_constraint=ValueConstraint.POSITIVE)

    return JaggedTensor(
        [
            generate_chebyshev_spaced_ijk(
                num_candidates,
                volume_shapes[i],
                min_separations[i],
                avoid_borders=avoid_borders,
                dtype=dtype,
                device=device,
            )
            for i in range(batch_size)
        ]
    )


def generate_hermit_impulses_dense(
    num_candidates: int,
    volume_shape: NumericMaxRank1,
    kernel_size: NumericMaxRank1,
    impulse_value: NumericMaxRank1 = 1,
    dtype: torch.dtype = torch.float32,
    device: DeviceIdentifier | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dense volume with impulse values at well-separated locations.

    This function creates a dense 3D tensor filled with zeros except at
    Chebyshev-spaced locations where it places the specified impulse values.
    The locations are chosen to be separated by at least the kernel size,
    making this ideal for testing convolution operations where impulse
    responses should not interfere with each other.

    Args:
        num_candidates (int): The number of random candidate points to generate
            and test. The final number of impulses will be less than or equal
            to this value.
        volume_shape (NumericMaxRank1): The (I, J, K) dimensions of the
            dense volume to create.
        kernel_size (NumericMaxRank1): The minimum required separation between
            impulses, measured by Chebyshev distance. Typically set to the
            convolution kernel size.
        impulse_value (NumericMaxRank1): The value(s) to place at each impulse
            location. Can be a scalar or tensor to support multi-channel data.
            Defaults to 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - impulse_coords: The (i, j, k) coordinates where impulses were placed.
            - vals: The dense volume tensor with impulses at the specified locations.
    """
    device = resolve_device(device)
    volume_shape = to_Vec3i(volume_shape, value_constraint=ValueConstraint.POSITIVE)
    kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
    impulse_value = torch.tensor(impulse_value, device=device, dtype=dtype)

    dense_shape = volume_shape.tolist() + list(impulse_value.shape)

    vals = torch.zeros(dense_shape, device=device, dtype=dtype)
    impulse_coords = generate_chebyshev_spaced_ijk(
        num_candidates, volume_shape, kernel_size, avoid_borders=True, dtype=torch.long, device=device
    )

    assert isinstance(impulse_coords, torch.Tensor)
    assert impulse_coords.dtype == torch.long

    vals[impulse_coords[:, 0], impulse_coords[:, 1], impulse_coords[:, 2]] = impulse_value
    return impulse_coords, vals


def generate_hermit_impulses_dense_batch(
    batch_size: int,
    num_candidates: int,
    volume_shape: NumericMaxRank1,
    kernel_size: NumericMaxRank1,
    *,
    impulse_value: NumericMaxRank1 = 1,
    dtype: torch.dtype = torch.float32,
    device: DeviceIdentifier | None = None,
) -> tuple[JaggedTensor, torch.Tensor]:
    """
    Generates batched dense volumes with impulse values at well-separated locations.

    This is the batch version of `generate_hermit_impulses_dense`. It creates
    a batch of dense 3D volumes, each filled with zeros except at Chebyshev-spaced
    locations where it places the specified impulse values. All volumes in the batch
    share the same shape and kernel size, but each has independently generated
    impulse locations.

    Args:
        batch_size (int): The number of volumes to generate in the batch.
        num_candidates (int): The number of random candidate points to generate
            and test for each volume. The final number of impulses per volume
            will be less than or equal to this value.
        volume_shape (NumericMaxRank1): The (I, J, K) dimensions of each
            dense volume. This shape is applied to all volumes in the batch.
        kernel_size (NumericMaxRank1): The minimum required separation between
            impulses, measured by Chebyshev distance. Typically set to the
            convolution kernel size. Applied uniformly across the batch.
        impulse_value (NumericMaxRank1): The value(s) to place at each impulse
            location. Can be a scalar or tensor to support multi-channel data.
            Defaults to 1.
        dtype (torch.dtype): The data type of the coordinates.
        device (DeviceIdentifier): The device to generate the coordinates on.

    Returns:
        tuple[JaggedTensor, torch.Tensor]: A tuple containing:
            - impulse_coords_batch: A jagged tensor of (i, j, k) coordinates for
              each batch item, where impulses were placed. Each batch may have
              a different number of impulses.
            - vals_batch: A dense tensor of shape (batch_size, I, J, K, ...) with
              impulses at the specified locations.
    """
    device = resolve_device(device)
    volume_shape = to_Vec3i(volume_shape, value_constraint=ValueConstraint.POSITIVE)
    kernel_size = to_Vec3i(kernel_size, value_constraint=ValueConstraint.POSITIVE)
    impulse_value = torch.tensor(impulse_value, device=device, dtype=dtype)

    dense_shape = [batch_size] + volume_shape.tolist() + list(impulse_value.shape)

    vals_batch = torch.zeros(dense_shape, device=device, dtype=dtype)
    # Broadcast single volume_shape and kernel_size to batch by repeating for each batch item
    impulse_coords_batch = generate_chebyshev_spaced_ijk_batch(
        batch_size,
        num_candidates,
        [volume_shape.tolist()] * batch_size,
        [kernel_size.tolist()] * batch_size,
        avoid_borders=True,
        dtype=torch.long,
        device=device,
    )
    impulse_coords_ub = impulse_coords_batch.unbind()
    assert len(impulse_coords_ub) == batch_size
    for i in range(batch_size):
        impulse_coords = impulse_coords_ub[i]
        assert isinstance(impulse_coords, torch.Tensor)
        vals = vals_batch[i]
        vals[impulse_coords[:, 0], impulse_coords[:, 1], impulse_coords[:, 2]] = impulse_value

    return impulse_coords_batch, vals_batch


def _wavenumber_from_frequency(frequency):
    return frequency * math.tau


def fourier_anti_symmetric_kernel(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: DeviceIdentifier | None = None,
) -> torch.Tensor:
    r"""
    Construct a deterministic, float32-friendly n-D kernel whose values are intentionally
    non-palindromic per axis and axis-distinct, so that accidental flips or transpositions
    during convolution testing are very unlikely to go undetected.

    This kernel is designed for testing (not for learning): it encodes orientation and
    axis identity using a small bank of Fourier components per axis plus weak cross-axis
    interactions. The pattern is stable, differentiable, and cheap to generate on CPU or
    GPU.

    -------------------------------------------------------------------------------
    Problem the kernel addresses
    -------------------------------------------------------------------------------
    In convolution test harnesses it is common to miss bugs where:
      1) a spatial axis is reversed (correlation vs. convolution, or explicit flip),
      2) two axes are swapped (e.g. H/W, D/H, channel dims), or
      3) an off-by-one/stride/padding error partially mirrors or permutes the kernel.

    Symmetric or near-symmetric test kernels can hide such defects because the output
    is unchanged (or nearly unchanged) under those transformations. We want a kernel
    whose numerical pattern makes such mistakes obvious.

    -------------------------------------------------------------------------------
    High-level construction
    -------------------------------------------------------------------------------
    Let the kernel shape be shape = (n0, n1, ..., n_{R-1}) with rank R (1..5).
    We build:
      * Per-axis components: For each axis "a" we take three harmonics (fixed L=3)
        of sine and cosine with axis-unique base frequencies and nonzero phases.
        These produce orientation-sensitive patterns that are not palindromic.
      * Cross-axis fingerprints: For every axis pair (a,b) we add a tiny sinusoid whose
        argument mixes both coordinates. These terms ensure that swapping equal-length
        axes still changes the pattern.

    Critically, we sample coordinates at bin centers:
        g_a[i] = (i + 0.5) / n_a  for  i in {0, ..., n_a-1}.
    Center sampling avoids endpoint periodicity (0 and 1 coincide for integer
    frequencies), which otherwise makes length-2 axes ambiguous under flips.

    -------------------------------------------------------------------------------
    Exact formula
    -------------------------------------------------------------------------------
    Let g_a be the normalized grid for axis a (center sampled), and let
    p_a be an axis-unique small prime number (chosen from [2, 3, 5, 7, 11]).
    We fix the number of harmonics to L = 3 regardless of rank.

    The kernel value at index tuple i is:
        K(i) = sum_{a=0}^{R-1} sum_{k=1}^{3} [
                   (p_a^-k)      * sin( 2*pi*(k*p_a)*g_a(i_a) + phi1_a )
                 + (p_a^-(k+0.5))* cos( 2*pi*((k*p_a)+0.5)*g_a(i_a) + phi2_a )
               ]
               + sum_{0<=a<b<R} alpha_{a,b} * sin( 2*pi*( p_b*g_a(i_a) + p_a*g_b(i_b) ) )

    where:
      * phi1_a = pi / (2*(p_a + 1))
      * phi2_a = pi / (3*(p_a + 2))
      * alpha_{a,b} = 1e-2 * (1 + 0.05*(a + b))

    Notes:
      * Different primes per axis -> different base frequencies -> axis identity.
      * Nonzero phases and mixed sin/cos -> no palindromic symmetry along an axis.
      * Cross-terms couple axes -> swapping equal-sized axes changes values globally.

    -------------------------------------------------------------------------------
    Why L=3 is sufficient (and fixed)
    -------------------------------------------------------------------------------
    Increasing L adds more frequency content but does not materially improve the ability
    to detect flips or permutations after L≈3 for typical kernel sizes. The mixture
    already:
      1) breaks palindromes (sin vs cos, nonzero phases),
      2) labels axes (distinct primes),
      3) defeats equal-size axis swaps (cross-terms),
      4) remains numerically tame in float32 (bounded dynamic range).

    Fixing L=3 removes a hyperparameter while keeping tests robust and deterministic.

    -------------------------------------------------------------------------------
    Numeric and stability considerations
    -------------------------------------------------------------------------------
      * dtype=float32 by default to mirror common conv setups; works on float16/bfloat16
        too, but those dtypes can reduce distinctness in edge cases. For strongest
        separability keep float32.
      * The overall amplitude is O(1) due to the p_a^-k scaling; cross-terms are 1e-2
        by design. You should not run into overflow/underflow even on large kernels.
      * The function is fully differentiable (useful if it ends up in a computation
        graph during tests). It is deterministic and seedless.

    -------------------------------------------------------------------------------
    Limitations (inherent)
    -------------------------------------------------------------------------------
      * An axis of length 1 cannot reveal a flip (it is invariant by definition).
        The kernel still distinguishes axis identity via cross-terms with other axes.
      * With constant inputs, correlation vs. convolution both reduce to sum(kernel),
        so they can match; use non-constant inputs in those tests.

    -------------------------------------------------------------------------------
    Parameters
    -------------------------------------------------------------------------------
    shape : Sequence[int]
        Desired kernel shape. Rank must be 1 <= len(shape) <= 5. Each n_a >= 1.
    dtype : torch.dtype, default=torch.float32
        Use float32 for portability/performance; float64 also works (overkill here).
    device : torch.device or None
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape `shape`, dtype `dtype`, on `device`.

    Example
    -------
    >>> K = fourier_anti_symmetric_kernel((5, 7))      # 2D spatial kernel
    >>> W = K.unsqueeze(0).unsqueeze(0)                # conv2d weights (1x1)
    >>> K_flip = K.flip((-2, -1))                      # emulate convolution
    >>> # In tests, F.conv2d(x, W) should differ from F.conv2d(x, W.flip)
    """
    device = resolve_device(device)
    shape = tuple(int(s) for s in shape)
    if not (1 <= len(shape) <= 5):
        raise ValueError(f"rank must be in [1, 5], got {len(shape)}")
    if any(s < 1 for s in shape):
        raise ValueError(f"all dimensions must be >=1, got {shape}")

    rank = len(shape)

    # 1) Center-sampled normalized coordinates per axis: (i + 0.5)/n
    coords = [(torch.arange(n, dtype=dtype, device=device) + 0.5) / n for n in shape]
    grids = torch.meshgrid(*coords, indexing="ij")

    # 2) Axis-unique base frequencies via small primes. Repeat if rank > len(primes).
    primes = [2, 3, 5, 7, 11]

    # 3) Fixed number of harmonics per axis (see docstring for rationale).
    L = 3

    out = torch.zeros(shape, dtype=dtype, device=device)

    # Per-axis asymmetric components (distinct frequency, amplitude, and phase).
    for ax, g in enumerate(grids):
        p = primes[ax % len(primes)]
        phase1 = math.pi / (2 * (p + 1))
        phase2 = math.pi / (3 * (p + 2))
        for k in range(1, L + 1):
            out = out + (1.0 / (p**k)) * torch.sin(_wavenumber_from_frequency(k * p) * g + phase1)
            out = out + (1.0 / (p ** (k + 0.5))) * torch.cos(_wavenumber_from_frequency((k * p) + 0.5) * g + phase2)

    # Cross-axis fingerprints: tiny, axis-unique amplitude; foil equal-size axis swaps.
    for a in range(rank):
        for b in range(a + 1, rank):
            pa, pb = primes[a % len(primes)], primes[b % len(primes)]
            scale = 1e-2 * (1.0 + 0.05 * (a + b))  # very small, but axis-specific
            out = out + scale * torch.sin(_wavenumber_from_frequency(pb * grids[a] + pa * grids[b]))

    return out


def has_any_symmetry_witnessed(
    tensor: torch.Tensor, *, rtol: float = 1e-5, atol: float = 1e-6, ignore_length1_axes: bool = True
) -> tuple[bool, dict[str, Any]]:
    r"""
    Detect whether a tensor is invariant (within tolerance) under ANY non-trivial
    combination of axis permutation and axis flip (reversal).

    This is intended to validate "asymmetric test kernels" used for convolution
    testing. If this function returns True, then there exists at least one
    transformation consisting of:
        - a permutation of axes, possibly followed by
        - flipping (reversing) one or more axes,
    that leaves the tensor numerically unchanged (within allclose tolerance).
    Such a symmetry is often undesirable for test kernels because it can mask
    bugs like accidental flips (correlation vs. convolution) or axis swaps.

    Definition of non-trivial:
      - A transformation is considered non-trivial if it affects at least one
        axis whose length > 1. Pure reordering or flipping of length-1 axes is
        ignored when ignore_length1_axes=True (the default), since those actions
        have no observable effect on values.

    What is checked:
      - Exhaustive search over all axis permutations (R!), and for each, all flip
        masks (2^R) where R = tensor.ndim. For R <= 5 this is at most 120 * 32
        = 3840 comparisons. Each comparison uses torch.allclose(rtol, atol).
      - Flips are applied AFTER the permutation. The reported flip axes, when
        return_witness=True, are indices in the permuted tensor's axis order.

    Notes and scope:
      - This routine only checks permutation and reversal symmetries. It does not
        check circular shifts, scaling, sign changes, or other group actions.
      - For float dtypes, comparisons use torch.allclose with the given tolerances.
        For non-floating tensors, the data are compared in float32 to avoid type
        restrictions in allclose.
      - The function is deterministic and does not allocate large temporaries
        beyond the transformed views needed for comparison.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor of rank R (typically 1..5 for convolution kernels).
    rtol : float, default 1e-5
        Relative tolerance for torch.allclose.
    atol : float, default 1e-6
        Absolute tolerance for torch.allclose.
    ignore_length1_axes : bool, default True
        If True, transformations that only affect axes of length 1 are not
        considered symmetries for the purpose of this check.

    Returns
    -------
    bool
        True if a non-trivial symmetry was found; False otherwise.
    dict
        The witness dict records one
        example transformation that preserved the tensor.
        Describing the first symmetry detected:
          {
            "perm": tuple[int, ...],              # permutation applied
            "flip_after_permute": tuple[int, ...],# axes flipped after permute
            "original_shape": tuple[int, ...],
            "transformed_shape": tuple[int, ...]
          }
        The "perm" tuple is exactly what was passed to tensor.permute(perm).

    Implementation details
    ----------------------
    Complexity is O(N * R! * 2^R) comparisons, where N is the number of elements.
    With R <= 5 this is practical for testing. Early exit occurs on the first
    detected symmetry. Identity transformation is excluded from consideration.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    R = tensor.ndim
    if R == 0:
        # Scalar: by convention it is symmetric under any transformation.
        return (True, {"perm": (), "flip_after_permute": (), "original_shape": (), "transformed_shape": ()})

    if R > 5:
        # Safety guard: exhaustive enumeration grows quickly beyond rank 5.
        raise ValueError(f"Rank {R} too large for exhaustive symmetry check (max 5).")

    # Special case: single element tensors are symmetric
    if tensor.numel() == 1:
        return (
            True,
            {
                "perm": tuple(range(R)),
                "flip_after_permute": (),
                "original_shape": tuple(tensor.shape),
                "transformed_shape": tuple(tensor.shape),
            },
        )

    # Use a float view for robust allclose comparisons if needed.
    x = tensor if torch.is_floating_point(tensor) else tensor.to(torch.float32)

    shape = tuple(x.shape)
    axes = tuple(range(R))

    # Precompute which axes are "effective" (length > 1) for non-triviality.
    effective = tuple(s > 1 for s in shape)

    def is_nontrivial(perm: tuple[int, ...], flip_mask: int, permuted_shape: tuple[int, ...]) -> bool:
        """Return True if the transform touches any length>1 axis."""
        if ignore_length1_axes:
            # Non-trivial permutation if any length>1 axis moves.
            moved = any(effective[a] and perm[a] != a for a in axes)
            # Non-trivial flip if any flipped axis has length>1 after permute.
            flipped = any(((flip_mask >> j) & 1) and (permuted_shape[j] > 1) for j in range(R))
            return moved or flipped
        else:
            # Any change at all (perm != id or any flip bit) counts.
            if any(perm[a] != a for a in axes):
                return True
            if flip_mask & ((1 << R) - 1):
                return True
            return False

    # Enumerate all permutations and flip masks, with early exit.
    for perm in itertools.permutations(axes, R):
        permuted = x.permute(perm)
        perm_shape = tuple(permuted.shape)

        # Iterate all flip masks from 0..(2^R - 1).
        # Skip identity (mask==0) when perm is identity unless it is non-trivial per the rule.
        for mask in range(1 << R):
            # Optionally skip masks that only flip size-1 axes to reduce useless work.
            if ignore_length1_axes and mask != 0:
                only_len1 = True
                for j in range(R):
                    if ((mask >> j) & 1) and perm_shape[j] > 1:
                        only_len1 = False
                        break
                if only_len1:
                    continue

            # Exclude identity transformation unless it is deemed "trivial-violating"
            # (which it should not be). This prevents immediate True due to allclose(x, x).
            if perm == axes and mask == 0:
                continue

            if not is_nontrivial(perm, mask, perm_shape):
                continue

            # Additional check: skip transformations that only flip length-1 axes
            # when ignore_length1_axes=False, as these are considered trivial
            if not ignore_length1_axes:
                # Check if the flip only affects length-1 axes
                only_len1_flips = True
                for j in range(R):
                    if ((mask >> j) & 1) and perm_shape[j] > 1:
                        only_len1_flips = False
                        break
                if only_len1_flips and mask != 0:
                    continue

            # Build the transformed view.
            if mask == 0:
                y = permuted
                flip_axes: tuple[int, ...] = ()
            else:
                flip_axes = tuple(j for j in range(R) if (mask >> j) & 1)
                y = permuted.flip(flip_axes)

            # Only compare if shapes match after transformation
            if x.shape == y.shape and torch.allclose(x, y, rtol=rtol, atol=atol):
                return True, {
                    "perm": tuple(int(p) for p in perm),
                    "flip_after_permute": flip_axes,
                    "original_shape": shape,
                    "transformed_shape": perm_shape,
                }

            # Special case for ignore_length1_axes: check if tensors are equivalent
            # when length-1 axes are ignored, but only for simple cases like (1,3) <-> (3,1)
            if ignore_length1_axes and x.shape != y.shape:
                # Only handle the case where we have exactly one length-1 axis and one non-length-1 axis
                # and the permutation swaps them
                x_len1_count = sum(1 for s in x.shape if s == 1)
                y_len1_count = sum(1 for s in y.shape if s == 1)
                x_non_len1 = tuple(s for s in x.shape if s > 1)
                y_non_len1 = tuple(s for s in y.shape if s > 1)

                # Check if this is a simple case: one axis of length 1, one axis of length > 1
                if (
                    x_len1_count == 1
                    and y_len1_count == 1
                    and len(x_non_len1) == 1
                    and len(y_non_len1) == 1
                    and x_non_len1 == y_non_len1
                ):
                    # This is a case like (1,3) <-> (3,1), check if values match
                    if torch.allclose(x.squeeze(), y.squeeze(), rtol=rtol, atol=atol):
                        return True, {
                            "perm": tuple(int(p) for p in perm),
                            "flip_after_permute": flip_axes,
                            "original_shape": shape,
                            "transformed_shape": perm_shape,
                        }

    return False, {}


def has_any_symmetry(
    tensor: torch.Tensor, *, rtol: float = 1e-5, atol: float = 1e-6, ignore_length1_axes: bool = True
) -> bool:
    r"""
    Detect whether a tensor is invariant (within tolerance) under ANY non-trivial
    combination of axis permutation and axis flip (reversal).

    This is intended to validate "asymmetric test kernels" used for convolution
    testing. If this function returns True, then there exists at least one
    transformation consisting of:
        - a permutation of axes, possibly followed by
        - flipping (reversing) one or more axes,
    that leaves the tensor numerically unchanged (within allclose tolerance).
    Such a symmetry is often undesirable for test kernels because it can mask
    bugs like accidental flips (correlation vs. convolution) or axis swaps.

    Definition of non-trivial:
      - A transformation is considered non-trivial if it affects at least one
        axis whose length > 1. Pure reordering or flipping of length-1 axes is
        ignored when ignore_length1_axes=True (the default), since those actions
        have no observable effect on values.

    What is checked:
      - Exhaustive search over all axis permutations (R!), and for each, all flip
        masks (2^R) where R = tensor.ndim. For R <= 5 this is at most 120 * 32
        = 3840 comparisons. Each comparison uses torch.allclose(rtol, atol).
      - Flips are applied AFTER the permutation. The reported flip axes, when
        return_witness=True, are indices in the permuted tensor's axis order.

    Notes and scope:
      - This routine only checks permutation and reversal symmetries. It does not
        check circular shifts, scaling, sign changes, or other group actions.
      - For float dtypes, comparisons use torch.allclose with the given tolerances.
        For non-floating tensors, the data are compared in float32 to avoid type
        restrictions in allclose.
      - The function is deterministic and does not allocate large temporaries
        beyond the transformed views needed for comparison.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor of rank R (typically 1..5 for convolution kernels).
    rtol : float, default 1e-5
        Relative tolerance for torch.allclose.
    atol : float, default 1e-6
        Absolute tolerance for torch.allclose.
    ignore_length1_axes : bool, default True
        If True, transformations that only affect axes of length 1 are not
        considered symmetries for the purpose of this check.

    Returns
    -------
    bool
        True if a non-trivial symmetry was found; False otherwise.

    Implementation details
    ----------------------
    Complexity is O(N * R! * 2^R) comparisons, where N is the number of elements.
    With R <= 5 this is practical for testing. Early exit occurs on the first
    detected symmetry. Identity transformation is excluded from consideration.
    """
    return has_any_symmetry_witnessed(tensor, rtol=rtol, atol=atol, ignore_length1_axes=ignore_length1_axes)[0]


from .timer import ScopedTimer

__all__ = [
    "set_testing_git_tag",
    "get_fvdb_test_data_path",
    "get_fvdb_example_data_path",
    "make_dense_grid_and_point_data",
    "make_dense_grid_batch_and_jagged_point_data",
    "make_grid_batch_and_jagged_point_data",
    "make_grid_and_point_data",
    "generate_random_4x4_xform",
    "create_uniform_grid_points_at_depth",
    "generate_center_frame_point_at_depth",
    "dtype_to_atol",
    "expand_tests",
    "ScopedTimer",
    "generate_chebyshev_spaced_ijk_batch",
    "generate_chebyshev_spaced_ijk",
    "generate_hermit_impulses_dense",
    "generate_hermit_impulses_dense_batch",
    "fourier_anti_symmetric_kernel",
    "has_any_symmetry_witnessed",
    "has_any_symmetry",
]
