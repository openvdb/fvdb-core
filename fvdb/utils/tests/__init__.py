# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import functools
import site
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

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


def _clone_git_repo(git_url: str, git_tag: str, repo_name: str) -> Tuple[Path, git.repo.Repo]:
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
    repo.remotes.origin.fetch()
    repo.git.checkout(git_tag)

    return repo_path, repo


def _clone_fvdb_test_data() -> Tuple[Path, git.repo.Repo]:
    """Clone the fvdb-test-data repository for unit tests."""
    global git_tag_for_data
    git_url = "https://github.com/voxel-foundation/fvdb-test-data.git"
    return _clone_git_repo(git_url, git_tag_for_data, "fvdb_test_data")


def _clone_fvdb_example_data() -> Tuple[Path, git.repo.Repo]:
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
    conditional_args: Optional[List[List]] = None,
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

    Args:
        num_candidates (int): The number of random candidate points to generate
            and test. The final number of points returned will be less than or
            equal to this value.
        volume_shape (NumericMaxRank1): The (I, J, K) dimensions of the
            volume from which to sample points.
        min_separation (NumericMaxRank1): The minimum required separation between
            any two points, measured by Chebyshev distance.

    Returns:
        torch.Tensor: A list of accepted (i, j, k) coordinates.
    """
    device = resolve_device(device)
    volume_shape = to_Vec3i(volume_shape, value_constraint=ValueConstraint.POSITIVE)
    min_separation = to_Vec3i(min_separation, value_constraint=ValueConstraint.POSITIVE)

    num_candidates = int(num_candidates)
    I, J, K = volume_shape.tolist()

    # Generate tensor of random coordinates within the volume
    candidates = torch.stack(
        [
            torch.randint(0, I, (num_candidates,), dtype=dtype, device="cpu"),
            torch.randint(0, J, (num_candidates,), dtype=dtype, device="cpu"),
            torch.randint(0, K, (num_candidates,), dtype=dtype, device="cpu"),
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

    Returns:
        JaggedTensor: A jagged tensor containing the accepted (i, j, k) coordinates
            for each batch item. Each batch may have a different number of points.
    """
    volume_shapes = to_Vec3iBatchBroadcastable(volume_shapes, value_constraint=ValueConstraint.POSITIVE)
    min_separations = to_Vec3iBatchBroadcastable(min_separations, value_constraint=ValueConstraint.POSITIVE)

    return JaggedTensor(
        [
            generate_chebyshev_spaced_ijk(num_candidates, volume_shapes[i], min_separations[i], dtype, device)
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
        num_candidates, volume_shape, kernel_size, dtype=torch.long, device=device
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
]
