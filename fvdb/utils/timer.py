# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import time

import torch


class ScopedTimer:
    """
    A context manager for timing code blocks with optional CUDA synchronization.

    Examples:
        Basic usage:
        >>> with ScopedTimer(message="Elapsed time"):
        ...     # some code
        ...     pass
        >>> # Elapsed time: 1.2345 seconds

        With CUDA synchronization:
        >>> with ScopedTimer(message="GPU operation", cuda=True):
        ...     # GPU operations
        ...     result = torch.matmul(a, b)
        >>> # GPU operation: 1.2345 seconds

        Splitting timings:
        >>> with ScopedTimer(cuda=True) as timer:
        ...     # setup phase
        ...     kmap, _ = grid.sparse_conv_kernel_map(kernel_size=kernel.size(-1), stride=1)
        ...     setup_time = timer.split()
        ...     # computation phase
        ...     out_feature = kmap.sparse_conv_3d(in_feature, kernel)
        ...     compute_time = timer.split()
        >>> print(f"Setup: {setup_time:.4f}s, Compute: {compute_time:.4f}s")
    """

    def __init__(self, cuda: bool = False, message: str = ""):
        """
        Initialize the timer.

        Args:
            cuda (bool): If True, calls torch.cuda.synchronize() before timing measurements.
                        Defaults to False.
        """
        self.cuda: bool = cuda
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed_time: float | None = None
        self._last_split_time: float | None = None
        self.message: str = message

    def __enter__(self):
        """Enter the context manager and start timing."""
        if self.cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        self._last_split_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and calculate elapsed time."""
        if self.cuda:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        assert isinstance(self.start_time, float)
        self.elapsed_time = self.end_time - self.start_time
        if self.message != "":
            print(f"{self.message}: {self.elapsed_time:.4f} seconds")

    def split(self) -> float:
        """
        Get the time elapsed since the last split (or start if no previous split).

        Returns:
            float: Time elapsed since last split in seconds.
        """
        if self.cuda:
            torch.cuda.synchronize()
        current_time = time.perf_counter()

        if self._last_split_time is None:
            raise RuntimeError("ScopedTimer must be used within a 'with' block before calling split()")

        split_time = current_time - self._last_split_time
        self._last_split_time = current_time
        return split_time
