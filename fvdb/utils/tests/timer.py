# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any

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
        ...     plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid)
        ...     setup_time = timer.split()
        ...     # computation phase
        ...     out_feature = plan.execute(in_feature, kernel)
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
        # CUDA event-based timing (created/used only when cuda=True)
        self._start_event: Any | None = None
        self._last_split_event: Any | None = None

    def __enter__(self):
        """Enter the context manager and start timing."""
        if self.cuda:
            # Record a CUDA event on the current stream to mark the start.
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            self._start_event = start_event
            self._last_split_event = start_event
            # Also initialize CPU timer for non-CUDA fallback paths if needed
            self.start_time = None
            self._last_split_time = None
        else:
            self.start_time = time.perf_counter()
            self._last_split_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and calculate elapsed time."""
        if self.cuda:
            # Record an end event and measure elapsed time using CUDA timing
            assert self._start_event is not None
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream())
            # Ensure end event has completed without a device-wide sync
            end_event.synchronize()
            self.elapsed_time = self._start_event.elapsed_time(end_event) / 1000.0
            self.end_time = None
        else:
            self.end_time = time.perf_counter()
            assert isinstance(self.start_time, float)
            self.elapsed_time = self.end_time - self.start_time
        if self.message != "" and isinstance(self.elapsed_time, float):
            print(f"{self.message}: {self.elapsed_time:.4f} seconds")

    def split(self) -> float:
        """
        Get the time elapsed since the last split (or start if no previous split).

        Returns:
            float: Time elapsed since last split in seconds.
        """
        if self.cuda:
            # Measure time since last split using CUDA events
            if self._last_split_event is None:
                raise RuntimeError("ScopedTimer must be used within a 'with' block before calling split()")
            current_event = torch.cuda.Event(enable_timing=True)
            current_event.record(torch.cuda.current_stream())
            current_event.synchronize()
            split_ms = self._last_split_event.elapsed_time(current_event)
            self._last_split_event = current_event
            return split_ms / 1000.0
        else:
            current_time = time.perf_counter()
            if self._last_split_time is None:
                raise RuntimeError("ScopedTimer must be used within a 'with' block before calling split()")
            split_time = current_time - self._last_split_time
            self._last_split_time = current_time
            return split_time
