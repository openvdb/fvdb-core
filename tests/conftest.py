# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

# Release the CUDA caching allocator's reserved pool periodically
_EMPTY_CACHE_EVERY = 10
_cuda_test_count = 0


@pytest.fixture(autouse=True)
def _free_cuda_memory(request):
    """Periodically release the CUDA caching allocator's reserved pool.

    Without this, the allocator's reserved high-water mark only grows across the
    full parametrized matrix (all tests share one process), inflating GPU memory
    usage well beyond any single test's live footprint.
    """
    yield
    # parameterized.expand encodes the device into the test id (e.g. "..._cuda"),
    # so CPU-only cases are skipped without importing torch or touching the GPU.
    if "cuda" not in request.node.name.lower():
        return
    global _cuda_test_count
    _cuda_test_count += 1
    if _cuda_test_count % _EMPTY_CACHE_EVERY == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pytest_addoption(parser):
    parser.addoption(
        "--doc-dirs",
        action="append",
        default=None,
        help="Directories of markdown files to scan for API ref validation "
        "(relative to repo root). Repeatable. Default: docs/TEACHME, docs/tutorials",
    )
