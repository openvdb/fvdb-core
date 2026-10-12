# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest
import torch

# Runs in a fresh interpreter so the fork does not inherit pytest's state. The grid sits in an unreachable
# reference cycle at fork time, so the child's cycle collector frees its own copy of it, as happens in
# forked DataLoader workers.
_FORK_FREE_SCRIPT = textwrap.dedent(
    """
    import gc
    import multiprocessing as mp

    import torch

    import fvdb


    def child():
        gc.collect()


    if __name__ == "__main__":
        gc.disable()
        ijk = torch.randint(0, 32, (1000, 3), device="cuda", dtype=torch.int32)
        holder = {"grid": fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([ijk]))}
        holder["self"] = holder
        del holder
        p = mp.get_context("fork").Process(target=child)
        p.start()
        p.join()
        raise SystemExit(p.exitcode)
    """
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forked_child_frees_inherited_cuda_grid():
    # With caching disabled, every free calls cudaFree, which fails in a forked child for torch tensors too.
    # Freeing in a forked child is only supported with the caching allocator, so run with it enabled.
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_NO_CUDA_MEMORY_CACHING"}
    result = subprocess.run(
        [sys.executable, "-c", _FORK_FREE_SCRIPT], capture_output=True, text=True, timeout=300, env=env
    )
    assert result.returncode == 0, f"forked child exited with {result.returncode}:\n{result.stderr}"
