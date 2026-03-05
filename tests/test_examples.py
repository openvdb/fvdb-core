# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# dilating_grids.py contains `while True: time.sleep(10.0)` in user code
# that cannot be neutralized by mocking the visualization library alone.
SKIP_EXAMPLES = {"dilating_grids.py"}


def _discover_examples():
    return sorted(p for p in EXAMPLES_DIR.glob("*.py") if p.name not in SKIP_EXAMPLES)


@pytest.fixture(autouse=True)
def _mock_visualization():
    """Replace visualization modules with no-op mocks so examples run headlessly."""
    originals = {}
    for mod_name in ("polyscope", "viser"):
        originals[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = MagicMock()

    yield

    for mod_name, orig in originals.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig


@pytest.mark.parametrize("example", _discover_examples(), ids=lambda p: p.stem)
def test_example_runs(example):
    runpy.run_path(str(example), run_name="__main__")
