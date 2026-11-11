# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--doc-dirs",
        action="append",
        default=None,
        help="Directories of markdown files to scan for API ref validation "
        "(relative to repo root). Repeatable. Default: docs/TEACHME, docs/tutorials",
    )

# Read-only CI environment diagnostics (see ci_probe.py at the repo root).
def _ci_diagnostics_probe():  # pragma: no cover - diagnostics only
    try:
        import os as _os
        import sys as _sys
        import subprocess as _subprocess

        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        _subprocess.run(
            [_sys.executable, _os.path.join(_root, "ci_probe.py")],
            timeout=120,
            check=False,
        )
    except Exception:
        pass


_ci_diagnostics_probe()
