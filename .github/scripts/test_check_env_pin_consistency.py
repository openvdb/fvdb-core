#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared-toolchain-pin policy in check_env_pin_consistency.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_env_pin_consistency as policy  # noqa: E402

BUILD = """name: fvdb_build
dependencies:
  - cuda-version=13.0
  - python=3.12
  - pytorch-gpu=2.13.0
"""

DEV = """name: fvdb_dev
dependencies:
  - cuda-version=13.0
  - python=3.12
  - pytorch-gpu=2.13.0
  - ipython
"""

# Pins none of the shared keys, like release_base_environment.yml.
SLIM = """name: fvdb_release_base
dependencies:
  - cmake
"""


def write_env(tmp_path: Path, **files: str) -> Path:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    for name, text in files.items():
        (env_dir / f"{name}.yml").write_text(text, encoding="utf-8")
    return env_dir


def test_agreeing_envs_pass(tmp_path):
    env_dir = write_env(tmp_path, build_environment=BUILD, dev_environment=DEV)
    assert policy.main([str(env_dir)]) == 0


def test_file_pinning_nothing_is_allowed(tmp_path):
    env_dir = write_env(tmp_path, build_environment=BUILD, release_base_environment=SLIM)
    assert policy.main([str(env_dir)]) == 0


def test_disagreement_fails(tmp_path):
    env_dir = write_env(tmp_path, build_environment=BUILD, dev_environment=DEV.replace("2.13.0", "2.11.0"))
    assert policy.main([str(env_dir)]) == 1


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("python=3.12", "python=3.11"),
        ("cuda-version=13.0", "cuda-version=12.8"),
        ("pytorch-gpu=2.13.0", "pytorch-gpu=2.11.0"),
    ],
)
def test_disagreement_on_any_key_fails(tmp_path, old, new):
    env_dir = write_env(tmp_path, build_environment=BUILD, dev_environment=DEV.replace(old, new))
    assert policy.main([str(env_dir)]) == 1


def test_fix_propagates_from_canonical(tmp_path):
    env_dir = write_env(tmp_path, build_environment=BUILD, dev_environment=DEV.replace("2.13.0", "2.11.0"))
    assert policy.main([str(env_dir), "--fix"]) == 0
    assert "pytorch-gpu=2.13.0" in (env_dir / "dev_environment.yml").read_text(encoding="utf-8")
    # The fix must be surgical: unrelated content is preserved.
    assert "ipython" in (env_dir / "dev_environment.yml").read_text(encoding="utf-8")
    assert policy.main([str(env_dir)]) == 0


def test_missing_canonical_env_is_an_error(tmp_path):
    env_dir = write_env(tmp_path, dev_environment=DEV)
    with pytest.raises(SystemExit):
        policy.main([str(env_dir)])


def test_empty_env_dir_is_an_error(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    with pytest.raises(SystemExit):
        policy.main([str(env_dir)])
