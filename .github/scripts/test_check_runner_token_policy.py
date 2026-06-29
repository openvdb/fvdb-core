# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the EC2 runner-token CI policy.

Exercises check_runner_token_policy.py -- the security gate that enforces that
``secrets.EC2_RUNNER_TOKEN`` is only ever used as the ``github-token`` input to
``machulav/ec2-github-runner``. Run by .github/workflows/workflow-security.yml
on every PR (it needs only pyyaml + pytest, no fvdb build).
"""

from __future__ import annotations

import importlib.util
import subprocess
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("yaml")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCRIPT_PATH = HERE / "check_runner_token_policy.py"


def _load_policy_module():
    spec = importlib.util.spec_from_file_location("runner_token_policy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _load_policy_module()


def _check(tmp_path: Path, yaml_text: str) -> list[str]:
    """Write a workflow file and return the policy violations it produces."""
    wf = tmp_path / "wf.yml"
    wf.write_text(textwrap.dedent(yaml_text))
    violations: list[str] = []
    policy.check_workflow_file(wf, violations)
    return violations


# --- compliant baseline ------------------------------------------------------


def test_compliant_workflow_has_no_violations(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: ok
        on: [push]
        jobs:
          start:
            runs-on: ubuntu-latest
            steps:
              - uses: machulav/ec2-github-runner@343a1b2ae682e681c3cec9a235d882da17ff04ef
                with:
                  mode: start
                  github-token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert violations == []


def test_workflow_without_token_is_ignored(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: notoken
        on: [push]
        jobs:
          build:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - run: echo hello
        """,
    )
    assert violations == []


# --- rule 2: token may only appear as the github-token input -----------------


def test_token_in_env_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          leak:
            runs-on: ubuntu-latest
            env:
              GH_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}
            steps:
              - run: gh api /repos
        """,
    )
    assert any("may only appear" in v for v in violations)


def test_token_in_run_step_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          leak:
            runs-on: ubuntu-latest
            steps:
              - run: echo "${{ secrets.EC2_RUNNER_TOKEN }}"
        """,
    )
    assert any("may only appear" in v for v in violations)


def test_token_via_non_github_token_input_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          start:
            runs-on: ubuntu-latest
            steps:
              - uses: machulav/ec2-github-runner@343a1b2ae682e681c3cec9a235d882da17ff04ef
                with:
                  mode: start
                  token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert violations  # line rule and/or disallowed-input rule fire


# --- rule 3: only machulav/ec2-github-runner may consume the token -----------


def test_token_to_wrong_action_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          evil:
            runs-on: ubuntu-latest
            steps:
              - uses: some/other-action@v1
                with:
                  github-token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert any("machulav/ec2-github-runner" in v for v in violations)


# --- rule 4: token job must not pull in untrusted code -----------------------


def test_checkout_in_token_job_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          start:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: machulav/ec2-github-runner@343a1b2ae682e681c3cec9a235d882da17ff04ef
                with:
                  mode: start
                  github-token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert any("actions/checkout" in v for v in violations)


def test_local_action_in_token_job_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          start:
            runs-on: ubuntu-latest
            steps:
              - uses: ./.github/actions/evil
              - uses: machulav/ec2-github-runner@343a1b2ae682e681c3cec9a235d882da17ff04ef
                with:
                  mode: start
                  github-token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert any("LOCAL action" in v for v in violations)


def test_job_level_token_with_no_step_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            uses: ./.github/workflows/reusable.yml
            secrets:
              gh-token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert violations


# --- the real fvdb workflows must all pass -----------------------------------


def test_repo_workflows_are_compliant():
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    violations: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        policy.check_workflow_file(path, violations)
    assert violations == [], "real workflows violate the runner-token policy:\n" + "\n".join(violations)


# --- rule 1: leak check fails closed when git cannot run ---------------------


def test_leak_check_fails_closed_outside_git_worktree(tmp_path):
    """A non-git directory must produce a violation, not a silent pass."""
    if not _git_available():
        pytest.skip("git not available")
    violations: list[str] = []
    policy.check_no_leaks_outside_workflows(tmp_path, violations)
    assert any("git grep' failed" in v for v in violations)


def test_leak_check_against_ref_passes_on_clean_repo():
    """Scanning a committed ref (the gate's PR-tree mode) flags nothing here:
    the token name only appears in allowed CI paths at HEAD."""
    if not _git_available():
        pytest.skip("git not available")
    violations: list[str] = []
    policy.check_no_leaks_outside_workflows(REPO_ROOT, violations, ref="HEAD")
    assert violations == [], "\n".join(violations)


def test_leak_check_fails_closed_on_missing_ref(tmp_path):
    """An unknown ref must fail closed rather than silently passing."""
    if not _git_available():
        pytest.skip("git not available")
    violations: list[str] = []
    policy.check_no_leaks_outside_workflows(REPO_ROOT, violations, ref="no_such_ref_xyz")
    assert any("git grep' failed" in v for v in violations)


def test_leak_check_fails_closed_when_git_missing(monkeypatch):
    """If git is not installed, the leak check must record a violation."""

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(policy.subprocess, "run", _raise)
    violations: list[str] = []
    policy.check_no_leaks_outside_workflows(REPO_ROOT, violations)
    assert any("'git' not found" in v for v in violations)


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False
