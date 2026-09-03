# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the EC2 runner-token CI policy.

Exercises check_runner_token_policy.py -- the security gate that enforces that
``secrets.EC2_RUNNER_TOKEN`` only ever reaches ``machulav/ec2-github-runner``,
either as that action's ``github-token`` input or forwarded by name to a *local*
reusable workflow that this same scan covers. Run by
.github/workflows/workflow-security.yml on every PR (it needs only pyyaml +
pytest, no fvdb build).
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


def _check(tmp_path: Path, yaml_text: str, callees: tuple[str, ...] = ()) -> list[str]:
    """Write a workflow file and return the policy violations it produces.

    ``callees`` names sibling workflow files to create first, for the rule-3b
    check that a forwarding target actually exists in this directory (and is
    therefore covered by the same scan).
    """
    for name in callees:
        (tmp_path / name).write_text("name: callee\n")
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


# --- rule 5: no dynamic secret access (evades the textual name check) --------


def test_dynamic_secret_access_via_format_is_rejected(tmp_path):
    """secrets[format(...)] resolves the token without spelling its name."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          leak:
            runs-on: ubuntu-latest
            steps:
              - run: echo "${{ secrets[format('EC2_RUNNER_%s', 'TOKEN')] }}"
        """,
    )
    assert any("dynamic secret access" in v for v in violations)


def test_dynamic_secret_access_via_matrix_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          leak:
            runs-on: ubuntu-latest
            steps:
              - run: echo "${{ secrets[matrix.name] }}"
        """,
    )
    assert any("dynamic secret access" in v for v in violations)


def test_dynamic_secret_access_is_rejected_even_without_token_name(tmp_path):
    """The check runs on every workflow, even ones that never name the token."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          leak:
            runs-on: ubuntu-latest
            steps:
              - run: echo "${{ secrets['SOME_OTHER_SECRET'] }}"
        """,
    )
    assert any("dynamic secret access" in v for v in violations)


def test_identifier_ending_in_secrets_is_not_flagged(tmp_path):
    """`\\b` must not match e.g. mysecrets[0] (not a secrets-context access)."""
    violations = _check(
        tmp_path,
        """
        name: ok
        on: [push]
        jobs:
          build:
            runs-on: ubuntu-latest
            steps:
              - run: echo "${{ fromJSON(needs.x.outputs.mysecrets)[0] }}"
        """,
    )
    assert violations == []


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


# --- rule 3b: forwarding to a local reusable workflow ------------------------


def test_forward_to_local_reusable_workflow_is_allowed(tmp_path):
    """The DRY shape: the callee is a workflow file this same scan covers."""
    violations = _check(
        tmp_path,
        """
        name: ok
        on: [push]
        jobs:
          call:
            uses: ./.github/workflows/start-ec2-runner.yml
            secrets:
              EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
        callees=("start-ec2-runner.yml",),
    )
    assert violations == []


def test_callee_declaring_the_token_secret_is_allowed(tmp_path):
    """The other half of 3b: `on.workflow_call.secrets` naming the token."""
    violations = _check(
        tmp_path,
        """
        name: ok
        on:
          workflow_call:
            secrets:
              EC2_RUNNER_TOKEN:
                required: true
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


def test_forward_to_remote_reusable_workflow_is_rejected(tmp_path):
    """The hole `secrets: inherit` used to walk through: the token leaves the
    repo, where no scan of ours can see what the callee does with it."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            uses: other-org/other-repo/.github/workflows/x.yml@main
            secrets:
              EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert any("only be forwarded to a local reusable workflow" in v for v in violations)


def test_forward_to_missing_local_workflow_is_rejected(tmp_path):
    """A callee that is not in this directory would not be scanned."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            uses: ./.github/workflows/nope.yml
            secrets:
              EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
    )
    assert any("does not exist" in v for v in violations)


def test_forward_under_a_different_name_is_rejected(tmp_path):
    """One name everywhere, so a single grep still enumerates every file."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            uses: ./.github/workflows/start-ec2-runner.yml
            secrets:
              gh-token: ${{ secrets.EC2_RUNNER_TOKEN }}
        """,
        callees=("start-ec2-runner.yml",),
    )
    assert violations


def test_forward_from_a_job_with_no_uses_is_rejected(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            runs-on: ubuntu-latest
            secrets:
              EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}
            steps:
              - run: echo hi
        """,
    )
    assert any("no job-level uses" in v for v in violations)


def test_workflow_level_env_is_rejected(tmp_path):
    """Same textual shape as a legitimate forward -- caught structurally."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        env:
          EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}
        jobs:
          leak:
            runs-on: ubuntu-latest
            steps:
              - run: env
        """,
    )
    assert any("outside any job" in v for v in violations)


# --- rule 6: no blanket secret forwarding ------------------------------------


def test_secrets_inherit_is_rejected(tmp_path):
    """`inherit` forwards the admin token while naming nothing, so no textual
    check can see where it goes -- including a forward out of the repo."""
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            uses: ./.github/workflows/start-ec2-runner.yml
            secrets: inherit
        """,
        callees=("start-ec2-runner.yml",),
    )
    assert any("secrets: inherit" in v for v in violations)


def test_secrets_inherit_is_rejected_even_to_a_remote_workflow(tmp_path):
    violations = _check(
        tmp_path,
        """
        name: bad
        on: [push]
        jobs:
          call:
            uses: other-org/other-repo/.github/workflows/x.yml@main
            secrets: inherit
        """,
    )
    assert any("secrets: inherit" in v for v in violations)


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
