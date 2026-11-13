# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the EC2 runner GitHub App key CI policy.

Exercises check_runner_token_policy.py -- the security gate that enforces that
``secrets.RUNNER_APP_KEY`` (the fvdb-ec2-runners App's private key) is only used
by ``actions/create-github-app-token`` in a job that declares the ``ci-aws``
environment, and that the token it mints only reaches
``machulav/ec2-github-runner`` as its ``github-token`` input. Run by
.github/workflows/workflow-security.yml on every PR (it needs only pyyaml +
pytest, no fvdb build).
"""

from __future__ import annotations

import importlib.util
import subprocess
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

MINT = """\
      - name: Mint a runner-registration token
        id: runner-app-token
        uses: actions/create-github-app-token@64ab729a6294000f5aa4a4bc0b73d1f32bc45510
        with:
          client-id: ${{ vars.RUNNER_APP_CLIENT_ID }}
          private-key: ${{ secrets.RUNNER_APP_KEY }}
          permission-administration: write
"""

RUNNER = """\
      - uses: machulav/ec2-github-runner@343a1b2ae682e681c3cec9a235d882da17ff04ef
        with:
          mode: start
          github-token: ${{ steps.runner-app-token.outputs.token }}
"""


def _workflow(steps: str, job_extra: str = "    environment: ci-aws\n", top_extra: str = "") -> str:
    return (
        f"name: wf\non: [push]\n{top_extra}jobs:\n  start:\n    runs-on: ubuntu-latest\n{job_extra}    steps:\n{steps}"
    )


def _check(tmp_path: Path, yaml_text: str) -> list[str]:
    """Write a workflow file and return the policy violations it produces."""
    wf = tmp_path / "wf.yml"
    wf.write_text(yaml_text)
    violations: list[str] = []
    policy.check_workflow_file(wf, violations)
    return violations


def _has(violations: list[str], fragment: str) -> bool:
    return any(fragment in v for v in violations)


# --- compliant baselines ------------------------------------------------------


def test_compliant_workflow_has_no_violations(tmp_path):
    assert _check(tmp_path, _workflow(MINT + RUNNER)) == []


def test_environment_mapping_form_is_accepted(tmp_path):
    extra = "    environment:\n      name: ci-aws\n      deployment: false\n"
    assert _check(tmp_path, _workflow(MINT + RUNNER, job_extra=extra)) == []


def test_workflow_without_key_is_ignored(tmp_path):
    steps = "      - run: echo hello\n"
    assert _check(tmp_path, _workflow(steps, job_extra="")) == []


def test_identifier_ending_in_secrets_is_not_flagged(tmp_path):
    steps = "      - run: echo ${{ fromJSON('{}').mysecrets }}\n"
    assert _check(tmp_path, _workflow(steps, job_extra="")) == []


# --- rule 2: the one textual form ----------------------------------------------


def test_key_in_run_step_is_rejected(tmp_path):
    steps = MINT + "      - run: echo ${{ secrets.RUNNER_APP_KEY }}\n"
    assert _has(_check(tmp_path, _workflow(steps)), "may only appear as")


def test_key_in_job_env_is_rejected(tmp_path):
    extra = "    environment: ci-aws\n    env:\n      KEY: ${{ secrets.RUNNER_APP_KEY }}\n"
    violations = _check(tmp_path, _workflow(MINT + RUNNER, job_extra=extra))
    assert _has(violations, "may only appear as")
    assert _has(violations, "referenced at job level")


def test_key_in_workflow_env_is_rejected(tmp_path):
    top = "env:\n  KEY: ${{ secrets.RUNNER_APP_KEY }}\n"
    violations = _check(tmp_path, _workflow(MINT + RUNNER, top_extra=top))
    assert _has(violations, "referenced outside any job")


# --- rule 3: where the key may be used ----------------------------------------


def test_key_to_another_action_is_rejected(tmp_path):
    steps = (
        "      - id: other\n"
        "        uses: some/other-action@0123456789012345678901234567890123456789\n"
        "        with:\n"
        "          private-key: ${{ secrets.RUNNER_APP_KEY }}\n"
    )
    assert _has(_check(tmp_path, _workflow(steps)), "is not 'actions/create-github-app-token'")


def test_key_without_ci_aws_environment_is_rejected(tmp_path):
    assert _has(_check(tmp_path, _workflow(MINT + RUNNER, job_extra="")), "must declare the 'ci-aws' environment")


def test_key_in_other_environment_is_rejected(tmp_path):
    extra = "    environment: pypi\n"
    assert _has(_check(tmp_path, _workflow(MINT + RUNNER, job_extra=extra)), "must declare the 'ci-aws' environment")


def test_mint_step_without_id_is_rejected(tmp_path):
    steps = MINT.replace("        id: runner-app-token\n", "")
    assert _has(_check(tmp_path, _workflow(steps)), "must have an 'id'")


@pytest.mark.parametrize("extra_input", ["owner: openvdb", "repositories: fvdb-core", "skip-token-revoke: true"])
def test_widening_token_inputs_are_rejected(tmp_path, extra_input):
    steps = MINT + f"          {extra_input}\n" + RUNNER
    assert _has(_check(tmp_path, _workflow(steps)), "must stay limited to this repository")


# --- rule 4: no untrusted code alongside the key ------------------------------


def test_checkout_in_key_job_is_rejected(tmp_path):
    steps = "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n" + MINT + RUNNER
    assert _has(_check(tmp_path, _workflow(steps)), "actions/checkout")


def test_local_action_in_key_job_is_rejected(tmp_path):
    steps = MINT + "      - uses: ./.github/actions/thing\n" + RUNNER
    assert _has(_check(tmp_path, _workflow(steps)), "LOCAL action")


# --- rule 5: where the minted token may go ------------------------------------


def test_token_in_run_step_is_rejected(tmp_path):
    steps = MINT + "      - run: echo ${{ steps.runner-app-token.outputs.token }}\n"
    assert _has(_check(tmp_path, _workflow(steps)), "uses the token of step 'runner-app-token'")


def test_token_in_step_env_is_rejected(tmp_path):
    steps = (
        MINT
        + "      - run: gh api user\n        env:\n          GH_TOKEN: ${{ steps.runner-app-token.outputs.token }}\n"
    )
    assert _has(_check(tmp_path, _workflow(steps)), "uses the token of step 'runner-app-token'")


def test_token_to_another_action_is_rejected(tmp_path):
    steps = (
        MINT + "      - uses: some/other-action@0123456789012345678901234567890123456789\n"
        "        with:\n"
        "          github-token: ${{ steps.runner-app-token.outputs.token }}\n"
    )
    assert _has(_check(tmp_path, _workflow(steps)), "uses the token of step 'runner-app-token'")


def test_token_via_other_runner_input_is_rejected(tmp_path):
    steps = MINT + RUNNER + "          label: ${{ steps.runner-app-token.outputs.token }}\n"
    assert _has(_check(tmp_path, _workflow(steps)), "uses the token of step 'runner-app-token'")


def test_token_in_concatenated_value_is_rejected(tmp_path):
    steps = MINT + RUNNER.replace(
        "${{ steps.runner-app-token.outputs.token }}", "x${{ steps.runner-app-token.outputs.token }}"
    )
    assert _has(_check(tmp_path, _workflow(steps)), "uses the token of step 'runner-app-token'")


def test_token_in_job_outputs_is_rejected(tmp_path):
    extra = "    environment: ci-aws\n    outputs:\n      t: ${{ steps.runner-app-token.outputs.token }}\n"
    assert _has(_check(tmp_path, _workflow(MINT + RUNNER, job_extra=extra)), "referenced at job level")


# --- rules 6, 7 and 8 ---------------------------------------------------------


def test_dynamic_secret_access_is_rejected(tmp_path):
    steps = "      - run: echo ${{ secrets[format('RUNNER_APP_{0}', 'KEY')] }}\n"
    assert _has(_check(tmp_path, _workflow(steps, job_extra="")), "dynamic secret access")


def _caller(uses: str, secrets_line: str = "    secrets: inherit\n") -> str:
    return f"name: wf\non: [push]\njobs:\n  call:\n    uses: {uses}\n{secrets_line}"


def test_secrets_inherit_to_local_workflow_is_allowed(tmp_path):
    (tmp_path / "start-ec2-runner.yml").write_text("name: callee\n")
    assert _check(tmp_path, _caller("./.github/workflows/start-ec2-runner.yml")) == []


def test_secrets_inherit_with_trailing_comment_is_still_checked(tmp_path):
    line = "    secrets: inherit # zizmor: ignore[secrets-inherit]\n"
    violations = _check(
        tmp_path, _caller("some/org/.github/workflows/x.yml@0123456789012345678901234567890123456789", line)
    )
    assert _has(violations, "only allowed into a local reusable workflow")


def test_secrets_inherit_to_remote_workflow_is_rejected(tmp_path):
    violations = _check(tmp_path, _caller("some/org/.github/workflows/x.yml@0123456789012345678901234567890123456789"))
    assert _has(violations, "only allowed into a local reusable workflow")


def test_secrets_inherit_to_missing_local_workflow_is_rejected(tmp_path):
    assert _has(_check(tmp_path, _caller("./.github/workflows/missing.yml")), "does not exist")


def test_secret_outside_allowlist_is_rejected(tmp_path):
    steps = "      - run: echo hi\n        env:\n          X: ${{ secrets.CODEBUILD_SECRET }}\n"
    assert _has(_check(tmp_path, _workflow(steps, job_extra="")), "'CODEBUILD_SECRET' is not on the allowlist")


def test_allowlisted_secrets_are_accepted(tmp_path):
    steps = "      - run: gh api user\n        env:\n          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n"
    assert _check(tmp_path, _workflow(steps, job_extra="")) == []


def test_retired_pat_name_is_rejected(tmp_path):
    steps = (
        "      - uses: machulav/ec2-github-runner@343a1b2ae682e681c3cec9a235d882da17ff04ef\n"
        "        with:\n"
        "          github-token: ${{ secrets.EC2_RUNNER_TOKEN }}\n"
    )
    assert _has(_check(tmp_path, _workflow(steps, job_extra="")), "'EC2_RUNNER_TOKEN' is retired")


# --- the real fvdb workflows must all pass -----------------------------------


def test_repo_workflows_are_compliant():
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    violations: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        policy.check_workflow_file(path, violations)
    assert violations == [], "real workflows violate the runner-token policy:\n" + "\n".join(violations)


def test_repo_runner_workflows_use_the_app_key():
    """Guard against the policy passing vacuously: both runner workflows mint
    the token from the App key."""
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    for name in ("start-ec2-runner.yml", "stop-ec2-runner.yml"):
        text = (workflow_dir / name).read_text()
        assert "private-key: ${{ secrets.RUNNER_APP_KEY }}" in text, name
        assert "github-token: ${{ steps.runner-app-token.outputs.token }}" in text, name


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
    the key name only appears in allowed CI paths at HEAD."""
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
