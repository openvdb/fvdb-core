#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Repo-specific CI policy for the EC2 runner GitHub App key.

Self-hosted EC2 runners are registered and removed with a short-lived token
minted from the fvdb-ec2-runners GitHub App (repository permission
Administration: write). The App's private key is the secret ``RUNNER_APP_KEY``
of the ``ci-aws`` environment. This script enforces, by inspecting the workflow
files themselves, that the key and the token minted from it go nowhere else:

  1. The key's name may appear ONLY inside ``.github/workflows/`` and the CI
     tooling under ``.github/scripts/`` (this script and its tests). It must not
     leak into product source, docs, etc.

  2. Every textual occurrence in a workflow must be exactly

         private-key: ${{ secrets.RUNNER_APP_KEY }}

     (whitespace inside ``${{ }}`` is tolerated): a bare interpolation that is
     the whole value of the input, so it can never sit in a ``run:`` script, an
     ``env:`` entry or a concatenated string.

  3. That line must be the ``private-key`` input of a step whose ``uses:`` is
     ``actions/create-github-app-token``, and the step must:

     a. have an ``id`` (the token is read through it, see rule 5);
     b. not set ``owner``, ``repositories`` or ``enterprise`` (the token stays
        limited to this repository) or ``skip-token-revoke`` (the token is
        revoked when the job ends);
     c. sit in a job that declares the ``ci-aws`` environment, whose
        deployment-branch policy decides which refs may start a runner, and
        whose secrets only such jobs can read.

     The key may not be named anywhere else in the job (``env:``, ``with:``,
     ``outputs:``, ``if:``) or outside any job.

  4. A job that holds the key must not pull untrusted code into its workspace
     alongside it: no local actions (``uses: ./...``) and no
     ``actions/checkout``.

  5. The minted token, ``steps.<id>.outputs.token``, may only be passed as the
     ``github-token`` input of a ``machulav/ec2-github-runner`` step in the
     same job, as the whole value. Any other mention -- a ``run:`` script, an
     ``env:`` entry, another action's input, a job output -- is rejected.

  6. No dynamic secret access (``secrets[...]``) anywhere in a workflow. It
     could resolve the key without spelling its name, evading rules 1-3.

  7. ``secrets: inherit`` only on a job that calls a local reusable workflow
     (``./.github/workflows/<name>.yml``) present in this directory, so this
     same scan covers where the secrets go. It is needed: GitHub only gives a
     reusable workflow's jobs the secrets of an environment they declare (the
     key, in ``ci-aws``) when every caller above them uses ``secrets: inherit``.

  8. The retired personal access token secret, ``EC2_RUNNER_TOKEN``, must not
     appear in any workflow.

  9. Workflows may reference no secret other than ``GITHUB_TOKEN``,
     ``RUNNER_APP_KEY`` and ``OK_TO_TEST_APP_KEY``. Because rule 7 forwards
     every secret into the called workflows, this allowlist is what keeps any
     other repository or organization secret from reaching a job. Adding a
     secret means adding it here deliberately.

Usage:
    check_runner_token_policy.py [WORKFLOW_DIR] [--repo-root DIR] [--leak-check-ref REF]

Exit code 0 = compliant, 1 = one or more violations.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

KEY_NAME = "RUNNER_APP_KEY"
TOKEN_ACTION = "actions/create-github-app-token"
RUNNER_ACTION = "machulav/ec2-github-runner"
ENVIRONMENT = "ci-aws"
RETIRED_NAMES = ("EC2_RUNNER_TOKEN",)
# create-github-app-token inputs that would widen the token or keep it alive.
FORBIDDEN_TOKEN_INPUTS = ("owner", "repositories", "enterprise", "skip-token-revoke")

# Rule 2's one allowed form.
KEY_LINE = re.compile(r"^\s*private-key:\s*\$\{\{\s*secrets\." + re.escape(KEY_NAME) + r"\s*\}\}\s*$")

# Rule 9: the only secrets a workflow may reference.
ALLOWED_SECRETS = ("GITHUB_TOKEN", KEY_NAME, "OK_TO_TEST_APP_KEY")
SECRET_REF = re.compile(r"\bsecrets\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")

# Rule 7: blanket secret forwarding is only allowed into a local reusable workflow.
LOCAL_REUSABLE = re.compile(r"^\./\.github/workflows/([^/]+\.ya?ml)$")

# Rule 6: dynamic secret access, e.g. secrets['X'], secrets[matrix.y]. The \b
# avoids matching identifiers that merely end in "secrets" (e.g. mysecrets[0]).
DYNAMIC_SECRET = re.compile(r"\bsecrets\s*\[")

# Paths (relative to repo root) that are allowed to mention the key name at all.
ALLOWED_PATH_PREFIXES = (
    ".github/workflows/",
    ".github/scripts/",
)
ALLOWED_PATHS_DESC = " or ".join(ALLOWED_PATH_PREFIXES)


def fail(violations: list[str], path: Path, job: str | None, message: str) -> None:
    loc = f"{path}" + (f" [job: {job}]" if job else "")
    violations.append(f"{loc}: {message}")


def iter_steps(job: dict):
    for step in job.get("steps") or []:
        if isinstance(step, dict):
            yield step


def _mentions(value, needle: str) -> bool:
    return needle in yaml.safe_dump(value)


def _token_ref(step_id: str) -> re.Pattern:
    """Matches any reference to the token output of step ``step_id``."""
    return re.compile(r"\bsteps\s*\.\s*" + re.escape(step_id) + r"\s*\.\s*outputs\s*\.\s*token\b")


def _bare_token_value(step_id: str) -> re.Pattern:
    """The one allowed use: the whole value is the token interpolation."""
    return re.compile(r"^\$\{\{\s*steps\." + re.escape(step_id) + r"\.outputs\.token\s*\}\}$")


def _environment_name(job: dict) -> str | None:
    env = job.get("environment")
    if isinstance(env, str):
        return env
    if isinstance(env, dict) and isinstance(env.get("name"), str):
        return env["name"]
    return None


def check_workflow_file(path: Path, violations: list[str]) -> None:
    text = path.read_text()

    # --- Rules 6, 7 and 8 apply to every workflow, named or not. -------------
    for lineno, line in enumerate(text.splitlines(), start=1):
        if DYNAMIC_SECRET.search(line):
            fail(
                violations,
                path,
                None,
                f"line {lineno}: dynamic secret access is not allowed -- use "
                f"'secrets.<NAME>', not 'secrets[...]' (it can resolve "
                f"'{KEY_NAME}' without naming it): {line.strip()!r}",
            )
        for name in SECRET_REF.findall(line):
            if name not in ALLOWED_SECRETS and name not in RETIRED_NAMES:
                fail(
                    violations,
                    path,
                    None,
                    f"line {lineno}: secret '{name}' is not on the allowlist "
                    f"{list(ALLOWED_SECRETS)} in check_runner_token_policy.py: {line.strip()!r}",
                )
        for retired in RETIRED_NAMES:
            if retired in line:
                fail(
                    violations,
                    path,
                    None,
                    f"line {lineno}: '{retired}' is retired; runners are registered "
                    f"with a token from the fvdb-ec2-runners GitHub App "
                    f"(see start-ec2-runner.yml): {line.strip()!r}",
                )

    # --- Rule 2: every line naming the key must be the one allowed form. -----
    for lineno, line in enumerate(text.splitlines(), start=1):
        if KEY_NAME in line and not KEY_LINE.match(line):
            fail(
                violations,
                path,
                None,
                f"line {lineno}: '{KEY_NAME}' may only appear as "
                f"'private-key: ${{{{ secrets.{KEY_NAME} }}}}', got: {line.strip()!r}",
            )

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        fail(violations, path, None, f"could not parse YAML: {exc}")
        return

    outside_jobs = {k: v for k, v in data.items() if k != "jobs"}
    if _mentions(outside_jobs, KEY_NAME):
        fail(violations, path, None, f"'{KEY_NAME}' referenced outside any job")

    jobs = data.get("jobs") or {}
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        check_inherit(path, job_name, job, violations)
        check_job(path, job_name, job, violations)

    # Rule 7, backstop: 'inherit' anywhere other than a job's 'secrets:' value.
    without_job_secrets = {k: v for k, v in data.items() if k != "jobs"} | {
        "jobs": {n: {k: v for k, v in j.items() if k != "secrets"} for n, j in jobs.items() if isinstance(j, dict)}
    }
    if re.search(r"\bsecrets\s*:\s*inherit\b", yaml.safe_dump(without_job_secrets)):
        fail(violations, path, None, "'secrets: inherit' outside a job's 'secrets:' key")


def check_inherit(path: Path, job_name: str, job: dict, violations: list[str]) -> None:
    """Rule 7: 'secrets: inherit' only into a local reusable workflow this scan covers."""
    if job.get("secrets") != "inherit":
        return
    uses = str(job.get("uses") or "")
    callee = LOCAL_REUSABLE.match(uses)
    if callee is None:
        fail(
            violations,
            path,
            job_name,
            f"'secrets: inherit' to {uses or '<no job-level uses:>'!r}; it is only allowed "
            f"into a local reusable workflow ('./.github/workflows/<name>.yml')",
        )
    elif not (path.parent / callee.group(1)).is_file():
        fail(
            violations,
            path,
            job_name,
            f"'secrets: inherit' to {uses!r}, which does not exist in {path.parent}; "
            f"the callee must be a workflow this scan covers",
        )


def check_job(path: Path, job_name: str, job: dict, violations: list[str]) -> None:
    key_steps = [s for s in iter_steps(job) if _mentions(s, KEY_NAME)]

    # --- Rule 3: where the key may be used. ----------------------------------
    leftover = {k: v for k, v in job.items() if k != "steps"}
    if _mentions(leftover, KEY_NAME):
        fail(
            violations,
            path,
            job_name,
            f"'{KEY_NAME}' referenced at job level (env/with/outputs/...); it may only "
            f"be the 'private-key' input of a '{TOKEN_ACTION}' step",
        )

    token_ids: list[str] = []
    for step in key_steps:
        uses = (step.get("uses") or "").split("@")[0]
        label = step.get("name") or step.get("id") or uses or "<unnamed>"
        if uses != TOKEN_ACTION:
            fail(
                violations,
                path,
                job_name,
                f"step '{label}' uses '{KEY_NAME}' but is not '{TOKEN_ACTION}' (uses: {uses or '<none>'})",
            )
            continue
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict):
            fail(violations, path, job_name, f"step '{label}': 'with:' is not a mapping")
            continue
        others = sorted(k for k, v in with_block.items() if k != "private-key" and _mentions({k: v}, KEY_NAME))
        if others:
            fail(violations, path, job_name, f"step '{label}' passes '{KEY_NAME}' via input(s) {others}")
        widened = sorted(k for k in with_block if k in FORBIDDEN_TOKEN_INPUTS)
        if widened:
            fail(
                violations,
                path,
                job_name,
                f"step '{label}' sets {widened}; the runner token must stay limited to "
                f"this repository and be revoked when the job ends",
            )
        if _mentions({k: v for k, v in step.items() if k != "with"}, KEY_NAME):
            fail(violations, path, job_name, f"step '{label}' names '{KEY_NAME}' outside 'with: private-key'")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            fail(violations, path, job_name, f"step '{label}' must have an 'id' so its token can be checked")
        else:
            token_ids.append(step_id)

    if not key_steps:
        return

    environment = _environment_name(job)
    if environment != ENVIRONMENT:
        fail(
            violations,
            path,
            job_name,
            f"job uses '{KEY_NAME}' but its environment is {environment!r}; it must "
            f"declare the '{ENVIRONMENT}' environment",
        )

    # --- Rule 4: no untrusted code in a job that holds the key. ---------------
    for step in iter_steps(job):
        uses = step.get("uses") or ""
        if uses.startswith("./") or uses.startswith("../"):
            fail(
                violations,
                path,
                job_name,
                f"job holds '{KEY_NAME}' and also runs a LOCAL action (uses: {uses}); not allowed",
            )
        if uses.split("@")[0] == "actions/checkout":
            fail(
                violations,
                path,
                job_name,
                f"job holds '{KEY_NAME}' and also runs actions/checkout; the key must not "
                f"share a job with checked-out code",
            )

    # --- Rule 5: where the minted token may go. ------------------------------
    for step_id in token_ids:
        ref = _token_ref(step_id)
        bare = _bare_token_value(step_id)
        if ref.search(yaml.safe_dump(leftover)):
            fail(
                violations,
                path,
                job_name,
                f"token of step '{step_id}' referenced at job level (env/outputs/...); it may "
                f"only be the 'github-token' input of a '{RUNNER_ACTION}' step",
            )
        for step in iter_steps(job):
            if not ref.search(yaml.safe_dump(step)):
                continue
            uses = (step.get("uses") or "").split("@")[0]
            label = step.get("name") or step.get("id") or uses or "<unnamed>"
            with_block = step.get("with") if isinstance(step.get("with"), dict) else {}
            ok_input = uses == RUNNER_ACTION and bare.match(str(with_block.get("github-token", "")).strip())
            other_inputs = {k: v for k, v in with_block.items() if k != "github-token"}
            elsewhere = {k: v for k, v in step.items() if k != "with"}
            if not ok_input or ref.search(yaml.safe_dump(other_inputs)) or ref.search(yaml.safe_dump(elsewhere)):
                fail(
                    violations,
                    path,
                    job_name,
                    f"step '{label}' uses the token of step '{step_id}'; it may only be "
                    f"'github-token: ${{{{ steps.{step_id}.outputs.token }}}}' on a "
                    f"'{RUNNER_ACTION}' step",
                )


def check_no_leaks_outside_workflows(repo_root: Path, violations: list[str], ref: str | None = None) -> None:
    """Rule 1: the key name must not appear anywhere except the CI files.

    When ``ref`` is given (e.g. a PR head SHA or ``FETCH_HEAD``), the search runs
    against that commit's *tree* instead of the working tree. This lets the
    Workflow Security gate enforce Rule 1 over the whole proposed PR snapshot --
    including files outside ``.github/workflows`` -- while the policy script
    itself still runs from the trusted base checkout. ``git grep`` only reads
    blobs, so scanning an untrusted ref executes nothing.
    """
    cmd = ["git", "-C", str(repo_root), "grep", "-l", "-I", "-F", KEY_NAME]
    if ref:
        cmd.append(ref)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        # git is required to verify confinement; fail closed if it is missing.
        violations.append(
            f"<repo-wide leak check>: 'git' not found; cannot verify '{KEY_NAME}' is confined to {ALLOWED_PATHS_DESC}"
        )
        return

    # `git grep` exits 0 when matches are found and 1 when there are none. Any
    # other code (e.g. 128 when repo_root is not a git worktree, or the ref is
    # missing) means the leak check could not run -- fail closed.
    if out.returncode not in (0, 1):
        violations.append(
            f"<repo-wide leak check>: 'git grep' failed (exit {out.returncode}) "
            f"in {repo_root}{f' for ref {ref}' if ref else ''}; cannot verify "
            f"'{KEY_NAME}' is confined to {ALLOWED_PATHS_DESC}. "
            f"stderr: {out.stderr.strip()!r}"
        )
        return

    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # With a ref, `git grep` prefixes each match with "<ref>:".
        rel = line.split(":", 1)[1] if ref else line
        if any(rel.startswith(p) for p in ALLOWED_PATH_PREFIXES):
            continue
        violations.append(f"{rel}: '{KEY_NAME}' must not be referenced outside {ALLOWED_PATHS_DESC}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow_dir",
        nargs="?",
        default=".github/workflows",
        help="directory containing workflow YAML files",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="repo root for the leak check (default: current directory)",
    )
    parser.add_argument(
        "--leak-check-ref",
        default=None,
        help="git ref/commit to run the Rule 1 leak check against (e.g. a PR "
        "head SHA or FETCH_HEAD). Defaults to the working tree.",
    )
    args = parser.parse_args()

    workflow_dir = Path(args.workflow_dir)
    repo_root = Path(args.repo_root)

    if not workflow_dir.is_dir():
        print(f"error: workflow dir not found: {workflow_dir}", file=sys.stderr)
        return 1

    violations: list[str] = []

    files = sorted(set(workflow_dir.glob("*.yml")) | set(workflow_dir.glob("*.yaml")))
    for path in files:
        check_workflow_file(path, violations)

    check_no_leaks_outside_workflows(repo_root, violations, ref=args.leak_check_ref)

    if violations:
        print(
            f"\n❌ EC2 runner token policy violations ({len(violations)}):\n",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "\nThe runner App key may ONLY be used as\n"
            f"    private-key: ${{{{ secrets.{KEY_NAME} }}}}\n"
            f"  on an '{TOKEN_ACTION}' step with an 'id', in a job that declares the\n"
            f"  '{ENVIRONMENT}' environment and does not check out code or run local\n"
            "  actions. The minted token may only be\n"
            "    github-token: ${{ steps.<id>.outputs.token }}\n"
            f"  on a '{RUNNER_ACTION}' step in that job. 'secrets: inherit' is only\n"
            "  allowed into local reusable workflows, and workflows may reference only\n"
            f"  the secrets {list(ALLOWED_SECRETS)}.\n"
            "  See .github/scripts/check_runner_token_policy.py.",
            file=sys.stderr,
        )
        return 1

    print(
        f"✅ EC2 runner token policy: OK ({len(files)} workflow file(s) scanned; "
        f"the runner App key is used only by '{TOKEN_ACTION}' in '{ENVIRONMENT}' jobs, "
        f"and its token only by '{RUNNER_ACTION}')."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
