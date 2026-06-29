#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Repo-specific CI policy for the EC2 runner admin token.

This script enforces, by inspecting the workflow files themselves:

  1. The token name may appear ONLY inside ``.github/workflows/*.{yml,yaml}``
     (and in this enforcement script). It must not leak into source, docs, etc.

  2. Every textual occurrence in a workflow must be EXACTLY the action input:
         github-token: ${{ secrets.EC2_RUNNER_TOKEN }}
     (whitespace inside ``${{ }}`` is tolerated). This single rule already
     forbids putting the token in ``env:``, ``GH_TOKEN``/``GITHUB_TOKEN``,
     ``with.token``, a ``run:`` script, or a reusable-workflow ``secrets:``
     block, because none of those match this pattern.

  3. The step that consumes the token must ``uses: machulav/ec2-github-runner``.

  4. A job that references the token must not pull untrusted code into its
     workspace alongside the privileged context: no local actions
     (``uses: ./...``) and no ``actions/checkout``. (Sibling ``run:`` steps are
     fine -- rule 2 guarantees they can never reference the token, since it
     only ever appears as the ``github-token`` input.)

Usage:
    check_runner_token_policy.py [WORKFLOW_DIR] [--repo-root DIR]

Exit code 0 = compliant, 1 = one or more violations.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

TOKEN_NAME = "EC2_RUNNER_TOKEN"
TOKEN_REF = f"secrets.{TOKEN_NAME}"
ALLOWED_ACTION = "machulav/ec2-github-runner"

# The one and only allowed textual form, e.g.
#   github-token: ${{ secrets.EC2_RUNNER_TOKEN }}
ALLOWED_LINE = re.compile(r"^\s*github-token:\s*\$\{\{\s*secrets\." + re.escape(TOKEN_NAME) + r"\s*\}\}\s*$")

# Paths (relative to repo root) that are allowed to mention the token name at
# all. The workflows are where it is legitimately used; CI tooling under
# .github/scripts/ (this script and its tests) references it by name out of
# necessity. None of these can expose the token *value*: rules 2-3 guarantee the
# secret is only ever interpolated into the machulav action, so a file merely
# containing the name string is harmless.
ALLOWED_PATH_PREFIXES = (
    ".github/workflows/",
    ".github/scripts/",
)


def fail(violations: list[str], path: Path, job: str | None, message: str) -> None:
    loc = f"{path}" + (f" [job: {job}]" if job else "")
    violations.append(f"{loc}: {message}")


def iter_steps(job: dict):
    for step in job.get("steps") or []:
        if isinstance(step, dict):
            yield step


def check_workflow_file(path: Path, violations: list[str]) -> None:
    text = path.read_text()
    if TOKEN_REF not in text and TOKEN_NAME not in text:
        return

    # --- Rule 2: every line mentioning the token must be the exact input. -----
    for lineno, line in enumerate(text.splitlines(), start=1):
        if TOKEN_NAME not in line:
            continue
        if not ALLOWED_LINE.match(line):
            fail(
                violations,
                path,
                None,
                f"line {lineno}: '{TOKEN_NAME}' may only appear as "
                f"'github-token: ${{{{ secrets.{TOKEN_NAME} }}}}', got: {line.strip()!r}",
            )

    # --- Structural rules 3 & 4 via parsed YAML. ------------------------------
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        fail(violations, path, None, f"could not parse YAML: {exc}")
        return

    jobs = data.get("jobs") or {}
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        job_text = yaml.safe_dump(job)
        if TOKEN_NAME not in job_text:
            continue

        token_steps = []
        for step in iter_steps(job):
            step_text = yaml.safe_dump(step)
            if TOKEN_NAME not in step_text:
                continue
            token_steps.append(step)

            # Rule 3: the consuming step must be the EC2 runner action.
            uses = (step.get("uses") or "").split("@")[0]
            if uses != ALLOWED_ACTION:
                fail(
                    violations,
                    path,
                    job_name,
                    f"step '{step.get('name', uses or '<unnamed>')}' uses the token "
                    f"but is not '{ALLOWED_ACTION}' (uses: {uses or '<none>'})",
                )

            # Rule 2 (structural backstop): only via the github-token input.
            with_block = step.get("with") or {}
            offending = {
                k: v for k, v in with_block.items() if k != "github-token" and TOKEN_NAME in yaml.safe_dump({k: v})
            }
            if offending:
                fail(
                    violations,
                    path,
                    job_name,
                    f"token passed via disallowed input(s): {sorted(offending)}",
                )

        if not token_steps:
            # Token appears in the job but in no step (e.g. job-level env: or a
            # reusable-workflow `secrets:` block). That is never allowed.
            fail(
                violations,
                path,
                job_name,
                "token referenced at job level (env/secrets/with), not as an " f"'{ALLOWED_ACTION}' step input",
            )
            continue

        # Rule 4: a privileged job must not pull untrusted code into its
        # workspace (no local actions, no checkout) next to the token.
        for step in iter_steps(job):
            uses = step.get("uses") or ""
            bare = uses.split("@")[0]
            if uses.startswith("./") or uses.startswith("../"):
                fail(
                    violations,
                    path,
                    job_name,
                    f"job exposes the token and also runs a LOCAL action " f"(uses: {uses}); not allowed",
                )
            if bare == "actions/checkout":
                fail(
                    violations,
                    path,
                    job_name,
                    "job exposes the token and also runs actions/checkout; "
                    "the privileged token must not share a job with checked-out "
                    "code",
                )


def check_no_leaks_outside_workflows(repo_root: Path, violations: list[str], ref: str | None = None) -> None:
    """Rule 1: the token name must not appear anywhere except the workflows.

    When ``ref`` is given (e.g. a PR head SHA or ``FETCH_HEAD``), the search runs
    against that commit's *tree* instead of the working tree. This lets the
    Workflow Security gate enforce Rule 1 over the whole proposed PR snapshot --
    including files outside ``.github/workflows`` -- while the policy script
    itself still runs from the trusted base checkout. ``git grep`` only reads
    blobs, so scanning an untrusted ref executes nothing.
    """
    cmd = ["git", "-C", str(repo_root), "grep", "-l", "-I", "-F", TOKEN_NAME]
    if ref:
        cmd.append(ref)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        # git is required to verify confinement; if it is unavailable we cannot
        # run the check, so fail closed rather than silently passing.
        violations.append(
            f"<repo-wide leak check>: 'git' not found; cannot verify "
            f"'{TOKEN_NAME}' is confined to .github/workflows/"
        )
        return

    # `git grep` exits 0 when matches are found and 1 when there are none. Any
    # other code (e.g. 128 when repo_root is not a git worktree, or the ref is
    # missing) means the leak check could not run -- fail closed rather than
    # silently passing.
    if out.returncode not in (0, 1):
        violations.append(
            f"<repo-wide leak check>: 'git grep' failed (exit {out.returncode}) "
            f"in {repo_root}{f' for ref {ref}' if ref else ''}; cannot verify "
            f"'{TOKEN_NAME}' is confined to .github/workflows/. "
            f"stderr: {out.stderr.strip()!r}"
        )
        return

    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # With a ref, `git grep` prefixes each match with "<ref>:"; strip it to
        # get the repo-relative path.
        rel = line.split(":", 1)[1] if ref else line
        if any(rel.startswith(p) for p in ALLOWED_PATH_PREFIXES):
            continue
        violations.append(f"{rel}: '{TOKEN_NAME}' must not be referenced outside " f".github/workflows/")


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
            "\nThe admin-scoped runner token may ONLY be used as:\n"
            f"    github-token: ${{{{ secrets.{TOKEN_NAME} }}}}\n"
            f"  in a step that uses '{ALLOWED_ACTION}', inside a job that does\n"
            "  not check out code or run local actions. See "
            ".github/scripts/check_runner_token_policy.py.",
            file=sys.stderr,
        )
        return 1

    print(
        f"✅ EC2 runner token policy: OK ({len(files)} workflow file(s) scanned; "
        f"token used only as '{ALLOWED_ACTION}' input)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
