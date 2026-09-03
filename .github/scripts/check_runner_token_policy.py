#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Repo-specific CI policy for the EC2 runner admin token.

This script enforces, by inspecting the workflow files themselves:

  1. The token name may appear ONLY inside ``.github/workflows/`` and the CI
     tooling under ``.github/scripts/`` (this enforcement script and its tests).
     It must not leak into product source, docs, etc.

  2. Every textual occurrence in a workflow must be one of exactly three forms
     (whitespace inside ``${{ }}`` is tolerated):

         github-token: ${{ secrets.EC2_RUNNER_TOKEN }}   # the action input
         EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}   # caller forwarding
         EC2_RUNNER_TOKEN:                               # callee declaration

     Each is a bare interpolation that is the *whole* value of a scalar key, so
     this rule alone forbids the token in a ``run:`` script or in any
     concatenated string. Which of the three a line is allowed to be is then
     settled structurally by rules 3 and 4 -- e.g. an ``env:`` entry happens to
     have the same shape as the forwarding form, and is rejected below.

     The forwarded name is required to be ``EC2_RUNNER_TOKEN`` itself, so the
     token keeps one name across caller and callee and a single
     ``git grep EC2_RUNNER_TOKEN`` still enumerates every file that touches it.

  3. The token may be consumed in exactly two ways:

     a. by a *step* whose ``uses:`` is ``machulav/ec2-github-runner``, via its
        ``github-token`` input; or
     b. by a *job* whose ``uses:`` is a local reusable workflow
        (``./.github/workflows/<name>.yml``), via a ``secrets:`` mapping.

     (b) is safe precisely because the callee is itself a file in
     ``.github/workflows/``: this same scan covers it, so rules 2-4 apply to it
     directly, and under ``pull_request_target`` it is base-controlled like
     every other workflow. Forwarding to anything else -- a remote reusable
     workflow, or a callee that is not present in this directory -- would send
     the token somewhere the scan cannot see, and is rejected.

  4. A job that references the token *in a step* must not pull untrusted code
     into its workspace alongside the privileged context: no local actions
     (``uses: ./...``) and no ``actions/checkout``. (Sibling ``run:`` steps are
     fine -- rule 2 guarantees they can never reference the token.) A
     reusable-workflow caller job under 3(b) has no ``steps:`` and no workspace
     at all, so there is nothing for this rule to bite on.

  5. No dynamic secret access (``secrets[...]``) anywhere in a workflow. Rules
     1-4 are textual: they key on the literal name ``EC2_RUNNER_TOKEN``. Dynamic
     indexing such as ``secrets[format('EC2_RUNNER_%s', 'TOKEN')]`` or
     ``secrets[matrix.name]`` could resolve the admin token *without* spelling
     its name, evading every other rule. It is never used legitimately here, so
     reject it outright (this check runs on every workflow, even ones that never
     mention the token by name).

  6. No ``secrets: inherit`` anywhere in a workflow. It forwards *every* secret,
     including the admin token, without ever spelling a name -- so it is
     invisible to rules 1-3 and would let a caller hand the token to a remote
     reusable workflow with this scan reporting OK. Forward the one secret
     explicitly instead (rule 3b), which is both narrower and checkable.

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

# Rule 2's three allowed textual forms. Each is a bare interpolation that is the
# whole value of a scalar key, so none of them can hide inside a `run:` script or
# a concatenated string.
_INTERP = r"\$\{\{\s*secrets\." + re.escape(TOKEN_NAME) + r"\s*\}\}"
#   github-token: ${{ secrets.EC2_RUNNER_TOKEN }}      -- the action input
ACTION_INPUT_LINE = re.compile(r"^\s*github-token:\s*" + _INTERP + r"\s*$")
#   EC2_RUNNER_TOKEN: ${{ secrets.EC2_RUNNER_TOKEN }}  -- caller forwarding
FORWARD_LINE = re.compile(r"^\s*" + re.escape(TOKEN_NAME) + r":\s*" + _INTERP + r"\s*$")
#   EC2_RUNNER_TOKEN:                                  -- callee declaration
DECLARE_LINE = re.compile(r"^\s*" + re.escape(TOKEN_NAME) + r":\s*$")
ALLOWED_LINES = (ACTION_INPUT_LINE, FORWARD_LINE, DECLARE_LINE)

# The only `uses:` a job may have if it is forwarded the token (rule 3b): a
# reusable workflow in this repo's own .github/workflows, which this same scan
# therefore covers. The capture group is the callee's file name, checked to exist.
LOCAL_REUSABLE = re.compile(r"^\./\.github/workflows/([^/]+\.ya?ml)$")

# Rule 6: blanket secret forwarding, which names nothing and so evades rules 1-3.
INHERIT_LINE = re.compile(r"^\s*secrets:\s*inherit\s*$")

# Dynamic secret access, e.g. secrets['X'], secrets[matrix.y], secrets[format(...)].
# The \b avoids matching identifiers that merely end in "secrets" (e.g. mysecrets[0]).
DYNAMIC_SECRET = re.compile(r"\bsecrets\s*\[")

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
# Human-readable form of the allowed locations, for violation/error messages.
ALLOWED_PATHS_DESC = " or ".join(ALLOWED_PATH_PREFIXES)


def fail(violations: list[str], path: Path, job: str | None, message: str) -> None:
    loc = f"{path}" + (f" [job: {job}]" if job else "")
    violations.append(f"{loc}: {message}")


def iter_steps(job: dict):
    for step in job.get("steps") or []:
        if isinstance(step, dict):
            yield step


def _on_block(data: dict) -> dict:
    """The workflow's trigger mapping.

    YAML 1.1 parses a bare ``on:`` key as the boolean ``True``, so accept both
    spellings rather than depending on how the file happens to quote it.
    """
    for key in ("on", True):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _token_secret_declaration(data: dict) -> dict | None:
    """The ``on.workflow_call.secrets`` mapping, if it declares the token.

    This is the callee side of rule 3b -- the one legitimate mention of the
    token name outside a job. Returns the enclosing mapping so the caller can
    remove the declaration before scanning what is left.
    """
    call = _on_block(data).get("workflow_call")
    if not isinstance(call, dict):
        return None
    secrets = call.get("secrets")
    if isinstance(secrets, dict) and TOKEN_NAME in secrets:
        return secrets
    return None


def _mentions_token(value) -> bool:
    return TOKEN_NAME in yaml.safe_dump(value)


def check_workflow_file(path: Path, violations: list[str]) -> None:
    text = path.read_text()

    # --- Rules 5 & 6: blanket / unnamed secret access (every workflow). -------
    # These must run before the token-name early return below: the whole point
    # of both is that they can reach the token without ever naming it.
    for lineno, line in enumerate(text.splitlines(), start=1):
        if DYNAMIC_SECRET.search(line):
            fail(
                violations,
                path,
                None,
                f"line {lineno}: dynamic secret access is not allowed -- use "
                f"'secrets.<NAME>', not 'secrets[...]' (it can resolve "
                f"'{TOKEN_NAME}' without naming it): {line.strip()!r}",
            )
        if INHERIT_LINE.match(line):
            fail(
                violations,
                path,
                None,
                f"line {lineno}: 'secrets: inherit' is not allowed -- it forwards "
                f"every secret, including '{TOKEN_NAME}', without naming one, so no "
                f"textual check can see where the token goes. Forward the single "
                f"secret explicitly: '{TOKEN_NAME}: ${{{{ secrets.{TOKEN_NAME} }}}}'.",
            )

    if TOKEN_REF not in text and TOKEN_NAME not in text:
        return

    # --- Rule 2: every line naming the token must be one of the three forms. --
    for lineno, line in enumerate(text.splitlines(), start=1):
        if TOKEN_NAME not in line:
            continue
        if not any(rx.match(line) for rx in ALLOWED_LINES):
            fail(
                violations,
                path,
                None,
                f"line {lineno}: '{TOKEN_NAME}' may only appear as "
                f"'github-token: ${{{{ secrets.{TOKEN_NAME} }}}}' (action input), "
                f"'{TOKEN_NAME}: ${{{{ secrets.{TOKEN_NAME} }}}}' (forward to a local "
                f"reusable workflow), or '{TOKEN_NAME}:' (workflow_call declaration), "
                f"got: {line.strip()!r}",
            )

    # --- Structural rules 3 & 4 via parsed YAML. ------------------------------
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        fail(violations, path, None, f"could not parse YAML: {exc}")
        return

    # Outside of `jobs:`, the only legitimate mention is the workflow_call
    # declaration; drop it and anything left over is a leak (workflow-level
    # `env:`, a default, a trigger expression, ...).
    declaration = _token_secret_declaration(data)
    if declaration is not None:
        declaration.pop(TOKEN_NAME)
    outside_jobs = {k: v for k, v in data.items() if k != "jobs"}
    if _mentions_token(outside_jobs):
        fail(
            violations,
            path,
            None,
            "token referenced outside any job; the only legitimate mention there "
            "is an 'on.workflow_call.secrets' declaration",
        )

    jobs = data.get("jobs") or {}
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if not _mentions_token(job):
            continue

        # --- Rule 3b: forwarding to a local reusable workflow. ---------------
        secrets_block = job.get("secrets")
        forwarded = {}
        if isinstance(secrets_block, dict):
            forwarded = {k: v for k, v in secrets_block.items() if _mentions_token({k: v})}

        if forwarded:
            renamed = sorted(k for k in forwarded if k != TOKEN_NAME)
            if renamed:
                fail(
                    violations,
                    path,
                    job_name,
                    f"token forwarded under a different name {renamed}; it must be "
                    f"forwarded as '{TOKEN_NAME}' so that a single "
                    f"'git grep {TOKEN_NAME}' still finds every file that touches it",
                )
            uses = job.get("uses") or ""
            callee = LOCAL_REUSABLE.match(uses)
            if callee is None:
                fail(
                    violations,
                    path,
                    job_name,
                    f"token forwarded via 'secrets:' to {uses or '<no job-level uses:>'!r}; "
                    f"it may only be forwarded to a local reusable workflow "
                    f"('./.github/workflows/<name>.yml'), which this same scan covers",
                )
            elif not (path.parent / callee.group(1)).is_file():
                fail(
                    violations,
                    path,
                    job_name,
                    f"token forwarded to {uses!r}, which does not exist in "
                    f"{path.parent}; the callee must be a workflow this scan covers",
                )

        # --- Rule 3a: only the EC2 runner action may consume it in a step. ----
        token_steps = []
        for step in iter_steps(job):
            if not _mentions_token(step):
                continue
            token_steps.append(step)

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
            offending = {k: v for k, v in with_block.items() if k != "github-token" and _mentions_token({k: v})}
            if offending:
                fail(
                    violations,
                    path,
                    job_name,
                    f"token passed via disallowed input(s): {sorted(offending)}",
                )

        # Anything left -- job-level `env:`, a `with:` input, a `if:` expression
        # -- is a mention in a position neither rule 3a nor 3b sanctions.
        leftover = {k: v for k, v in job.items() if k not in ("steps", "secrets")}
        if _mentions_token(leftover):
            fail(
                violations,
                path,
                job_name,
                f"token referenced at job level (env/with/...), not as an "
                f"'{ALLOWED_ACTION}' step input or a 'secrets:' forward to a local "
                f"reusable workflow",
            )
        elif not token_steps and not forwarded:
            # The job names the token but in none of the shapes above (e.g. a
            # `secrets:` block that is not a mapping). Fail closed.
            fail(
                violations,
                path,
                job_name,
                "token referenced in this job in an unrecognised position",
            )

        # --- Rule 4: no untrusted code in a job that touches the token. -------
        # A rule 3b caller job has no steps, so this only ever applies to 3a.
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
            f"'{TOKEN_NAME}' is confined to {ALLOWED_PATHS_DESC}"
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
            f"'{TOKEN_NAME}' is confined to {ALLOWED_PATHS_DESC}. "
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
        violations.append(f"{rel}: '{TOKEN_NAME}' must not be referenced outside {ALLOWED_PATHS_DESC}")


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
            "\nThe admin-scoped runner token may ONLY be used in one of two ways:\n"
            f"    github-token: ${{{{ secrets.{TOKEN_NAME} }}}}\n"
            f"  in a step that uses '{ALLOWED_ACTION}', inside a job that does\n"
            "  not check out code or run local actions; or\n"
            f"    secrets:\n      {TOKEN_NAME}: ${{{{ secrets.{TOKEN_NAME} }}}}\n"
            "  on a job whose 'uses:' is a local './.github/workflows/<name>.yml'\n"
            "  (which this same scan covers). 'secrets: inherit' is never allowed.\n"
            "  See .github/scripts/check_runner_token_policy.py.",
            file=sys.stderr,
        )
        return 1

    print(
        f"✅ EC2 runner token policy: OK ({len(files)} workflow file(s) scanned; "
        f"token reaches only '{ALLOWED_ACTION}', directly or via a local "
        f"reusable workflow)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
