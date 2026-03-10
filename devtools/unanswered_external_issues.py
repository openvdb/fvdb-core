#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Report open issues from external users that lack a team-member response.

Requires the GitHub CLI (``gh``) to be installed and authenticated.
The token should have ``read:org`` scope (classic PAT) or
``Organization > Members > Read`` (fine-grained) so that team members
can be reliably identified.  If that scope is missing, the script falls
back to the ``author_association`` field, which may be unreliable with
tokens that lack org-level visibility (e.g. the default GITHUB_TOKEN).

Usage::

    python devtools/unanswered_external_issues.py [--format terminal|slack]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

REPOS = [
    "openvdb/fvdb-core",
    "openvdb/fvdb-reality-capture",
]

TEAM_SLUG = "fvdb-dev"

INSIDER_ASSOCIATIONS = {"MEMBER", "COLLABORATOR", "OWNER"}

# ---------------------------------------------------------------------------
# Terminal colours and clickable hyperlinks (OSC 8)
# ---------------------------------------------------------------------------

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _sgr(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""


BOLD = _sgr("1")
DIM = _sgr("2")
RED = _sgr("31")
GREEN = _sgr("32")
YELLOW = _sgr("33")
CYAN = _sgr("36")
MAGENTA = _sgr("35")
RESET = _sgr("0")


def hyperlink(url: str, text: str | None = None) -> str:
    """Return an OSC 8 clickable hyperlink if the terminal supports it."""
    text = text or url
    if not _USE_COLOR:
        return text
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


@dataclass
class Issue:
    number: int
    title: str
    author: str
    author_association: str
    created_at: str
    url: str
    comment_count: int
    last_commenter: str | None = None
    last_commenter_association: str | None = None


@dataclass
class RepoReport:
    repo: str
    no_response: list[Issue] = field(default_factory=list)
    awaiting_response: list[Issue] = field(default_factory=list)


def gh_api(endpoint: str) -> list[dict] | dict:
    """Call ``gh api`` and return parsed JSON (single page)."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def gh_api_paginated(endpoint: str) -> list[dict]:
    """Call ``gh api`` with manual pagination and return all items."""
    all_items: list[dict] = []
    sep = "&" if "?" in endpoint else "?"
    page = 1
    while True:
        batch = gh_api(f"{endpoint}{sep}per_page=100&page={page}")
        if not isinstance(batch, list) or not batch:
            break
        all_items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return all_items


def fetch_open_issues(repo: str) -> list[dict]:
    """Return all open issues (not pull requests) for *repo*."""
    return [i for i in gh_api_paginated(f"repos/{repo}/issues?state=open") if "pull_request" not in i]


def fetch_comments(repo: str, issue_number: int) -> list[dict]:
    """Return all comments on a single issue."""
    return gh_api_paginated(f"repos/{repo}/issues/{issue_number}/comments")


def fetch_team_members(org: str, team_slug: str) -> set[str]:
    """Return login names of all members of *org*/*team_slug*.

    Raises if the API call fails so that a misconfigured token does not
    silently produce a wrong report.  The token needs ``read:org`` scope
    (classic PAT) or ``Organization > Members > Read`` (fine-grained).
    """
    members = gh_api_paginated(f"orgs/{org}/teams/{team_slug}/members")
    if not members:
        raise RuntimeError(
            f"No members returned for {org}/{team_slug}. "
            "Check that the team exists and the token has read:org scope."
        )
    return {m["login"] for m in members}


def is_insider(association: str) -> bool:
    return association in INSIDER_ASSOCIATIONS


def analyse_repo(repo: str, org_members: set[str]) -> RepoReport:
    report = RepoReport(repo=repo)
    issues = fetch_open_issues(repo)

    def _is_team(login: str, association: str) -> bool:
        return login in org_members or is_insider(association)

    external_issues = [i for i in issues if not _is_team(i["user"]["login"], i.get("author_association", ""))]

    total = len(external_issues)
    for idx, raw in enumerate(external_issues, 1):
        number = raw["number"]
        issue = Issue(
            number=number,
            title=raw["title"],
            author=raw["user"]["login"],
            author_association=raw.get("author_association", "NONE"),
            created_at=raw["created_at"][:10],
            url=raw["html_url"],
            comment_count=raw["comments"],
        )

        if issue.comment_count == 0:
            report.no_response.append(issue)
            print(f"  [{idx}/{total}] #{number} -- no comments", file=sys.stderr)
            continue

        print(f"  [{idx}/{total}] #{number} -- fetching {issue.comment_count} comment(s)", file=sys.stderr)
        comments = fetch_comments(repo, number)

        has_insider_reply = any(_is_team(c["user"]["login"], c.get("author_association", "")) for c in comments)
        if not has_insider_reply:
            report.no_response.append(issue)
            continue

        if comments:
            last = comments[-1]
            issue.last_commenter = last["user"]["login"]
            issue.last_commenter_association = last.get("author_association", "NONE")
            if not _is_team(issue.last_commenter, issue.last_commenter_association):
                report.awaiting_response.append(issue)

    return report


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------


def _format_issue_line(issue: Issue, suffix: str = "") -> str:
    num = hyperlink(issue.url, f"#{issue.number}")
    parts = [
        f"    {CYAN}{num:<6}{RESET}",
        f" {DIM}{issue.created_at}{RESET}",
        f"  {MAGENTA}@{issue.author:<20}{RESET}",
        f" {issue.title}",
    ]
    if suffix:
        parts.append(f"  {DIM}{suffix}{RESET}")
    if not _USE_COLOR:
        parts.append(f"\n           {issue.url}")
    return "".join(parts)


def print_report_terminal(reports: list[RepoReport]) -> None:
    for report in reports:
        total = len(report.no_response) + len(report.awaiting_response)
        color = RED if total else GREEN
        print(f"\n{BOLD}{'=' * 72}{RESET}")
        print(f"  {BOLD}{report.repo}{RESET}  --  {color}{total} issue(s) needing attention{RESET}")
        print(f"{BOLD}{'=' * 72}{RESET}")

        if report.no_response:
            print(f"\n  {RED}{BOLD}No team response ({len(report.no_response)}):{RESET}\n")
            for issue in sorted(report.no_response, key=lambda i: i.created_at):
                print(_format_issue_line(issue))
        else:
            print(f"\n  {GREEN}No team response: (none){RESET}")

        if report.awaiting_response:
            print(f"\n  {YELLOW}{BOLD}Awaiting follow-up ({len(report.awaiting_response)}):{RESET}\n")
            for issue in sorted(report.awaiting_response, key=lambda i: i.created_at):
                print(_format_issue_line(issue, suffix=f"[last: @{issue.last_commenter}]"))
        else:
            print(f"\n  {GREEN}Awaiting follow-up: (none){RESET}")

    print()


# ---------------------------------------------------------------------------
# Slack mrkdwn formatter
# ---------------------------------------------------------------------------


def _slack_issue_line(issue: Issue, suffix: str = "") -> str:
    line = f"\u2022 <{issue.url}|#{issue.number}> {issue.created_at}  @{issue.author} -- {issue.title}"
    if suffix:
        line += f"  _{suffix}_"
    return line


def print_report_slack(reports: list[RepoReport]) -> None:
    lines: list[str] = []
    for report in reports:
        total = len(report.no_response) + len(report.awaiting_response)
        lines.append(f"*{report.repo} -- {total} issue(s) needing attention*")

        if report.no_response:
            lines.append(f"\n:red_circle: *No team response ({len(report.no_response)}):*")
            for issue in sorted(report.no_response, key=lambda i: i.created_at):
                lines.append(_slack_issue_line(issue))
        else:
            lines.append("\n:white_check_mark: No team response: (none)")

        if report.awaiting_response:
            lines.append(f"\n:large_yellow_circle: *Awaiting follow-up ({len(report.awaiting_response)}):*")
            for issue in sorted(report.awaiting_response, key=lambda i: i.created_at):
                lines.append(_slack_issue_line(issue, suffix=f"last: @{issue.last_commenter}"))
        else:
            lines.append("\n:white_check_mark: Awaiting follow-up: (none)")

        lines.append("")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FORMATTERS = {
    "terminal": print_report_terminal,
    "slack": print_report_slack,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=FORMATTERS,
        default="terminal",
        dest="fmt",
        help="Output format (default: terminal)",
    )
    args = parser.parse_args()

    orgs = sorted({repo.split("/")[0] for repo in REPOS})
    team_members: set[str] = set()
    for org in orgs:
        print(f"\nFetching members of {org}/{TEAM_SLUG} ...", file=sys.stderr)
        team_members.update(fetch_team_members(org, TEAM_SLUG))
    if team_members:
        print(f"  {len(team_members)} team member(s) will be treated as insiders", file=sys.stderr)

    reports: list[RepoReport] = []
    for repo in REPOS:
        print(f"\nScanning {repo} ...", file=sys.stderr)
        reports.append(analyse_repo(repo, team_members))
    FORMATTERS[args.fmt](reports)


if __name__ == "__main__":
    main()
