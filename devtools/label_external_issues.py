#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""One-off script to label existing open issues.

Adds ``external`` to issues not filed by fvdb-dev team members, and
``triage`` to external issues that have no team-member comment.

Requires the GitHub CLI (``gh``) authenticated with a token that has
``read:org`` scope (classic PAT) or ``Organization > Members > Read``
(fine-grained) and write access to issues.

Usage::

    python devtools/label_external_issues.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess

REPOS = [
    "openvdb/fvdb-core",
    "openvdb/fvdb-reality-capture",
]

TEAM_SLUG = "fvdb-dev"


def gh_api(endpoint: str) -> list[dict] | dict:
    result = subprocess.run(
        ["gh", "api", endpoint],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def gh_api_pages(endpoint: str):
    """Yield successive pages (lists) from a paginated GitHub API endpoint."""
    sep = "&" if "?" in endpoint else "?"
    page = 1
    while True:
        batch = gh_api(f"{endpoint}{sep}per_page=100&page={page}")
        if not isinstance(batch, list) or not batch:
            break
        yield batch
        if len(batch) < 100:
            break
        page += 1


def gh_api_paginated(endpoint: str) -> list[dict]:
    items: list[dict] = []
    for batch in gh_api_pages(endpoint):
        items.extend(batch)
    return items


def fetch_team_members(org: str, team_slug: str) -> set[str]:
    members = gh_api_paginated(f"orgs/{org}/teams/{team_slug}/members")
    if not members:
        raise RuntimeError(
            f"No members returned for {org}/{team_slug}. "
            "Check that the team exists and the token has read:org scope "
            "(classic PAT) or Organization > Members > Read permission "
            "(fine-grained PAT)."
        )
    return {m["login"] for m in members}


def add_label(repo: str, issue_number: int, label: str) -> None:
    subprocess.run(
        ["gh", "issue", "edit", str(issue_number), "--repo", repo, "--add-label", label],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be labeled without making changes")
    args = parser.parse_args()

    orgs = sorted({repo.split("/")[0] for repo in REPOS})
    team_members: set[str] = set()
    for org in orgs:
        print(f"Fetching members of {org}/{TEAM_SLUG} ...")
        team_members.update(fetch_team_members(org, TEAM_SLUG))
    print(f"  {len(team_members)} team member(s)\n")

    for repo in REPOS:
        print(f"Scanning {repo} ...")
        issues = [i for i in gh_api_paginated(f"repos/{repo}/issues?state=open") if "pull_request" not in i]
        ext_count = 0
        triage_count = 0

        for raw in issues:
            number = raw["number"]
            author = raw["user"]["login"]
            labels = {l["name"] for l in raw.get("labels", [])}

            if author in team_members:
                continue

            needs_external = "external" not in labels
            needs_triage = False

            if raw["comments"] == 0:
                needs_triage = "triage" not in labels
            elif "triage" not in labels:
                has_team_reply = any(
                    c["user"]["login"] in team_members
                    for page in gh_api_pages(f"repos/{repo}/issues/{number}/comments")
                    for c in page
                )
                needs_triage = not has_team_reply

            actions = []
            if needs_external:
                actions.append("external")
                ext_count += 1
            if needs_triage:
                actions.append("triage")
                triage_count += 1

            if not actions:
                continue

            label_str = ", ".join(actions)
            if args.dry_run:
                print(f"  [dry-run] #{number} @{author} -- would add: {label_str}")
            else:
                for label in actions:
                    add_label(repo, number, label)
                print(f"  #{number} @{author} -- added: {label_str}")

        print(f"  +external: {ext_count}, +triage: {triage_count}\n")


if __name__ == "__main__":
    main()
