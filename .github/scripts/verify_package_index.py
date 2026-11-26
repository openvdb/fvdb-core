# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Check the public release package index against hashes we know we published.

Run daily by .github/workflows/verify-package-index.yml. pip checks every wheel
it downloads against the ``#sha256=`` hash in the index, so the index is what
has to be right. For each project in the release index (prefix ``simple``):

- every wheel listed in .github/package-index-baseline.json (the release index
  as of 2026-11-16, before wheels were attested) must still be listed with the
  same hash;
- every other wheel must have a build provenance attestation in the project's
  repository, made by that repository's publish workflow, for exactly that
  hash.

The root index must list only the known projects.

Nothing is downloaded except the index pages and attestation metadata.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote

INDEX_URL = "https://d36m13axqqhiit.cloudfront.net/simple"
PROJECTS = {
    "fvdb-core": "openvdb/fvdb-core",
    "fvdb-reality-capture": "openvdb/fvdb-reality-capture",
}
PUBLISH_WORKFLOW = ".github/workflows/publish.yml"
_ENTRY_RE = re.compile(r'href="([^"#]+)#sha256=([0-9a-f]{64})"')
_HREF_RE = re.compile(r'href="([^"]+)"')


def http_get(url: str, token: str | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "fvdb-verify-package-index"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _text(body: bytes | str) -> str:
    return body.decode(errors="replace") if isinstance(body, bytes) else body


def snappy_decompress(data: bytes) -> bytes:
    """Decode a raw (unframed) Snappy block, the format of GitHub's attestation bundles."""
    length, shift, i = 0, 0, 0
    while True:
        b = data[i]
        i += 1
        length |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            break
    out = bytearray()
    while i < len(data):
        tag = data[i]
        i += 1
        kind = tag & 3
        if kind == 0:  # literal
            n = tag >> 2
            if n < 60:
                n += 1
            else:
                extra = n - 59
                n = int.from_bytes(data[i : i + extra], "little") + 1
                i += extra
            out += data[i : i + n]
            i += n
            continue
        if kind == 1:
            n = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | data[i]
            i += 1
        elif kind == 2:
            n = (tag >> 2) + 1
            offset = int.from_bytes(data[i : i + 2], "little")
            i += 2
        else:
            n = (tag >> 2) + 1
            offset = int.from_bytes(data[i : i + 4], "little")
            i += 4
        if offset == 0 or offset > len(out):
            raise ValueError("bad snappy copy offset")
        for _ in range(n):
            out.append(out[-offset])
    if len(out) != length:
        raise ValueError("snappy length mismatch")
    return bytes(out)


def parse_project_index(text: str) -> dict[str, str]:
    return {unquote(h): s for h, s in _ENTRY_RE.findall(text) if unquote(h).endswith(".whl")}


def parse_root_index(text: str) -> set[str]:
    return {h.rstrip("/") for h in _HREF_RE.findall(text)}


def attested_by_publish_workflow(attestations: list[dict], sha: str, repo: str, get=None) -> bool:
    """True if one attestation covers ``sha`` and was made by ``repo``'s publish workflow.

    The attestations API returns each Sigstore bundle either inline or, now
    usually, as a pre-signed ``bundle_url`` to a Snappy-compressed file,
    fetched without our token.
    """
    for att in attestations:
        try:
            bundle = att.get("bundle")
            if bundle is None and att.get("bundle_url") and get is not None:
                status, data = get(att["bundle_url"], None)
                if status != 200:
                    continue
                bundle = json.loads(snappy_decompress(data))
            payload = bundle["dsseEnvelope"]["payload"]
            statement = json.loads(base64.b64decode(payload))
            subjects = statement.get("subject", [])
            workflow = statement["predicate"]["buildDefinition"]["externalParameters"]["workflow"]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not any(s.get("digest", {}).get("sha256") == sha for s in subjects):
            continue
        if workflow.get("repository") == f"https://github.com/{repo}" and workflow.get("path") == PUBLISH_WORKFLOW:
            return True
    return False


def verify(baseline: dict[str, dict[str, str]], get, token: str | None) -> list[str]:
    problems: list[str] = []

    status, root = get(f"{INDEX_URL}/", None)
    if status != 200:
        return [f"root index: HTTP {status}"]
    unexpected = parse_root_index(_text(root)) - set(PROJECTS)
    if unexpected:
        problems.append(f"root index lists unknown project(s): {sorted(unexpected)}")

    for project, repo in PROJECTS.items():
        status, text = get(f"{INDEX_URL}/{project}/", None)
        if status != 200:
            problems.append(f"{project}: index HTTP {status}")
            continue
        listed = parse_project_index(_text(text))
        known = baseline.get(project, {})

        for name, sha in known.items():
            if name not in listed:
                problems.append(f"{project}: {name} is no longer listed")
            elif listed[name] != sha:
                problems.append(f"{project}: {name} hash changed from {sha} to {listed[name]}")

        for name, sha in sorted(listed.items()):
            if name in known:
                continue
            status, body = get(f"https://api.github.com/repos/{repo}/attestations/sha256:{sha}", token)
            atts = json.loads(_text(body)).get("attestations", []) if status == 200 else []
            if not attested_by_publish_workflow(atts, sha, repo, get):
                problems.append(f"{project}: {name} (sha256 {sha}) has no attestation from {repo}'s publish workflow")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--baseline", type=Path, default=Path(__file__).resolve().parents[1] / "package-index-baseline.json"
    )
    args = ap.parse_args(argv)
    baseline = json.loads(args.baseline.read_text())
    problems = verify(baseline, http_get, os.environ.get("GH_TOKEN"))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = ["### Release package index"] + ([f"- {p}" for p in problems] or ["- All wheels verified."])
    if summary:
        Path(summary).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for p in problems:
        print(f"::error::{p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
