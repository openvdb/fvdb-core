# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Upload wheels to the fvdb-packages S3 simple index and rewrite the project's index.html.

Used by the publish-to-s3 jobs of fvdb-core (publish.yml, nightly-publish.yml)
and fvdb-reality-capture (publish.yml). fvdb-reality-capture keeps an identical
copy of this file; change both together. Tests: test_publish_simple_index.py
(fvdb-core).

The project index is the record of which wheels we published and their
SHA-256. pip refuses a download whose hash differs from the ``#sha256=``
fragment, so this script never takes a hash from a file already in S3:

- a wheel built in this run is hashed from the local file;
- every other wheel keeps the hash recorded in the current index;
- a wheel that is in S3 but neither built now nor in the index is an error.

So if a published wheel is replaced in S3, pip rejects the replacement instead
of a later publish run vouching for it, and a file slipped into the folder is
never listed.

Release wheels (prefix ``simple``) are write-once: a wheel already in the
index is not uploaded again if the local file is identical, and the run fails
if it differs. New release wheels are uploaded with ``If-None-Match: *`` so S3
refuses to overwrite an existing object; the bucket policy rejects release
wheel uploads without that header. Staging and nightly wheels may be
replaced (same version rebuilt), and their recorded hash is updated.

Every upload sends the SHA-256 so that S3 rejects a corrupted upload.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

WRITE_ONCE_PREFIXES = {"simple"}
INDEX_CACHE_CONTROL = "public, max-age=60, must-revalidate"
_ENTRY_RE = re.compile(r'href="([^"#]+)#sha256=([0-9a-f]{64})"')


class PublishError(Exception):
    pass


class AwsS3:
    """The few S3 operations this script needs, through the AWS CLI."""

    def __init__(self, bucket: str):
        self.bucket = bucket

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["aws", *args], capture_output=True, text=True)

    def get_text(self, key: str) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "obj"
            r = self._run("s3api", "get-object", "--bucket", self.bucket, "--key", key, str(out))
            if r.returncode != 0:
                if "NoSuchKey" in r.stderr:
                    return None
                raise PublishError(f"reading s3://{self.bucket}/{key} failed: {r.stderr.strip()}")
            return out.read_text()

    def list_keys(self, prefix: str) -> list[str]:
        r = self._run(
            "s3api", "list-objects-v2", "--bucket", self.bucket, "--prefix", prefix,
            "--query", "Contents[].Key", "--output", "json",
        )  # fmt: skip
        if r.returncode != 0:
            raise PublishError(f"listing s3://{self.bucket}/{prefix} failed: {r.stderr.strip()}")
        return json.loads(r.stdout or "null") or []

    def put(
        self,
        key: str,
        path: Path,
        sha256_hex: str,
        *,
        if_none_match: bool = False,
        content_type: str | None = None,
        cache_control: str | None = None,
    ) -> None:
        args = [
            "s3api", "put-object", "--bucket", self.bucket, "--key", key, "--body", str(path),
            "--checksum-sha256", base64.b64encode(bytes.fromhex(sha256_hex)).decode(),
        ]  # fmt: skip
        if if_none_match:
            args += ["--if-none-match", "*"]
        if content_type:
            args += ["--content-type", content_type]
        if cache_control:
            args += ["--cache-control", cache_control]
        r = self._run(*args)
        if r.returncode != 0:
            if if_none_match and "PreconditionFailed" in r.stderr:
                raise PublishError(f"s3://{self.bucket}/{key} already exists; release wheels are never replaced")
            raise PublishError(f"uploading s3://{self.bucket}/{key} failed: {r.stderr.strip()}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_index(text: str) -> dict[str, str]:
    """Map wheel file name -> sha256 from a simple-index project page."""
    entries: dict[str, str] = {}
    for href, sha in _ENTRY_RE.findall(text):
        name = unquote(href)
        if name.endswith(".whl"):
            entries[name] = sha
    return entries


def render_index(title: str, entries: dict[str, str]) -> str:
    lines = [
        "<!DOCTYPE html>",
        f'<html><head><meta charset="utf-8"><title>{html.escape(title)}</title></head><body>',
        f"<h1>{html.escape(title)}</h1>",
    ]
    for name in sorted(entries):
        lines.append(f'<a href="{quote(name)}#sha256={entries[name]}">{html.escape(name)}</a><br/>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def _wheels_in(s3, folder: str) -> set[str]:
    names = {k[len(folder) :] for k in s3.list_keys(folder)}
    return {n for n in names if n.endswith(".whl") and "/" not in n}


def publish(s3, prefix: str, project: str, title: str, dist: Path, log=print) -> dict[str, str]:
    folder = f"{prefix}/{project}/"
    write_once = prefix in WRITE_ONCE_PREFIXES

    local = {p.name: p for p in sorted(dist.glob("*.whl"))}
    if not local:
        raise PublishError(f"no wheels in {dist}")
    local_sha = {name: sha256_file(p) for name, p in local.items()}

    previous_text = s3.get_text(folder + "index.html")
    recorded = parse_index(previous_text) if previous_text is not None else {}
    log(f"{folder}: {len(recorded)} wheel(s) in the current index, {len(local)} built in this run")

    # Check before uploading anything, so a folder with a planted file stops
    # the run without changing it.
    unknown = sorted(_wheels_in(s3, folder) - set(recorded) - set(local))
    if unknown:
        raise PublishError(
            f"{len(unknown)} wheel(s) in s3://{s3.bucket}/{folder} were neither built in this run nor in the "
            f"current index, so their origin is unknown; refusing to publish: {unknown}"
        )

    for name, path in local.items():
        sha = local_sha[name]
        if write_once and name in recorded:
            if recorded[name] == sha:
                log(f"  {name}: already published with the same hash; not uploaded again")
                continue
            raise PublishError(
                f"{name} is already published with sha256 {recorded[name]}, and this build differs ({sha}); "
                "release wheels are never replaced. Bump the version instead."
            )
        s3.put(folder + name, path, sha, if_none_match=write_once)
        log(f"  {name}: uploaded (sha256 {sha[:16]}...)")

    in_s3 = _wheels_in(s3, folder)
    missing = sorted(set(local) - in_s3)
    if missing:
        raise PublishError(f"uploaded but not listed in S3: {missing}")

    entries: dict[str, str] = {}
    unknown = []
    for name in sorted(in_s3):
        if name in local:
            entries[name] = local_sha[name]
        elif name in recorded:
            entries[name] = recorded[name]
        else:
            unknown.append(name)
    if unknown:  # appeared while this run was uploading
        raise PublishError(
            f"{len(unknown)} wheel(s) in s3://{s3.bucket}/{folder} were neither built in this run nor in the "
            f"current index, so their origin is unknown; refusing to list them: {unknown}"
        )
    for name in sorted(set(recorded) - in_s3):
        log(f"  {name}: no longer in S3 (pruned); dropped from the index")

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index.html"
        index_path.write_text(render_index(title, entries))
        s3.put(
            folder + "index.html",
            index_path,
            sha256_file(index_path),
            content_type="text/html",
            cache_control=INDEX_CACHE_CONTROL,
        )
    log(f"{folder}index.html: {len(entries)} wheel(s)")
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True, help="simple, simple-staging or simple-nightly")
    ap.add_argument("--project", required=True, help="normalized project name, e.g. fvdb-core")
    ap.add_argument("--title", required=True)
    ap.add_argument("--dist", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        publish(AwsS3(args.bucket), args.prefix, args.project, args.title, args.dist)
    except PublishError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
