# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for publish_simple_index.py, with S3 replaced by an in-memory fake."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("publish_simple_index", HERE / "publish_simple_index.py")
psi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psi)

PROJECT = "fvdb-core"


class FakeS3:
    bucket = "fvdb-packages"

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.puts: list[tuple[str, bool]] = []

    def get_text(self, key):
        data = self.objects.get(key)
        return None if data is None else data.decode()

    def list_keys(self, prefix):
        return sorted(k for k in self.objects if k.startswith(prefix))

    def put(self, key, path, sha256_hex, *, if_none_match=False, content_type=None, cache_control=None):
        data = Path(path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256_hex
        if if_none_match and key in self.objects:
            raise psi.PublishError(f"{key} already exists")
        self.objects[key] = data
        self.puts.append((key, if_none_match))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wheel(name="fvdb_core-0.6.0+pt213.cu130-cp312-cp312-linux_x86_64.whl"):
    return name


def index_for(prefix, entries):
    return psi.render_index("Links for fvdb_core", entries).encode()


def make_dist(tmp_path, files: dict[str, bytes]) -> Path:
    d = tmp_path / "dist"
    d.mkdir()
    for n, data in files.items():
        (d / n).write_bytes(data)
    return d


def run(s3, prefix, dist):
    return psi.publish(s3, prefix, PROJECT, "Links for fvdb_core", dist, log=lambda *_: None)


def test_new_release_wheel_is_uploaded_write_once_and_indexed(tmp_path):
    old = wheel("fvdb_core-0.5.1+pt211.cu130-cp312-cp312-linux_x86_64.whl")
    new = wheel()
    s3 = FakeS3(
        {
            f"simple/{PROJECT}/{old}": b"old",
            f"simple/{PROJECT}/index.html": index_for("simple", {old: sha(b"old")}),
        }
    )
    entries = run(s3, "simple", make_dist(tmp_path, {new: b"new"}))
    assert entries == {old: sha(b"old"), new: sha(b"new")}
    assert (f"simple/{PROJECT}/{new}", True) in s3.puts
    assert psi.parse_index(s3.get_text(f"simple/{PROJECT}/index.html")) == entries


def test_swapped_release_wheel_keeps_its_original_hash(tmp_path):
    """A wheel replaced in S3 must not get a new hash in the index."""
    old = wheel("fvdb_core-0.5.1+pt211.cu130-cp312-cp312-linux_x86_64.whl")
    s3 = FakeS3(
        {
            f"simple/{PROJECT}/{old}": b"MALICIOUS",
            f"simple/{PROJECT}/index.html": index_for("simple", {old: sha(b"old")}),
        }
    )
    entries = run(s3, "simple", make_dist(tmp_path, {wheel(): b"new"}))
    assert entries[old] == sha(b"old")


def test_unknown_wheel_in_s3_is_refused(tmp_path):
    planted = wheel("fvdb_core-9.9.9+pt213.cu130-cp312-cp312-linux_x86_64.whl")
    s3 = FakeS3(
        {
            f"simple-nightly/{PROJECT}/{planted}": b"planted",
            f"simple-nightly/{PROJECT}/index.html": index_for("simple-nightly", {}),
        }
    )
    with pytest.raises(psi.PublishError, match="origin is unknown"):
        run(s3, "simple-nightly", make_dist(tmp_path, {wheel(): b"new"}))
    assert s3.puts == []  # stopped before uploading or rewriting anything


def test_release_rerun_with_identical_wheel_skips_upload(tmp_path):
    w = wheel()
    s3 = FakeS3(
        {f"simple/{PROJECT}/{w}": b"same", f"simple/{PROJECT}/index.html": index_for("simple", {w: sha(b"same")})}
    )
    entries = run(s3, "simple", make_dist(tmp_path, {w: b"same"}))
    assert entries == {w: sha(b"same")}
    assert all(k.endswith("index.html") for k, _ in s3.puts)


def test_release_rebuild_with_different_bytes_fails(tmp_path):
    w = wheel()
    s3 = FakeS3({f"simple/{PROJECT}/{w}": b"v1", f"simple/{PROJECT}/index.html": index_for("simple", {w: sha(b"v1")})})
    with pytest.raises(psi.PublishError, match="never replaced"):
        run(s3, "simple", make_dist(tmp_path, {w: b"v2"}))
    assert s3.objects[f"simple/{PROJECT}/{w}"] == b"v1"


def test_staging_rebuild_replaces_wheel_and_hash(tmp_path):
    w = wheel()
    s3 = FakeS3(
        {
            f"simple-staging/{PROJECT}/{w}": b"v1",
            f"simple-staging/{PROJECT}/index.html": index_for("simple-staging", {w: sha(b"v1")}),
        }
    )
    entries = run(s3, "simple-staging", make_dist(tmp_path, {w: b"v2"}))
    assert entries == {w: sha(b"v2")}
    assert (f"simple-staging/{PROJECT}/{w}", False) in s3.puts


def test_pruned_wheel_is_dropped(tmp_path):
    gone = wheel("fvdb_core-0.6.0.dev20261001+pt213.cu130-cp312-cp312-linux_x86_64.whl")
    s3 = FakeS3({f"simple-nightly/{PROJECT}/index.html": index_for("simple-nightly", {gone: sha(b"x")})})
    entries = run(s3, "simple-nightly", make_dist(tmp_path, {wheel(): b"new"}))
    assert gone not in entries


def test_first_publish_without_index(tmp_path):
    s3 = FakeS3()
    entries = run(s3, "simple", make_dist(tmp_path, {wheel(): b"new"}))
    assert entries == {wheel(): sha(b"new")}


def test_index_round_trip_encodes_plus():
    w = wheel()
    text = psi.render_index("Links for fvdb_core", {w: "a" * 64})
    assert "fvdb_core-0.6.0%2Bpt213" in text
    assert psi.parse_index(text) == {w: "a" * 64}


def test_parses_existing_unencoded_index():
    """Nightly indexes written before this script put a literal '+' in the href."""
    w = wheel()
    text = f'<a href="{w}#sha256={"b" * 64}">{w}</a><br/>'
    assert psi.parse_index(text) == {w: "b" * 64}
