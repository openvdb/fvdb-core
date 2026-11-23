# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for verify_package_index.py, with HTTP replaced by a fake."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("verify_package_index", HERE / "verify_package_index.py")
vpi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpi)

OLD = "fvdb_core-0.5.1+pt211.cu130-cp312-cp312-linux_x86_64.whl"
NEW = "fvdb_core-0.7.0+pt214.cu130-cp312-cp312-linux_x86_64.whl"
RC = "fvdb_reality_capture-0.6.0-py3-none-any.whl"
A, B, C, D = "a" * 64, "b" * 64, "c" * 64, "d" * 64


def page(entries):
    return "".join(f'<a href="{n.replace("+", "%2B")}#sha256={s}">{n}</a><br/>' for n, s in entries.items())


def attestation(sha, repo="openvdb/fvdb-core", path=".github/workflows/publish.yml"):
    statement = {
        "subject": [{"name": "x.whl", "digest": {"sha256": sha}}],
        "predicate": {
            "buildDefinition": {
                "externalParameters": {"workflow": {"repository": f"https://github.com/{repo}", "path": path}}
            }
        },
    }
    return {"bundle": {"dsseEnvelope": {"payload": base64.b64encode(json.dumps(statement).encode()).decode()}}}


def fake_get(core, rc, attestations=None, root="fvdb-core fvdb-reality-capture"):
    attestations = attestations or {}

    def get(url, token):
        if url == f"{vpi.INDEX_URL}/":
            return 200, "".join(f'<a href="{p}/">{p}</a>' for p in root.split())
        if url == f"{vpi.INDEX_URL}/fvdb-core/":
            return 200, page(core)
        if url == f"{vpi.INDEX_URL}/fvdb-reality-capture/":
            return 200, page(rc)
        for sha, atts in attestations.items():
            if url.endswith(f"/attestations/sha256:{sha}"):
                return 200, json.dumps({"attestations": atts})
        return 404, "{}"

    return get


BASELINE = {"fvdb-core": {OLD: A}, "fvdb-reality-capture": {RC: B}}


def test_clean_index_passes():
    assert vpi.verify(BASELINE, fake_get({OLD: A}, {RC: B}), None) == []


def test_changed_baseline_hash_is_reported():
    problems = vpi.verify(BASELINE, fake_get({OLD: C}, {RC: B}), None)
    assert any("hash changed" in p for p in problems)


def test_removed_baseline_wheel_is_reported():
    problems = vpi.verify(BASELINE, fake_get({}, {RC: B}), None)
    assert any("no longer listed" in p for p in problems)


def test_new_wheel_with_publish_attestation_passes():
    get = fake_get({OLD: A, NEW: C}, {RC: B}, {C: [attestation(C)]})
    assert vpi.verify(BASELINE, get, "t") == []


def test_new_wheel_without_attestation_is_reported():
    problems = vpi.verify(BASELINE, fake_get({OLD: A, NEW: C}, {RC: B}), "t")
    assert any("no attestation" in p and NEW in p for p in problems)


def test_attestation_from_other_workflow_or_repo_is_rejected():
    for att in (attestation(C, path=".github/workflows/nightly-publish.yml"), attestation(C, repo="evil/fork")):
        problems = vpi.verify(BASELINE, fake_get({OLD: A, NEW: C}, {RC: B}, {C: [att]}), "t")
        assert any("no attestation" in p for p in problems)


def test_attestation_for_other_digest_is_rejected():
    problems = vpi.verify(BASELINE, fake_get({OLD: A, NEW: C}, {RC: B}, {C: [attestation(D)]}), "t")
    assert any("no attestation" in p for p in problems)


def test_unknown_project_in_root_index_is_reported():
    get = fake_get({OLD: A}, {RC: B}, root="fvdb-core fvdb-reality-capture evil-pkg")
    assert any("unknown project" in p for p in vpi.verify(BASELINE, get, None))


def test_snappy_literal_and_copy():
    # varint length 11; literal "hello " (tag (6-1)<<2); copy-1 of 5 bytes at offset 6
    data = bytes([11, (6 - 1) << 2]) + b"hello " + bytes([((5 - 4) << 2) | 1, 6])
    assert vpi.snappy_decompress(data) == b"hello hello"


def test_bundle_fetched_from_bundle_url():
    bundle = attestation(C)["bundle"]
    raw = json.dumps(bundle).encode()
    sn = bytes([len(raw) & 0x7F | 0x80, len(raw) >> 7]) if len(raw) >= 128 else bytes([len(raw)])
    # literal with a 2-byte length (tag 61 << 2)
    sn += bytes([61 << 2]) + (len(raw) - 1).to_bytes(2, "little") + raw
    atts = [{"bundle": None, "bundle_url": "https://blob.example/x.json.sn"}]

    def get(url, token):
        assert token is None  # never send the GitHub token to blob storage
        return 200, sn

    assert vpi.attested_by_publish_workflow(atts, C, "openvdb/fvdb-core", get)
