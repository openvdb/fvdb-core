#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Build documentation from an old release tag using autodoc mocking
# (no CUDA compilation required).
#
# Usage: build-versioned-docs.sh <tag> <output-dir>
# Example: build-versioned-docs.sh v0.3.0 _docs_build/0.3

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

TAG="${1:-}"
OUTPUT="${2:-}"

[[ -n "$TAG" ]] || die "usage: build-versioned-docs.sh <tag> <output-dir>"
[[ -n "$OUTPUT" ]] || die "usage: build-versioned-docs.sh <tag> <output-dir>"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || die "must be run from inside a git repository"

ORIG_REF=$(git rev-parse HEAD)

TEMPLATES_DIR="${REPO_ROOT}/docs/_templates"
CSS_FILE="${REPO_ROOT}/docs/imgs/css/custom.css"
SAVED_TEMPLATES=""
SAVED_CSS=""

if [[ -d "$TEMPLATES_DIR" ]]; then
    SAVED_TEMPLATES=$(mktemp -d)
    cp -r "$TEMPLATES_DIR"/* "$SAVED_TEMPLATES/" 2>/dev/null || true
fi
if [[ -f "$CSS_FILE" ]]; then
    SAVED_CSS=$(mktemp)
    cp "$CSS_FILE" "$SAVED_CSS"
fi

cleanup() {
    echo "==> Restoring working tree to ${ORIG_REF}"
    git checkout "${ORIG_REF}" -- docs/ fvdb/ 2>/dev/null || true
    git checkout "${ORIG_REF}" -- .github/versions.json 2>/dev/null || true
    rm -rf "$SAVED_TEMPLATES" "$SAVED_CSS" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Checking out docs and source from ${TAG}"
git checkout "${TAG}" -- docs/ fvdb/
git checkout "${TAG}" -- .github/versions.json 2>/dev/null || true

if [[ -n "$SAVED_TEMPLATES" ]] && [[ -d "$SAVED_TEMPLATES" ]]; then
    echo "==> Restoring version switcher template into old docs"
    mkdir -p "$TEMPLATES_DIR"
    cp -r "$SAVED_TEMPLATES"/* "$TEMPLATES_DIR/"
fi
if [[ -n "$SAVED_CSS" ]] && [[ -f "$SAVED_CSS" ]]; then
    echo "==> Restoring custom CSS with version switcher styles"
    mkdir -p "$(dirname "$CSS_FILE")"
    cp "$SAVED_CSS" "$CSS_FILE"
fi

VERSION_NAME=$(basename "${OUTPUT}")
RELEASE="${TAG#v}"

echo "==> Patching docs/conf.py for mocked autodoc build (version=${VERSION_NAME}, release=${RELEASE})"
cat >> docs/conf.py <<PATCH

# --- Versioned docs: mock autodoc imports for archived builds ---
autodoc_mock_imports = ["fvdb", "fvdb._fvdb_cpp", "_fvdb_cpp"]
version = '${VERSION_NAME}'
release = '${RELEASE}'
PATCH

echo "==> Building docs into ${OUTPUT}"
sphinx-build docs/ -E -a "${OUTPUT}"

echo "==> Done building docs for ${TAG}"
