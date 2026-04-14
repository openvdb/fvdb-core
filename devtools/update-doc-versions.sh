#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Update version references in documentation and dependency metadata.
#
# This script is repo-agnostic: it works from inside fvdb-core,
# fvdb-reality-capture, or any repo that follows the same conventions.
#
# Usage: update-doc-versions.sh <fvdb-core-version> [options]
# Example: update-doc-versions.sh 0.4.0

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "error: must be run from inside a git repository" >&2; exit 1; }

# --- helpers ------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") <fvdb-core-version> [options]

Update fvdb-core version references in documentation (docs/conf.py) and,
if present, the fvdb-core floor dependency in pyproject.toml.

Arguments:
  <fvdb-core-version>   Version in MAJOR.MINOR.PATCH format (e.g. 0.4.0)

Options:
  --dry-run             Print what would change without modifying files
  -h, --help            Show this help message
EOF
}

die() { echo "error: $*" >&2; exit 1; }

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

# --- argument parsing ---------------------------------------------------------
VERSION=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)  usage; exit 0 ;;
        -*)         die "unknown option: $1" ;;
        *)
            [[ -z "$VERSION" ]] || die "unexpected argument: $1"
            VERSION="$1"; shift
            ;;
    esac
done

[[ -n "$VERSION" ]] || { usage; die "fvdb-core-version argument is required"; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must be MAJOR.MINOR.PATCH (got: $VERSION)"

# --- update docs/conf.py -----------------------------------------------------
CONF_PY="$REPO_ROOT/docs/conf.py"

if [[ -f "$CONF_PY" ]]; then
    if grep -q 'fvdb_core_stable_version' "$CONF_PY"; then
        log "Updating fvdb_core_stable_version in docs/conf.py to $VERSION"
        if ! $DRY_RUN; then
            sed -i "s/^fvdb_core_stable_version = \".*\"/fvdb_core_stable_version = \"${VERSION}\"/" "$CONF_PY"
            if ! grep -q "^fvdb_core_stable_version = \"${VERSION}\"" "$CONF_PY"; then
                warn "docs/conf.py sed replacement did not match -- check variable format"
            fi
        fi
    else
        warn "docs/conf.py exists but has no fvdb_core_stable_version variable"
    fi
else
    warn "docs/conf.py not found at $CONF_PY"
fi

# --- update docs/versions-config.json -----------------------------------------
VERSIONS_CONFIG="$REPO_ROOT/docs/versions-config.json"
MINOR="${VERSION%.*}"
TAG="v${VERSION}"
ARCHIVE="docs-${MINOR}.tar.gz"

if [[ -f "$VERSIONS_CONFIG" ]]; then
    if ! command -v jq &>/dev/null; then
        warn "jq not found; skipping versions-config.json update"
    else
        log "Updating docs/versions-config.json: version=${MINOR}, tag=${TAG}, stable=${MINOR}"
        if ! $DRY_RUN; then
            UPDATED=$(jq \
                --arg name "$MINOR" \
                --arg tag "$TAG" \
                --arg archive "$ARCHIVE" \
                '
                # Update or add the version entry for this minor
                .versions |= (
                    if any(.name == $name) then
                        map(if .name == $name then {name: $name, tag: $tag, archive: $archive} else . end)
                    else
                        [{name: $name, tag: $tag, archive: $archive}] + .
                    end
                )
                # Update stable pointer
                | .stable = $name
                ' "$VERSIONS_CONFIG")
            echo "$UPDATED" > "$VERSIONS_CONFIG"

            if ! jq empty "$VERSIONS_CONFIG" 2>/dev/null; then
                warn "docs/versions-config.json may have invalid JSON after update"
            fi
        fi
    fi
else
    log "No docs/versions-config.json found (skipping)"
fi

# --- update fvdb-core dependency floor in pyproject.toml ----------------------
PYPROJECT="$REPO_ROOT/pyproject.toml"

if [[ -f "$PYPROJECT" ]] && grep -q '"fvdb-core>=' "$PYPROJECT"; then
    log "Updating fvdb-core dependency floor in pyproject.toml to >=$VERSION"
    if ! $DRY_RUN; then
        sed -i "s/\(\"fvdb-core>=\)[0-9][0-9.]*/\1${VERSION}/" "$PYPROJECT"
        if ! grep -q "\"fvdb-core>=${VERSION}" "$PYPROJECT"; then
            warn "pyproject.toml sed replacement did not match -- check dependency format"
        fi
    fi
else
    log "No fvdb-core dependency in pyproject.toml (skipping)"
fi

log "Done."
