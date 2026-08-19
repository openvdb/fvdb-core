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
  --versions-json PATH  Path to the released tag's .github/versions.json; when
                        given, also snapshot its torch/cuda/python versions
                        into the stable-release block in docs/conf.py and drop
                        the pre-release marker for this minor version from
                        docs/installation.rst
  --dry-run             Print what would change without modifying files
  -h, --help            Show this help message
EOF
}

die() { echo "error: $*" >&2; exit 1; }

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

# --- argument parsing ---------------------------------------------------------
VERSION=""
VERSIONS_JSON=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --versions-json)
            [[ $# -ge 2 ]] || die "--versions-json requires a path argument"
            VERSIONS_JSON="$2"; shift 2 ;;
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

# --- snapshot released torch/cuda/python into docs/conf.py stable block --------
if [[ -n "$VERSIONS_JSON" ]]; then
    [[ -f "$VERSIONS_JSON" ]] || die "versions json not found: $VERSIONS_JSON"

    TORCH_FULL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch"]["full_version"])' "$VERSIONS_JSON")"
    CUDA_LIST="$(python3 -c 'import json,sys; print(", ".join(f"\"{v}\"" for v in json.load(open(sys.argv[1]))["cuda"]["versions"]))' "$VERSIONS_JSON")"
    PYTHON_RANGE="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["python"]["matrix"]; print(f"{m[0]} - {m[-1]}")' "$VERSIONS_JSON")"

    if [[ -f "$CONF_PY" ]] && grep -q 'fvdb_core_stable_torch_version' "$CONF_PY"; then
        log "Updating stable release metadata in docs/conf.py:" \
            "torch=$TORCH_FULL cuda=[$CUDA_LIST] python=\"$PYTHON_RANGE\""
        if ! $DRY_RUN; then
            sed -i "s/^fvdb_core_stable_torch_version = \".*\"/fvdb_core_stable_torch_version = \"${TORCH_FULL}\"/" "$CONF_PY"
            sed -i "s/^fvdb_core_stable_cuda_versions = .*/fvdb_core_stable_cuda_versions = [${CUDA_LIST}]/" "$CONF_PY"
            sed -i "s/^fvdb_core_stable_python_range = \".*\"/fvdb_core_stable_python_range = \"${PYTHON_RANGE}\"/" "$CONF_PY"
            grep -q "^fvdb_core_stable_torch_version = \"${TORCH_FULL}\"" "$CONF_PY" \
                || warn "docs/conf.py stable metadata replacement did not match -- check variable format"
        fi
    else
        warn "docs/conf.py has no stable release metadata block (skipping snapshot)"
    fi

    # Drop the pre-release marker from this minor version's row in the docs
    # version matrix, now that its wheels are published. The marker is only
    # stripped from the row matching the released minor version so a patch
    # release of an older series does not un-mark the in-development row.
    INSTALL_RST="$REPO_ROOT/docs/installation.rst"
    MINOR="${VERSION%.*}"
    MARKER=" (pre-release, nightly wheels only)"
    if [[ -f "$INSTALL_RST" ]] && grep -q "^\s*\* - ${MINOR//./\\.}${MARKER}" "$INSTALL_RST"; then
        log "Removing pre-release marker for ${MINOR} from docs/installation.rst"
        if ! $DRY_RUN; then
            sed -i "/^\s*\* - ${MINOR//./\\.} (pre-release/s/${MARKER}//" "$INSTALL_RST"
        fi
    else
        log "No pre-release marker for ${MINOR} in docs/installation.rst (skipping)"
    fi
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
