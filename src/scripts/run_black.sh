#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# CI uses psf/black@stable with:
#   --check --diff --verbose --target-version=py311 --line-length=120 --extend-exclude='wip/'
#   src: "./"
#   version: "~= 24.0"
# See: .github/workflows/codestyle.yml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${1:-format}" # "format" (default) or "check"

if ! python -c "import black" >/dev/null 2>&1; then
    echo "Error: black is not installed in this Python environment."
    echo "Install with:"
    echo "  python -m pip install -U \"black~=24.0\""
    exit 127
fi

COMMON_ARGS=(
    --target-version=py311
    --line-length=120
    --extend-exclude=wip/
)

pushd "${REPO_ROOT}" >/dev/null
if [[ "${MODE}" == "check" ]]; then
    python -m black --check --diff --verbose "${COMMON_ARGS[@]}" ./
elif [[ "${MODE}" == "format" ]]; then
    python -m black "${COMMON_ARGS[@]}" ./
else
    echo "Error: unknown mode '${MODE}'. Expected 'format' or 'check'."
    exit 2
fi
popd >/dev/null

