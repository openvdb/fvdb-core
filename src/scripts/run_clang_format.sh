#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# CI uses clang-format v18 with style=file on src/ for:
#   h,cpp,cc,cu,cuh
# See: .github/workflows/codestyle.yml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-}"
if [[ -z "${CLANG_FORMAT_BIN}" ]]; then
    if command -v clang-format-18 >/dev/null 2>&1; then
        CLANG_FORMAT_BIN="clang-format-18"
    elif command -v clang-format >/dev/null 2>&1; then
        CLANG_FORMAT_BIN="clang-format"
    else
        echo "Error: clang-format not found in PATH."
        echo ""
        echo "CI uses clang-format v18. To match CI locally, either:"
        echo "  - run inside your conda env: conda run -n fvdb ./build.sh clang-format"
        echo "  - or install clang-format-18 and re-run."
        exit 127
    fi
else
    if ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
        echo "Error: CLANG_FORMAT_BIN='${CLANG_FORMAT_BIN}' not found in PATH."
        exit 127
    fi
fi

echo "Using ${CLANG_FORMAT_BIN} ($(${CLANG_FORMAT_BIN} --version | head -n 1))"

format_files_in_dir() {
    local target_dir="$1"
    pushd "${target_dir}" >/dev/null
    # Use print0/xargs -0 to safely handle spaces.
    find . -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) -print0 \
        | xargs -0 "${CLANG_FORMAT_BIN}" -i -style=file
    popd >/dev/null
}

format_files_in_dir "${SRC_DIR}/benchmarks"
format_files_in_dir "${SRC_DIR}/dispatch"
format_files_in_dir "${SRC_DIR}/fvdb"
format_files_in_dir "${SRC_DIR}/python"
format_files_in_dir "${SRC_DIR}/tests"


