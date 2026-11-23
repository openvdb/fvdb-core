#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Formats C++/CUDA sources under src/ the same way CI checks them
# (.github/workflows/codestyle.yml: clang-format 18, style=file).
#
# Usage: run_clang_format.sh [format|check]
#   format  rewrite files in place (default)
#   check   exit non-zero if any file would change

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-format}"

CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-}"
if [[ -z "${CLANG_FORMAT_BIN}" ]]; then
    if command -v clang-format-18 >/dev/null 2>&1; then
        CLANG_FORMAT_BIN="clang-format-18"
    elif command -v clang-format >/dev/null 2>&1; then
        CLANG_FORMAT_BIN="clang-format"
    else
        echo "Error: clang-format not found in PATH."
        echo "CI uses clang-format 18. Install it or set CLANG_FORMAT_BIN=/path/to/clang-format."
        exit 127
    fi
elif ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
    echo "Error: CLANG_FORMAT_BIN='${CLANG_FORMAT_BIN}' not found in PATH."
    exit 127
fi

case "${MODE}" in
    format) CLANG_FORMAT_ARGS=(-i) ;;
    check)  CLANG_FORMAT_ARGS=(--dry-run --Werror) ;;
    *)
        echo "Error: unknown mode '${MODE}'. Expected 'format' or 'check'."
        exit 2
        ;;
esac

CLANG_FORMAT_VERSION="$(${CLANG_FORMAT_BIN} --version | head -n 1)"
echo "Using ${CLANG_FORMAT_BIN} (${CLANG_FORMAT_VERSION})"
if [[ "${CLANG_FORMAT_VERSION}" != *"version 18."* ]]; then
    echo "Warning: CI uses clang-format 18. Other versions may format differently."
fi

find "${SRC_DIR}" -type f \
    \( -name "*.h" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cu" -o -name "*.cuh" \) -print0 \
    | xargs -0 "${CLANG_FORMAT_BIN}" -style=file "${CLANG_FORMAT_ARGS[@]}"
