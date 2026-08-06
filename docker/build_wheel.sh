#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Builds a production fvdb-core wheel inside a Docker container from the local
# checkout (uncommitted changes included) and copies it out to --output-dir.
# This is the same recipe the publish workflows use; version defaults come
# from .github/versions.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSIONS_JSON="${REPO_ROOT}/.github/versions.json"

die() {
  echo "Error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [options]

Builds a production fvdb-core wheel in Docker from this checkout and copies it
to the output directory. Defaults come from .github/versions.json.

Options:
  --python <ver>          Python version (default: ${DEF_PYTHON})
                          Supported: ${PY_MATRIX}
  --torch <ver>           Full PyTorch version (default: ${DEF_TORCH})
  --cuda <ver>            CUDA version (default: ${DEF_CUDA})
                          Supported: ${CUDA_KEYS}
  --cuda-arch-list <list> TORCH_CUDA_ARCH_LIST (default: ${DEF_ARCHES}),
                          or "native" to detect the host GPU via nvidia-smi
  --version-mode <mode>   How to stamp the wheel version (default: suffix):
                            suffix  - append +pt<torch>.cu<cuda> to the
                                      pyproject.toml version (publish default)
                            nightly - <base>.dev<YYYYMMDD>+pt<torch>.cu<cuda>
                            none    - leave pyproject.toml version unchanged
  --version <string>      Exact version override (mutually exclusive with
                          --version-mode)
  --skip-auditwheel       Skip the auditwheel manylinux repair step
  --jobs <n>              Parallel build jobs (CMAKE_BUILD_PARALLEL_LEVEL);
                          recommended when limiting container memory, since
                          auto-detection sees total host RAM
  --output-dir <dir>      Where the wheel is written (default: ./dist)
  -h, --help              Show this help and exit

Examples:
  $0
  $0 --python 3.11 --cuda 12.8 --cuda-arch-list native
  $0 --torch ${DEF_TORCH} --cuda-arch-list "8.0;8.6+PTX" --jobs 8
EOF
  exit 0
}

command -v python3 >/dev/null 2>&1 || die "python3 is required but not found"
[ -f "${VERSIONS_JSON}" ] || die "cannot find ${VERSIONS_JSON}"

# Load defaults from .github/versions.json (single source of truth shared
# with the CI workflows).
eval "$(python3 - "${VERSIONS_JSON}" <<'PY'
import json, sys

v = json.load(open(sys.argv[1]))
cuda = v["cuda"]
print(f'DEF_PYTHON="{v["python"]["default"]}"')
print(f'PY_MATRIX="{" ".join(v["python"]["matrix"])}"')
print(f'DEF_TORCH="{v["torch"]["full_version"]}"')
print(f'DEF_CUDA="{cuda["default"]}"')
print(f'DEF_ARCHES="{cuda["arch_list_publish"]}"')
print(f'CUDA_KEYS="{" ".join(cuda["versions"])}"')
for ver, info in cuda["versions"].items():
    print(f'CUDA_PATCH_{ver.replace(".", "_")}="{info["patch"]}"')
print(f'GCC_TOOLSET="{v["gcc"]["toolset"]}"')
print(f'CMAKE_VERSION="{v["cmake_version"]}"')
print(f'UV_VERSION="{v["uv"]["version"]}"')
print(f'AUDITWHEEL_EXCLUDE_LIBS="{" ".join(v["auditwheel_excludes"])}"')
print(f'AUDITWHEEL_EXCLUDE_CUDA_MAJOR_LIBS="{" ".join(v["auditwheel_excludes_cuda_major"])}"')
PY
)"

PYTHON_VERSION="${DEF_PYTHON}"
TORCH_VERSION="${DEF_TORCH}"
CUDA_VERSION="${DEF_CUDA}"
CUDA_ARCH_LIST="${DEF_ARCHES}"
VERSION_MODE="suffix"
VERSION_OVERRIDE=""
RUN_AUDITWHEEL=1
BUILD_JOBS=""
OUTPUT_DIR="${REPO_ROOT}/dist"

require_value() {
  [ "$#" -ge 2 ] && [[ "$2" != -* ]] || die "$1 requires a value"
}

while (( "$#" )); do
  case "$1" in
    --python)          require_value "$@"; PYTHON_VERSION="$2"; shift ;;
    --torch)           require_value "$@"; TORCH_VERSION="$2"; shift ;;
    --cuda)            require_value "$@"; CUDA_VERSION="$2"; shift ;;
    --cuda-arch-list)  require_value "$@"; CUDA_ARCH_LIST="$2"; shift ;;
    --version-mode)    require_value "$@"; VERSION_MODE="$2"; shift ;;
    --version)         require_value "$@"; VERSION_OVERRIDE="$2"; shift ;;
    --skip-auditwheel) RUN_AUDITWHEEL=0 ;;
    --jobs)            require_value "$@"; BUILD_JOBS="$2"; shift ;;
    --output-dir)      require_value "$@"; OUTPUT_DIR="$2"; shift ;;
    -h|--help)         usage ;;
    *)                 die "unknown option '$1' (see --help)" ;;
  esac
  shift
done

# --- Validation ---
command -v docker >/dev/null 2>&1 || die "docker is required but not found"
docker buildx version >/dev/null 2>&1 \
  || die "docker with BuildKit (buildx) is required; upgrade Docker or install the buildx plugin"
if ! grep -qw "${PYTHON_VERSION}" <<< "${PY_MATRIX}"; then
  die "Python ${PYTHON_VERSION} is not supported; supported versions: ${PY_MATRIX}"
fi
if ! grep -qw "${CUDA_VERSION}" <<< "${CUDA_KEYS}"; then
  die "CUDA ${CUDA_VERSION} is not supported; supported versions: ${CUDA_KEYS} (see .github/versions.json)"
fi
case "${VERSION_MODE}" in
  suffix|nightly|none) ;;
  *) die "--version-mode must be one of: suffix, nightly, none" ;;
esac
if [ -n "${VERSION_OVERRIDE}" ] && [ "${VERSION_MODE}" != "suffix" ]; then
  die "--version and --version-mode are mutually exclusive"
fi
if [ -n "${BUILD_JOBS}" ] && ! [[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  die "--jobs must be a positive integer"
fi

# --- Derivations ---
CUDA_PATCH_VAR="CUDA_PATCH_${CUDA_VERSION//./_}"
CUDA_IMAGE_TAG="${!CUDA_PATCH_VAR}"
CUDA_TAG="cu$(tr -d '.' <<< "${CUDA_VERSION}")"
CUDA_MAJOR="$(cut -d. -f1 <<< "${CUDA_VERSION}")"
TORCH_TAG="$(cut -d. -f1,2 <<< "${TORCH_VERSION}" | tr -d '.')"

if [ "${CUDA_ARCH_LIST}" = "native" ]; then
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "--cuda-arch-list native requires nvidia-smi on the host"
  CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | awk 'NF' | awk '!seen[$0]++' | sed 's/$/+PTX/' | paste -sd';' -)
  [ -n "${CUDA_ARCH_LIST}" ] || die "could not detect a GPU compute capability via nvidia-smi"
  echo "Detected native CUDA architectures: ${CUDA_ARCH_LIST}"
fi

AUDITWHEEL_EXCLUDES=""
for lib in ${AUDITWHEEL_EXCLUDE_LIBS}; do
  AUDITWHEEL_EXCLUDES+=" --exclude ${lib}"
done
for lib in ${AUDITWHEEL_EXCLUDE_CUDA_MAJOR_LIBS}; do
  AUDITWHEEL_EXCLUDES+=" --exclude ${lib}.${CUDA_MAJOR}"
done

CURRENT_VERSION=$(grep -E '^version *=' "${REPO_ROOT}/pyproject.toml" | head -n1 \
  | sed -E 's/^version *= *"([^"]+)".*/\1/')
[ -n "${CURRENT_VERSION}" ] || die "failed to read version from pyproject.toml"

LOCAL_SUFFIX="+pt${TORCH_TAG}.cu$(tr -d '.' <<< "${CUDA_VERSION}")"
if [ -n "${VERSION_OVERRIDE}" ]; then
  WHEEL_VERSION="${VERSION_OVERRIDE}"
elif [ "${VERSION_MODE}" = "suffix" ]; then
  WHEEL_VERSION="${CURRENT_VERSION}${LOCAL_SUFFIX}"
elif [ "${VERSION_MODE}" = "nightly" ]; then
  # Anchor the nightly at the upcoming release recorded in pyproject.toml
  # (e.g. 0.6.0.dev0 -> 0.6.0) so PEP 440 ordering puts nightlies between
  # the previous final release and the next one.
  BASE_VERSION=$(echo "${CURRENT_VERSION%%+*}" \
    | sed -E 's/(\.dev[0-9]+|\.post[0-9]+|(a|b|c|rc)[0-9]+)+$//')
  [ -n "${BASE_VERSION}" ] \
    || die "failed to parse base version from pyproject.toml (got '${CURRENT_VERSION}')"
  WHEEL_VERSION="${BASE_VERSION}.dev$(date -u '+%Y%m%d')${LOCAL_SUFFIX}"
else
  WHEEL_VERSION=""
fi

echo "Building fvdb-core wheel with:"
echo "  Python:          ${PYTHON_VERSION}"
echo "  PyTorch:         ${TORCH_VERSION} (index: https://download.pytorch.org/whl/${CUDA_TAG})"
echo "  CUDA:            ${CUDA_VERSION} (image: nvidia/cuda:${CUDA_IMAGE_TAG}-cudnn-devel-rockylinux8)"
echo "  CUDA archs:      ${CUDA_ARCH_LIST}"
echo "  Wheel version:   ${WHEEL_VERSION:-${CURRENT_VERSION} (unchanged)}"
echo "  auditwheel:      $([ "${RUN_AUDITWHEEL}" = "1" ] && echo enabled || echo skipped)"
echo "  Build jobs:      ${BUILD_JOBS:-auto}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo ""

export DOCKER_BUILDKIT=1
docker build \
  --file "${SCRIPT_DIR}/Dockerfile.wheel" \
  --target export \
  --output "type=local,dest=${OUTPUT_DIR}" \
  --build-arg CUDA_IMAGE_TAG="${CUDA_IMAGE_TAG}" \
  --build-arg UV_VERSION="${UV_VERSION}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg CUDA_TAG="${CUDA_TAG}" \
  --build-arg CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
  --build-arg GCC_TOOLSET="${GCC_TOOLSET}" \
  --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
  --build-arg WHEEL_VERSION="${WHEEL_VERSION}" \
  --build-arg RUN_AUDITWHEEL="${RUN_AUDITWHEEL}" \
  --build-arg AUDITWHEEL_EXCLUDES="${AUDITWHEEL_EXCLUDES}" \
  --build-arg BUILD_JOBS="${BUILD_JOBS}" \
  "${REPO_ROOT}"

echo ""
echo "Wheel(s) written to ${OUTPUT_DIR}:"
ls -1 "${OUTPUT_DIR}"/fvdb_core-*.whl
echo ""
echo "Install with the matching PyTorch build, e.g.:"
echo "  pip install torch==${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}"
echo "  pip install ${OUTPUT_DIR}/fvdb_core-*.whl"
