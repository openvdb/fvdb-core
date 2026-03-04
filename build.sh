#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

usage() {
  echo "Usage: $0 [-h|--help] [build_type] [options...]"
  echo ""
  echo "Builds or tests FVDB."
  echo ""
  echo "Arguments:"
  echo "  build_type     Specifies the build operation. Can be one of:"
  echo "                   install    - Build and install the package (default)."
  echo "                   wheel      - Build the Python wheel."
  echo "                   ctest      - Run tests (requires tests to be built)."
  echo "                   docstest   - Run pytest markdown documentation tests."
  echo "                   black      - Run black over Python (matches CI)."
  echo "                   clang-format - Run clang-format over src/ (matches CI)."
  echo ""
  echo "Options:"
  echo "  -h, --help     Display this help message and exit."
  echo "  --cuda-arch-list <value>  Set TORCH_CUDA_ARCH_LIST (auto-detects if not specified; "
  echo "                            use 'default' to force auto-detect)."
  echo "                            Example: --cuda-arch-list=\"8.0;8.6+PTX\""
  echo ""
  echo "Build Modifiers (for 'install' and 'wheel' build types, typically passed after build_type):"
  echo "  gtests         Enable building tests (sets FVDB_BUILD_TESTS=ON)."
  echo "  benchmarks     Enable building benchmarks (sets FVDB_BUILD_BENCHMARKS=ON)."
  echo "  editor_skip    Skip building and installing the nanovdb_editor dependency (sets NANOVDB_EDITOR_SKIP=ON)."
  echo "  editor_force   Force rebuild of the nanovdb_editor dependency (sets NANOVDB_EDITOR_FORCE=ON)."
  echo "  debug          Build in debug mode with full debug symbols and no optimizations."
  echo "  lineinfo       Enable CUDA lineinfo (sets FVDB_LINEINFO=ON)."
  echo "  strip_symbols  Strip symbols from the build (will be ignored if debug is enabled)."
  echo "  verbose        Enable verbose build output for pip and CMake."
  echo "  trace          Enable CMake trace output for debugging configuration."
  echo ""
  echo "  Any modifier arguments not matching above are passed through to pip."
  exit 0
}

setup_parallel_build_jobs() {
  # Calculate the optimal number of parallel build jobs based on available RAM
  RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
  if [ -z "$RAM_GB" ]; then
      echo "Error: Unable to determine available RAM"
      exit 1
  fi
  JOB_RAM_GB=3

  # Get number of processors
  NPROC=$(nproc)

  # count the number of ';' in the TORCH_CUDA_ARCH_LIST
  NUM_ARCH=$(echo "$TORCH_CUDA_ARCH_LIST" | tr ';' '\n' | wc -l)
  if [ "$NUM_ARCH" -lt 1 ]; then
    NUM_ARCH=1
  fi
  NVCC_THREADS=$NUM_ARCH

  # Check if we have enough RAM for even one job with full NVCC_THREADS
  # Requirement: JOB_RAM_GB * NVCC_THREADS
  MIN_RAM_REQUIRED=$((JOB_RAM_GB * NVCC_THREADS))

  if [ "$RAM_GB" -lt "$MIN_RAM_REQUIRED" ]; then
      NVCC_THREADS=1
  fi

  # Limit NVCC_THREADS to NPROC to ensure we don't oversubscribe
  if [ "$NVCC_THREADS" -gt "$NPROC" ]; then
      NVCC_THREADS=$NPROC
  fi

  # Determine max jobs based on CPU:
  # We want CMAKE_BUILD_PARALLEL_LEVEL * NVCC_THREADS <= NPROC
  MAX_JOBS_CPU=$((NPROC / NVCC_THREADS))

  # Determine max jobs based on RAM:
  # Assume each job requires JOB_RAM_GB * NVCC_THREADS
  MAX_JOBS_RAM=$((RAM_GB / (JOB_RAM_GB * NVCC_THREADS)))

  # Take the minimum
  PARALLEL_JOBS=$((MAX_JOBS_CPU < MAX_JOBS_RAM ? MAX_JOBS_CPU : MAX_JOBS_RAM))

  # Ensure at least 1 job
  if [ "$PARALLEL_JOBS" -lt 1 ]; then
    PARALLEL_JOBS=1
  fi

  # if CMAKE_BUILD_PARALLEL_LEVEL is set, use that
  if [ -n "$CMAKE_BUILD_PARALLEL_LEVEL" ]; then
    echo "Using CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
  else

    CMAKE_BUILD_PARALLEL_LEVEL=$PARALLEL_JOBS

    echo "Setting nvcc --threads to $NVCC_THREADS based on the number of CUDA architectures ($NUM_ARCH)"
    echo "Setting CMAKE_BUILD_PARALLEL_LEVEL to $CMAKE_BUILD_PARALLEL_LEVEL"
    echo "  Constraint: Total Threads ($((CMAKE_BUILD_PARALLEL_LEVEL * NVCC_THREADS))) <= NPROC ($NPROC)"
    echo "  Constraint: Estimated RAM ($((CMAKE_BUILD_PARALLEL_LEVEL * NVCC_THREADS * JOB_RAM_GB))) GB <= Available RAM ($RAM_GB GB)"

    export CMAKE_BUILD_PARALLEL_LEVEL
    export NVCC_THREADS
  fi
}

# Set TORCH_CUDA_ARCH_LIST based on the user's input.
set_cuda_arch_list() {
  local list="$1"
  if [ -n "$list" ] && [ "$list" != "default" ]; then
    echo "Using specified TORCH_CUDA_ARCH_LIST=$list"
    export TORCH_CUDA_ARCH_LIST="$list"
  else
    if ([ "$list" == "default" ] && [ -z "$TORCH_CUDA_ARCH_LIST" ]) && command -v nvidia-smi >/dev/null 2>&1; then
      echo "Detecting CUDA architectures via nvidia-smi"
      # Try via nvidia-smi (compute_cap available on newer drivers)
      TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NF' | awk '!seen[$0]++' | sed 's/$/+PTX/' | paste -sd';' -)
      export TORCH_CUDA_ARCH_LIST
      echo "Detected CUDA architectures: $TORCH_CUDA_ARCH_LIST"
    elif ([ "$list" == "default" ] && [ -n "$TORCH_CUDA_ARCH_LIST" ]); then
      echo "Using environment TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    else
      echo "Warning: Could not auto-detect CUDA architectures. Consider setting TORCH_CUDA_ARCH_LIST manually (e.g., 8.0;8.6+PTX)."
    fi
  fi
}

# Add a Python package's lib directory to LD_LIBRARY_PATH, if available
add_python_pkg_lib_to_ld_path() {
  local module_name="$1"
  local friendly_name="$2"
  local missing_lib_hint="$3"

  local lib_dir
  lib_dir=$(python - <<PY
import os
try:
  import ${module_name} as m
  print(os.path.join(os.path.dirname(m.__file__), 'lib'))
except Exception:
  print('')
PY
)

  if [ -n "$lib_dir" ] && [ -d "$lib_dir" ]; then
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Added ${friendly_name} lib directory to LD_LIBRARY_PATH: $lib_dir"
  else
    echo "Warning: Could not determine ${friendly_name} lib directory; gtests may fail to find ${missing_lib_hint}"
  fi
}

# --- Path Sanitization for Virtualized Environments ---
# In certain virtualized or hybrid environments, the PATH may include directories
# on slow or remote-mounted filesystems (e.g., /mnt/* host mounts, /media/sf_* for
# VirtualBox, /mnt/hgfs/* for VMware). CMake's find_library() searches these paths,
# and each filesystem operation on such mounts can go through slow protocols (9P, vboxsf,
# vmhgfs-fuse), causing significant delays during configuration.
# By filtering out these paths, we prevent CMake from crawling slow directories.

has_slow_host_mounts() {
  # Detect virtualized environments that may have slow host filesystem mounts
  # - WSL/WSL2: /mnt/* paths use 9P protocol
  # - VirtualBox: /media/sf_* paths use vboxsf
  # - VMware: /mnt/hgfs/* paths use vmhgfs-fuse
  if [ -f /proc/version ]; then
    grep -qi "microsoft\|wsl" /proc/version && return 0
  fi
  # VirtualBox Guest Additions
  if [ -d /media/sf_* ] 2>/dev/null || grep -q vboxsf /proc/filesystems 2>/dev/null; then
    return 0
  fi
  # VMware Tools
  if [ -d /mnt/hgfs ] || grep -q vmhgfs /proc/filesystems 2>/dev/null; then
    return 0
  fi
  return 1
}

filter_slow_mount_paths() {
  local input_path="$1"
  if has_slow_host_mounts; then
    # Filter out paths on slow host filesystem mounts:
    # - /mnt/* (WSL host mounts, VMware hgfs)
    # - /media/sf_* (VirtualBox shared folders)
    echo "$input_path" | tr ':' '\n' | grep -v -E "^/mnt/|^/media/sf_" | tr '\n' ':' | sed 's/:$//'
  else
    echo "$input_path"
  fi
}

run_with_sanitized_paths() {
  # Run a command with sanitized paths in virtualized environments, or normally otherwise.
  # This prevents CMake from searching slow host-mounted filesystems.
  if has_slow_host_mounts; then
    local sanitized_path
    sanitized_path=$(filter_slow_mount_paths "$PATH")

    # Build env command with sanitized variables
    local env_args=("PATH=$sanitized_path")

    # Also sanitize CMAKE_PREFIX_PATH and CMAKE_LIBRARY_PATH if set
    if [ -n "$CMAKE_PREFIX_PATH" ]; then
      env_args+=("CMAKE_PREFIX_PATH=$(filter_slow_mount_paths "$CMAKE_PREFIX_PATH")")
    fi
    if [ -n "$CMAKE_LIBRARY_PATH" ]; then
      env_args+=("CMAKE_LIBRARY_PATH=$(filter_slow_mount_paths "$CMAKE_LIBRARY_PATH")")
    fi

    echo "Virtualized environment detected: Running with sanitized paths (excluding slow host mounts for performance)"
    env "${env_args[@]}" "$@"
  else
    "$@"
  fi
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
fi

# Determine BUILD_TYPE from the first positional argument, default to 'install'.
# Handle shifting of arguments appropriately.
_first_arg_val="$1"
BUILD_TYPE="install" # Default build type

if [[ -n "$_first_arg_val" ]]; then
  if [[ "$_first_arg_val" == "install" || "$_first_arg_val" == "wheel" || "$_first_arg_val" == "ctest" || "$_first_arg_val" == "docstest" || "$_first_arg_val" == "black" || "$_first_arg_val" == "clang-format" ]]; then
    BUILD_TYPE="$_first_arg_val"
    shift # Consume the build_type argument
  else
    # _first_arg_val is not a recognized build type. Print usage and exit.
    echo "Error: Argument '$_first_arg_val' is not a recognized build_type."
    usage # This will also exit
  fi
fi

CONFIG_SETTINGS=""
PASS_THROUGH_ARGS=""
CUDA_ARCH_LIST_ARG="default"

# --- Set CUDA_HOME / Hints ---
# This helps scikit-build-core find the CUDA toolkit directly.
#
# We only auto-detect CUDA_HOME for Conda environments, and only when:
#   1. CUDA_HOME is not already set (respect user's explicit choice)
#   2. The Conda environment actually has nvcc installed
#
# In venv environments, PyTorch ships with rpathed CUDA libraries, so we leave
# CUDA_HOME unset to avoid version mismatches. Users can still set CUDA_HOME
# explicitly if needed.

if [ -n "$CONDA_PREFIX" ] && [ -z "$CUDA_HOME" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    echo "Conda environment with CUDA detected. Setting CUDA_HOME to $CONDA_PREFIX"
    export CUDA_HOME="$CONDA_PREFIX"
elif [ -n "$CONDA_PREFIX" ] && [ -n "$CUDA_HOME" ]; then
    echo "Conda environment detected, but CUDA_HOME is already set to $CUDA_HOME (keeping existing value)"
fi

# Export hints to help CMake find CUDA (only if CUDA_HOME is set)
if [ -n "$CUDA_HOME" ]; then
    # Always pass CUDA_TOOLKIT_ROOT_DIR as a search hint
    CONFIG_SETTINGS+=" --config-settings=cmake.define.CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"

    # Only set CUDACXX and CMAKE_CUDA_COMPILER if nvcc actually exists at that location.
    # This avoids hard failures when CUDA_HOME points to a runtime-only installation.
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        export CUDACXX="${CUDA_HOME}/bin/nvcc"
        CONFIG_SETTINGS+=" --config-settings=cmake.define.CMAKE_CUDA_COMPILER=$CUDACXX"
    else
        echo "Warning: CUDA_HOME is set to $CUDA_HOME but nvcc not found at $CUDA_HOME/bin/nvcc"
        echo "         CMake will attempt to find nvcc elsewhere."
    fi
fi

# Default values for nanovdb_editor build options
NANOVDB_EDITOR_SKIP=OFF
NANOVDB_EDITOR_FORCE=OFF

while (( "$#" )); do
  is_config_arg_handled=false
  if [[ "$BUILD_TYPE" == "install" || "$BUILD_TYPE" == "wheel" ]]; then
    if [[ "$1" == "gtests" ]]; then
      echo "Detected 'gtests' flag for $BUILD_TYPE build. Enabling FVDB_BUILD_TESTS."
      CONFIG_SETTINGS+=" --config-settings=cmake.define.FVDB_BUILD_TESTS=ON"
      is_config_arg_handled=true
    elif [[ "$1" == "benchmarks" ]]; then
      echo "Detected 'benchmarks' flag for $BUILD_TYPE build. Enabling FVDB_BUILD_BENCHMARKS."
      CONFIG_SETTINGS+=" --config-settings=cmake.define.FVDB_BUILD_BENCHMARKS=ON"
      is_config_arg_handled=true
    elif [[ "$1" == "verbose" ]]; then
      echo "Enabling verbose build"
      CONFIG_SETTINGS+=" -v -C build.verbose=true"
      is_config_arg_handled=true
    elif [[ "$1" == "trace" ]]; then
      echo "Enabling CMake trace"
      CONFIG_SETTINGS+=" --config-settings=cmake.args=--trace-expand"
      is_config_arg_handled=true
    elif [[ "$1" == "debug" ]]; then
      echo "Enabling debug build"
      CONFIG_SETTINGS+=" --config-settings=cmake.build-type=Debug  -C cmake.define.CMAKE_BUILD_TYPE=Debug"
      is_config_arg_handled=true
    elif [[ "$1" == "lineinfo" ]]; then
      echo "Enabling CUDA lineinfo"
      CONFIG_SETTINGS+=" --config-settings=cmake.define.FVDB_LINEINFO=ON"
      is_config_arg_handled=true
    elif [[ "$1" == "strip_symbols" ]]; then
      echo "Enabling strip symbols build"
      CONFIG_SETTINGS+=" --config-settings=cmake.define.FVDB_STRIP_SYMBOLS=ON"
      is_config_arg_handled=true
    elif [[ "$1" == "editor_skip" ]]; then
      echo "Detected 'editor_skip' flag for $BUILD_TYPE build. Enabling NANOVDB_EDITOR_SKIP."
      NANOVDB_EDITOR_SKIP=ON
      is_config_arg_handled=true
    elif [[ "$1" == "editor_force" ]]; then
      echo "Detected 'editor_force' flag for $BUILD_TYPE build. Enabling NANOVDB_EDITOR_FORCE."
      NANOVDB_EDITOR_FORCE=ON
      is_config_arg_handled=true
    fi
  fi

  if ! $is_config_arg_handled; then
    case "$1" in
      --cuda-arch-list=*)
        CUDA_ARCH_LIST_ARG="${1#*=}"
        is_config_arg_handled=true
        ;;
      --cuda-arch-list)
        shift
        CUDA_ARCH_LIST_ARG="$1"
        is_config_arg_handled=true
        ;;
      *)
        # Append other arguments, handling potential spaces safely
        PASS_THROUGH_ARGS+=" $(printf "%q" "$1")"
        ;;
    esac
  fi
  shift
done

CONFIG_SETTINGS+=" --config-settings=cmake.define.NANOVDB_EDITOR_SKIP=$NANOVDB_EDITOR_SKIP"
CONFIG_SETTINGS+=" --config-settings=cmake.define.NANOVDB_EDITOR_FORCE=$NANOVDB_EDITOR_FORCE"

# Construct PIP_ARGS with potential CMake args and other pass-through args
export PIP_ARGS="--no-build-isolation$CONFIG_SETTINGS$PASS_THROUGH_ARGS"

# Detect and export CUDA architectures early so builds pick it up
set_cuda_arch_list "$CUDA_ARCH_LIST_ARG"

if [ "$BUILD_TYPE" != "ctest" ] && [ "$BUILD_TYPE" != "docstest" ]; then
    setup_parallel_build_jobs
fi

# if the user specified 'wheel' as the build type, then we will build the wheel
if [ "$BUILD_TYPE" == "black" ]; then
    echo "Running black (matches CI)"
    ./src/scripts/run_black.sh

elif [ "$BUILD_TYPE" == "clang-format" ]; then
    echo "Running clang-format (matches CI clangFormatVersion=18, style=file)"
    ./src/scripts/run_clang_format.sh

elif [ "$BUILD_TYPE" == "wheel" ]; then
    echo "Build wheel"
    echo "pip wheel . --no-deps --wheel-dir dist/ $PIP_ARGS"
    run_with_sanitized_paths pip wheel . --no-deps --wheel-dir dist/ $PIP_ARGS

elif [ "$BUILD_TYPE" == "install" ]; then
    echo "Build and install package"
    # Always use --force-reinstall to ensure the freshly built package is installed,
    # even if pip thinks the version is already satisfied. The --no-deps flag ensures
    # this only affects fvdb-core itself, not dependencies like torch.
    echo "pip install --no-deps --force-reinstall $PIP_ARGS ."
    run_with_sanitized_paths pip install --no-deps --force-reinstall $PIP_ARGS .

elif [ "$BUILD_TYPE" == "ctest" ]; then

    # --- Ensure Test Data is Cached via CMake Configure Step ---
    echo "Ensuring test data is available in CPM cache..."

    if [ -z "$CPM_SOURCE_CACHE" ]; then
         echo "CPM_SOURCE_CACHE is not set"
    else
        echo "Using CPM_SOURCE_CACHE: $CPM_SOURCE_CACHE"
    fi

    # Assume this script runs from the source root directory
    SOURCE_DIR=$(pwd)
    TEMP_BUILD_DIR="build_temp_download_data"

    # Clean up previous temp dir and create anew
    rm -rf "$TEMP_BUILD_DIR"
    mkdir "$TEMP_BUILD_DIR"

    echo "Running CMake configure in temporary directory ($TEMP_BUILD_DIR) to trigger data download..."
    pushd "$TEMP_BUILD_DIR" > /dev/null
    run_with_sanitized_paths cmake "$SOURCE_DIR/src/cmake/download_test_data"
    popd > /dev/null # Back to SOURCE_DIR

    # Clean up temporary directory
    rm -rf "$TEMP_BUILD_DIR"
    echo "Test data caching step finished."
    # --- End Test Data Caching ---

    # --- Find and Run Tests ---
    echo "Searching for test build directory..."
    # Find CMakeCache.txt to locate the build root
    CMAKE_CACHE=$(find build -name CMakeCache.txt -type f -print -quit 2>/dev/null)

    if [ -z "$CMAKE_CACHE" ]; then
        echo "Error: Could not find CMakeCache.txt in build directory"
        echo "Please build the project first with tests enabled:"
        echo "pip install . -C cmake.define.FVDB_BUILD_TESTS=ON"
        exit 1
    fi

    # Construct the test directory path (where CTestTestfile.cmake is generated)
    # This discovers all tests from both src/tests/ and src/dispatch/
    BUILD_DIR="$(dirname "$CMAKE_CACHE")/src"

    if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
        echo "Error: No CTestTestfile.cmake found in $BUILD_DIR"
        echo "Please enable tests by building with:"
        echo "pip install . -C cmake.define.FVDB_BUILD_TESTS=ON"
        exit 1
    fi
    echo "Found test build directory: $BUILD_DIR"

    # Ensure required shared libraries are discoverable when running native gtests
    add_python_pkg_lib_to_ld_path "torch" "PyTorch" "libtorch.so"
    add_python_pkg_lib_to_ld_path "nanovdb_editor" "NanoVDB Editor" "libpnanovdb*.so"

    # Run ctest within the test build directory
    # Note: ctest doesn't need path sanitization as it doesn't search for libraries
    pushd "$BUILD_DIR" > /dev/null
    echo "Running ctest..."
    ctest --output-on-failure -LE compile_fail
    CTEST_EXIT_CODE=$?
    popd > /dev/null # Back to SOURCE_DIR

    echo "ctest finished with exit code $CTEST_EXIT_CODE."
    exit $CTEST_EXIT_CODE

elif [ "$BUILD_TYPE" == "docstest" ]; then
    echo "Running pytest markdown documentation tests..."
    pytest --markdown-docs ./docs --ignore-glob="**/wip/**"
    PYTEST_EXIT_CODE=$?
    echo "pytest markdown tests finished with exit code $PYTEST_EXIT_CODE."
    exit $PYTEST_EXIT_CODE

else
    echo "Invalid build/run type: $BUILD_TYPE"
    echo "Valid build/run types are: wheel, install, ctest, docstest, black, clang-format"
    exit 1
fi
