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
  echo ""
  echo "Options:"
  echo "  -h, --help     Display this help message and exit."
  echo "  --cuda-arch-list <value>  Set TORCH_CUDA_ARCH_LIST (auto-detects if not specified; "
  echo "                            use 'default' to force auto-detect)."
  echo "                            Example: --cuda-arch-list=8.0;8.6+PTX"
  echo ""
  echo "Build Modifiers (for 'install' and 'wheel' build types, typically passed after build_type):"
  echo "  gtests         Enable building tests (sets FVDB_BUILD_TESTS=ON)."
  echo "  benchmarks     Enable building benchmarks (sets FVDB_BUILD_BENCHMARKS=ON)."
  echo "  editor_skip    Skip building and installing the nanovdb_editor dependency (sets NANOVDB_EDITOR_SKIP=ON)."
  echo "  editor_force   Force rebuild of the nanovdb_editor dependency (sets NANOVDB_EDITOR_FORCE=ON)."
  echo "  debug          Build in debug mode with full debug symbols and no optimizations."
  echo "  verbose        Enable verbose build output for pip and CMake."
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

  # Calculate max jobs based on RAM
  RAM_JOBS=$(awk -v ram="$RAM_GB" -v job_ram="$JOB_RAM_GB" 'BEGIN { print int(ram / job_ram) }')

  # Get number of processors
  NPROC=$(nproc)

  # count the number of ';' in the TORCH_CUDA_ARCH_LIST
  NUM_ARCH=$(echo "$TORCH_CUDA_ARCH_LIST" | tr ';' '\n' | wc -l)
  if [ "$NUM_ARCH" -lt "$RAM_JOBS" ]; then
    # divide PARALLEL_JOBS by NUM_ARCH
    RAM_JOBS=$((RAM_JOBS / NUM_ARCH))
  else
    NUM_ARCH=1
  fi

  # Determine the minimum of RAM-based jobs and processor count
  PARALLEL_JOBS=$((RAM_JOBS > NPROC ? NPROC : RAM_JOBS))

  # Ensure at least 1 job
  if [ "$PARALLEL_JOBS" -lt 1 ]; then
    PARALLEL_JOBS=1
  fi


  # if CMAKE_BUILD_PARALLEL_LEVEL is set, use that
  if [ -n "$CMAKE_BUILD_PARALLEL_LEVEL" ]; then
    echo "Using CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
  else

    CMAKE_BUILD_PARALLEL_LEVEL=$PARALLEL_JOBS
    NVCC_THREADS=$NUM_ARCH

    echo "Setting nvcc --threads to $NVCC_THREADS based on the number of CUDA architectures"
    echo "Setting CMAKE_BUILD_PARALLEL_LEVEL to $CMAKE_BUILD_PARALLEL_LEVEL based on available RAM to target $JOB_RAM_GB GB per translation unit per $NUM_ARCH CUDA architectures"
    export CMAKE_BUILD_PARALLEL_LEVEL
    export NVCC_THREADS
  fi
}

# Set TORCH_CUDA_ARCH_LIST based on the user's input.
set_cuda_arch_list() {
  local list="$1"
  if [ -n "$list" ] && [ "$list" != "default" ]; then
    echo "Using TORCH_CUDA_ARCH_LIST=$list"
    export TORCH_CUDA_ARCH_LIST="$list"
  else
    if ([ "$list" == "default" ] || [ -z "$TORCH_CUDA_ARCH_LIST" ]) && command -v nvidia-smi >/dev/null 2>&1; then
      echo "Detecting CUDA architectures via nvidia-smi"
      # Try via nvidia-smi (compute_cap available on newer drivers)
      TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NF' | awk '!seen[$0]++' | sed 's/$/+PTX/' | paste -sd';' -)
    fi

    if [ -n "$TORCH_CUDA_ARCH_LIST" ]; then
      export TORCH_CUDA_ARCH_LIST
      echo "Detected CUDA architectures: $TORCH_CUDA_ARCH_LIST"
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

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
fi

# Determine BUILD_TYPE from the first positional argument, default to 'install'.
# Handle shifting of arguments appropriately.
_first_arg_val="$1"
BUILD_TYPE="install" # Default build type

if [[ -n "$_first_arg_val" ]]; then
  if [[ "$_first_arg_val" == "install" || "$_first_arg_val" == "wheel" || "$_first_arg_val" == "ctest" || "$_first_arg_val" == "docstest" ]]; then
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
    elif [[ "$1" == "debug" ]]; then
      echo "Enabling debug build"
      CONFIG_SETTINGS+=" --config-settings=cmake.build-type=debug"
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
if [ "$BUILD_TYPE" == "wheel" ]; then
    echo "Build wheel"
    echo "pip wheel . --wheel-dir dist/ $PIP_ARGS"
    pip wheel . --wheel-dir dist/ $PIP_ARGS
elif [ "$BUILD_TYPE" == "install" ]; then
    echo "Build and install package"
    echo "pip install --force-reinstall $PIP_ARGS ."
    pip install --force-reinstall $PIP_ARGS .
# TODO: Fix editable install
# else
#     echo "Build and install editable package"
#     echo "pip install $PIP_ARGS -e .  "
#     pip install $PIP_ARGS -e .
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
    cmake "$SOURCE_DIR/src/cmake/download_test_data"
    popd > /dev/null # Back to SOURCE_DIR

    # Clean up temporary directory
    rm -rf "$TEMP_BUILD_DIR"
    echo "Test data caching step finished."
    # --- End Test Data Caching ---

    # --- Find and Run Tests ---
    echo "Searching for test build directory..."
    # Find the directory containing the compiled tests (adjust if needed)
    # Using -print -quit to stop after the first match for efficiency
    BUILD_DIR=$(find build -name tests -type d -print -quit)

    if [ -z "$BUILD_DIR" ]; then
        echo "Error: Could not find build directory with tests"
        echo "Please enable tests by building with pip argument"
        echo "-C cmake.define.FVDB_BUILD_TESTS=ON"
        exit 1
    fi
    echo "Found test build directory: $BUILD_DIR"

    # Ensure required shared libraries are discoverable when running native gtests
    add_python_pkg_lib_to_ld_path "torch" "PyTorch" "libtorch.so"
    add_python_pkg_lib_to_ld_path "nanovdb_editor" "NanoVDB Editor" "libpnanovdb*.so"

    # Run ctest within the test build directory
    pushd "$BUILD_DIR" > /dev/null
    echo "Running ctest..."
    ctest --output-on-failure
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
    echo "Valid build/run types are: wheel, install, ctest, docstest"
    exit 1
fi
