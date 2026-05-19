# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Include guard to prevent multiple inclusion
if(DEFINED _GET_TORCH_CMAKE_INCLUDED)
  return()
endif()
set(_GET_TORCH_CMAKE_INCLUDED TRUE)

# find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Check that PyTorch package uses the C++11 ABI and find site-packages directory
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env PYTHONPATH="${Python3_SITELIB}" "${Python3_EXECUTABLE}" -c "import os; import torch; print(f\"{torch._C._GLIBCXX_USE_CXX11_ABI};{os.path.dirname(torch.__file__)}\")"
  OUTPUT_VARIABLE TORCH_CXX11_ABI_AND_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_IMPORT_RESULT)

if(NOT TORCH_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to import PyTorch. Please ensure PyTorch is installed in the conda environment.")
endif()

list(GET TORCH_CXX11_ABI_AND_PACKAGE_DIR 0 TORCH_CXX11_ABI)
list(GET TORCH_CXX11_ABI_AND_PACKAGE_DIR 1 TORCH_PACKAGE_DIR)

if(NOT TORCH_CXX11_ABI)
  message(FATAL_ERROR "PyTorch package does not use the C++11 ABI. "
    "Please install PyTorch with the C++11 ABI (e.g. conda-forge package).")
endif()

# needed to correctly configure Torch with the conda-forge build
if(DEFINED ENV{CONDA_PREFIX})
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
endif()

find_package(Torch REQUIRED PATHS "${TORCH_PACKAGE_DIR}/share/cmake/Torch")

# Without this we can't find TH/THC headers
set(TORCH_SOURCE_INCLUDE_DIRS ${TORCH_PACKAGE_DIR}/include)

# Conda-forge's `pytorch-gpu` package installs the C++ headers at
# `$CONDA_PREFIX/include/{torch,ATen,c10,caffe2,tensorpipe}/`, and stages
# them into the site-packages tree via symlinks. Recent conda-forge builds
# (e.g. `pytorch-2.10.0-cuda130_mkl_py312_*_304`) have a packaging bug
# where those symlinks land as `torch.c~`, `ATen.c~`, ... (the conda
# file-conflict rename suffix), so `find_package(Torch)`'s
# `<site-packages>/torch/include/<...>` references don't resolve and
# `#include <torch/custom_class.h>` fails with "No such file or
# directory" when fvdb builds against a conda-forge torch.
#
# Append the conda env's bare `include/` so the canonical
# `$CONDA_PREFIX/include/torch/...` location is on the include path
# regardless of the symlink state. The IS_DIRECTORY guard keeps this a
# no-op for pip-installed torch (and for any non-conda environment).
if(DEFINED ENV{CONDA_PREFIX} AND IS_DIRECTORY "$ENV{CONDA_PREFIX}/include/torch")
    list(APPEND TORCH_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
endif()

if(NOT TORCH_PYTHON_LIBRARY)
  message(STATUS "Looking for torch_python library...")

  # Create a list of candidate paths
  set(TORCH_PYTHON_LIBRARY_CANDIDATES "${TORCH_PACKAGE_DIR}/lib/libtorch_python.so")

  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND TORCH_PYTHON_LIBRARY_CANDIDATES "$ENV{CONDA_PREFIX}/lib/libtorch_python.so")
  endif()

  # Iterate through candidates until found
  set(TORCH_PYTHON_LIBRARY_FOUND FALSE)

  foreach(CANDIDATE ${TORCH_PYTHON_LIBRARY_CANDIDATES})
    if(EXISTS "${CANDIDATE}")
      set(TORCH_PYTHON_LIBRARY "${CANDIDATE}")
      message(STATUS "Found libtorch_python.so at: ${TORCH_PYTHON_LIBRARY}")
      set(TORCH_PYTHON_LIBRARY_FOUND TRUE)
      break()
    endif()
  endforeach()

  # If not found, report error
  if(NOT TORCH_PYTHON_LIBRARY_FOUND)
    if(DEFINED ENV{CONDA_PREFIX})
      message(FATAL_ERROR "Could not find libtorch_python.so in any of the search locations.")
    else()
      message(FATAL_ERROR "Could not find libtorch_python.so. CONDA_PREFIX was not defined, so only site-packages location was checked.")
    endif()
  endif()
endif()

message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
