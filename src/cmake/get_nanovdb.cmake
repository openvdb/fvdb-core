# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# NanoVDB's CMake project provides the NanoVDB::nanovdb target. The NanoVDB
# headers carry the grid code, but Blosc support is a static C library,
# nanovdb_blosc, that NanoVDB::nanovdb links, so configuring NanoVDB needs a C
# compiler. With CPM_USE_LOCAL_PACKAGES=ON, CPM first looks for an installed
# NanoVDBConfig.cmake with a matching major version.
#
# fVDB needs Blosc and ZIP support for .nvdb I/O, and builds NanoVDB with CUDA.
# NanoVDB's CUDA option uses an existing CCCL::CCCL target, so find the CUDA
# toolkit's CCCL first. Without it, NanoVDB would clone its own CCCL, and fVDB
# would compile against different CUB and Thrust headers than torch.
if(NOT TARGET CCCL::CCCL)
    find_package(CUDAToolkit REQUIRED)
    find_package(CCCL CONFIG REQUIRED
        HINTS "${CUDAToolkit_TARGET_DIR}/lib/cmake" "${CUDAToolkit_LIBRARY_DIR}/cmake")
endif()

CPMAddPackage(
    NAME NanoVDB
    VERSION 32.11
    GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
    GIT_TAG 86eeb05eb7bc2a3edf5c979fb724651500eb8561
    SOURCE_SUBDIR nanovdb/nanovdb
    EXCLUDE_FROM_ALL YES
    OPTIONS
        "NANOVDB_BUILD_TOOLS OFF"
        "NANOVDB_USE_CUDA ON"
        "NANOVDB_USE_BLOSC ON"
        "NANOVDB_USE_ZLIB ON"
        "NANOVDB_USE_TBB OFF"
)

if(NOT TARGET NanoVDB::nanovdb)
    message(FATAL_ERROR "Expected the NanoVDB::nanovdb target from NanoVDB")
endif()

# The OPTIONS above apply only when CPM adds NanoVDB as a subproject. An
# installed NanoVDB carries the options it was built with, so check them.
# NANOVDB_USE_NCCL changes how nanovdb/cuda/DeviceMesh.h compiles and adds an
# NCCL link dependency, so fVDB rejects it.
get_target_property(_fvdb_nanovdb_definitions NanoVDB::nanovdb INTERFACE_COMPILE_DEFINITIONS)
if(NOT _fvdb_nanovdb_definitions)
    set(_fvdb_nanovdb_definitions "")
endif()
list(TRANSFORM _fvdb_nanovdb_definitions REPLACE "^-D" "")
foreach(_fvdb_required NANOVDB_USE_BLOSC NANOVDB_USE_ZIP)
    if(NOT _fvdb_required IN_LIST _fvdb_nanovdb_definitions)
        message(FATAL_ERROR
            "fVDB requires a NanoVDB built with Blosc and ZLIB support, but "
            "NanoVDB::nanovdb does not define ${_fvdb_required}. Rebuild NanoVDB "
            "with NANOVDB_USE_BLOSC=ON and NANOVDB_USE_ZLIB=ON, or set "
            "CPM_USE_LOCAL_PACKAGES=OFF to build the pinned NanoVDB.")
    endif()
endforeach()
if("NANOVDB_USE_NCCL" IN_LIST _fvdb_nanovdb_definitions)
    message(FATAL_ERROR
        "fVDB requires a NanoVDB built without NCCL, but NanoVDB::nanovdb defines "
        "NANOVDB_USE_NCCL. Rebuild NanoVDB with NANOVDB_USE_NCCL=OFF, or set "
        "CPM_USE_LOCAL_PACKAGES=OFF to build the pinned NanoVDB.")
endif()
unset(_fvdb_required)
unset(_fvdb_nanovdb_definitions)

# FVDB_NANOVDB_INCLUDE_DIR holds the nanovdb/ header tree that fVDB bundles
# into its installs
get_target_property(FVDB_NANOVDB_INCLUDE_DIR NanoVDB::nanovdb NANOVDB_INCLUDE_DIR)
if(NOT FVDB_NANOVDB_INCLUDE_DIR)
    message(FATAL_ERROR "NanoVDB::nanovdb does not set NANOVDB_INCLUDE_DIR")
endif()
