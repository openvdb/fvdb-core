# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# NanoVDB's CMake project provides the NanoVDB::nanovdb target. The NanoVDB
# headers carry the grid code, but Blosc support is a static C library,
# nanovdb_blosc, that NanoVDB::nanovdb links, so configuring NanoVDB needs a C
# compiler. With CPM_USE_LOCAL_PACKAGES=ON, CPM first looks for an installed
# NanoVDBConfig.cmake with a matching major version.
#
# fVDB needs Blosc and ZIP support for .nvdb I/O, and builds NanoVDB with CUDA.
#
# NanoVDB chooses CCCL. It uses the CUDA toolkit's CCCL when that meets
# NanoVDB's minimum, and otherwise fetches the CCCL release it pins. fVDB
# compiles against the same CCCL through NanoVDB::nanovdb, and its -I paths
# come before the toolkit's. Set CCCL_DIR to use another CCCL.

CPMAddPackage(
    NAME NanoVDB
    VERSION 32.11
    GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
    GIT_TAG 1ec456ef66ec6cde58cf175f693609c05ac9c98e
    SOURCE_SUBDIR nanovdb/nanovdb
    EXCLUDE_FROM_ALL YES
    OPTIONS
        "NANOVDB_BUILD_TOOLS OFF"
        "NANOVDB_USE_CUDA ON"
        "NANOVDB_USE_BLOSC ON"
        "NANOVDB_USE_ZLIB ON"
        "NANOVDB_USE_TBB OFF"
)

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

# libfvdb.so statically links nanovdb_blosc, so the wheel ships the notices for
# its BSD-licensed Blosc, LZ4 and FastLZ code. An installed NanoVDB has no
# source tree here and installs these notices itself.
set(FVDB_NANOVDB_BLOSC_DIR "${FVDB_NANOVDB_INCLUDE_DIR}/../ext/blosc/openvdb/ext/blosc")
if(NOT EXISTS "${FVDB_NANOVDB_BLOSC_DIR}/LICENSE.txt")
    unset(FVDB_NANOVDB_BLOSC_DIR)
endif()
