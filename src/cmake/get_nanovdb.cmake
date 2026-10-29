# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# NanoVDB is header only. Its CMake project provides the NanoVDB::nanovdb
# interface target. With CPM_USE_LOCAL_PACKAGES=ON, CPM first looks for an
# installed NanoVDBConfig.cmake with a matching major version.
#
# NanoVDB::nanovdb carries Blosc and ZIP support for .nvdb I/O. Blosc comes
# from NanoVDB's vendored nanovdb_blosc library. CUDA stays off because fVDB
# compiles against the CUDA toolkit's CCCL.
CPMAddPackage(
    NAME NanoVDB
    VERSION 32.9
    GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
    GIT_TAG 128555f50d19cd2f1ef1539a34768171070fa22e
    SOURCE_SUBDIR nanovdb/nanovdb
    EXCLUDE_FROM_ALL YES
    OPTIONS
        "NANOVDB_BUILD_TOOLS OFF"
        "NANOVDB_USE_CUDA OFF"
        "NANOVDB_USE_BLOSC ON"
        "NANOVDB_USE_ZLIB ON"
        "NANOVDB_USE_TBB OFF"
)

if(NOT TARGET NanoVDB::nanovdb)
    message(FATAL_ERROR "Expected the NanoVDB::nanovdb target from NanoVDB")
endif()

# FVDB_NANOVDB_INCLUDE_DIR holds the nanovdb/ header tree that fVDB bundles
# into its installs
get_target_property(_fvdb_nanovdb_imported NanoVDB::nanovdb IMPORTED)
if(_fvdb_nanovdb_imported)
    get_target_property(FVDB_NANOVDB_INCLUDE_DIR NanoVDB::nanovdb INTERFACE_INCLUDE_DIRECTORIES)
else()
    get_target_property(_fvdb_nanovdb_source_dir NanoVDB::nanovdb SOURCE_DIR)
    cmake_path(GET _fvdb_nanovdb_source_dir PARENT_PATH FVDB_NANOVDB_INCLUDE_DIR)
endif()
unset(_fvdb_nanovdb_imported)
unset(_fvdb_nanovdb_source_dir)
