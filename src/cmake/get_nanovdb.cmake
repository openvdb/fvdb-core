# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME nanovdb
    GITHUB_REPOSITORY matthewdcong/openvdb-aswf
    GIT_TAG 7bb1699a7db6830b50a66befcc245ae4e74455ed
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
if(nanovdb_ADDED)
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE ${nanovdb_SOURCE_DIR}/nanovdb)
endif()
