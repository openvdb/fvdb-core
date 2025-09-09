# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME nanovdb
    GITHUB_REPOSITORY matthewdcong/openvdb-aswf
    GIT_TAG 6f3f350dafc95f00f02d442b808e3838df6a30c6
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
if(nanovdb_ADDED)
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE ${nanovdb_SOURCE_DIR}/nanovdb)
endif()
