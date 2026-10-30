# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Temporary: the head of AcademySoftwareFoundation/openvdb#2291 (batched VBM build), on top of
# develop/nanovdb. Move back to AcademySoftwareFoundation/openvdb once #2291 merges.
CPMAddPackage(
    NAME nanovdb
    GITHUB_REPOSITORY swahtz/openvdb
    GIT_TAG 29165acbda12d2c6a3db809b9806a371038d7096
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
if(nanovdb_ADDED)
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE ${nanovdb_SOURCE_DIR}/nanovdb)
endif()
