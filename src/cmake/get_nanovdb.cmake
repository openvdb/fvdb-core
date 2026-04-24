# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# LOCAL BENCH REDIRECT (do not commit): picking up branchless HDDA changes from
# the openvdb-jswartz fork for end-to-end profiling of fvdb kernels. Revert
# this block to the GITHUB_REPOSITORY / GIT_TAG form before committing.
CPMAddPackage(
    NAME nanovdb
    SOURCE_DIR /home/jswartz/Development/openvdb-jswartz
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)
# Original pinned source (re-enable for upstream builds):
# CPMAddPackage(
#     NAME nanovdb
#     GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
#     GIT_TAG fec59777b768eae283659c5b87e377ac31653e41
#     SOURCE_SUBDIR nanovdb/nanovdb
#     DOWNLOAD_ONLY YES
# )

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
if(nanovdb_ADDED)
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE ${nanovdb_SOURCE_DIR}/nanovdb)
endif()
