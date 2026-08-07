# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# TEMPORARY: pinned to the merged nanovdb injectable-memory-resource stack
# (AcademySoftwareFoundation/openvdb PRs #2268, #2269, #2270, #2272, #2273 on
# top of upstream master d980ad92; tracking issue #2232). fvdb relies on the
# ResourceT seams these PRs add to route builder scratch through PyTorch's
# caching allocator (see src/fvdb/TorchResource.h). Repoint at
# AcademySoftwareFoundation/openvdb once the stack merges upstream.
CPMAddPackage(
    NAME nanovdb
    GITHUB_REPOSITORY swahtz/openvdb
    GIT_TAG 558bfb2ead7c993f980ba18e89fee4b69991fb39
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
if(nanovdb_ADDED)
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE ${nanovdb_SOURCE_DIR}/nanovdb)
endif()
