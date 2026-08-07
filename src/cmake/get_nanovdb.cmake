# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME nanovdb
    GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
    GIT_TAG b7fc4fc7af73e84071b5f625482cfad4c50eb247
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
if(nanovdb_ADDED)
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE ${nanovdb_SOURCE_DIR}/nanovdb)
endif()

# nanovdb::util::cuda::mallocAsync resolves to either cudaMallocAsync or plain
# cudaMalloc depending on whether NANOVDB_USE_SYNC_CUDA_MALLOC is defined. The
# choice is made in a header, so the macro has to be visible to every C++/CUDA
# translation unit that includes NanoVDB or an inline function ends up with two
# different bodies across the build. Hence a directory-wide compile definition
# rather than a per-target one.
#
# Async allocation depends on GPU unified memory, which is unavailable on vGPU
# slices (e.g. fractional-GPU cloud instances), so builds targeting those need
# the synchronous path.
option(FVDB_USE_SYNC_CUDA_MALLOC "Define NANOVDB_USE_SYNC_CUDA_MALLOC to force synchronous NanoVDB CUDA allocation" OFF)
if(FVDB_USE_SYNC_CUDA_MALLOC)
    message(STATUS "FVDB: compiling with NANOVDB_USE_SYNC_CUDA_MALLOC")
    add_compile_definitions(NANOVDB_USE_SYNC_CUDA_MALLOC)
endif()
