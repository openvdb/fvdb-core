# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME nanovdb
    GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
    GIT_TAG fec59777b768eae283659c5b87e377ac31653e41
    SOURCE_SUBDIR nanovdb/nanovdb
    DOWNLOAD_ONLY YES
)

# NanoVDB is header only, so we don't build it. Instead, we just add the headers
# to the include path and create an interface target.
#
# We also prepend an override directory that contains modified copies of a small
# number of upstream nanoVDB headers. The override directory comes first in the
# include search path so that, e.g., `#include <nanovdb/cuda/DeviceBuffer.h>`
# resolves to our forked copy under
# `src/fvdb/nanovdb_overrides/nanovdb/cuda/DeviceBuffer.h`. The rest of nanoVDB
# is still picked up from the upstream source tree, so the patch surface stays
# minimal and easy to resync.
#
# See src/fvdb/nanovdb_overrides/README.md for the rationale and the procedure
# for adding new overrides.
if(nanovdb_ADDED)
    get_filename_component(_fvdb_src_dir "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
    set(FVDB_NANOVDB_OVERRIDES_DIR
        "${_fvdb_src_dir}/fvdb/nanovdb_overrides"
        CACHE INTERNAL "Directory with fvdb-local overrides for nanoVDB headers")
    add_library(nanovdb INTERFACE)
    target_include_directories(nanovdb INTERFACE
        ${FVDB_NANOVDB_OVERRIDES_DIR}
        ${nanovdb_SOURCE_DIR}/nanovdb)
endif()
