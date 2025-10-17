# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME tinyply
    GITHUB_REPOSITORY ddiakopoulos/tinyply
    PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/../env/tinyply.patch || exit 0
    GIT_TAG 2.4
    VERSION 2.4
    DOWNLOAD_ONLY YES
)

# Create a header-only interface target to avoid installing tinyply
if(tinyply_ADDED)
    add_library(tinyply INTERFACE)
    target_include_directories(tinyply INTERFACE ${tinyply_SOURCE_DIR}/source)
endif()
