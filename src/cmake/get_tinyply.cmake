# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME tinyply
    GITHUB_REPOSITORY ddiakopoulos/tinyply
    GIT_TAG c9bb690dfe5e9105961e9e28120c48c9ae084bc6
    VERSION 3.0
    DOWNLOAD_ONLY YES
)

# Create a header-only interface target to avoid installing tinyply
if(tinyply_ADDED)
    add_library(tinyply INTERFACE)
    target_include_directories(tinyply INTERFACE ${tinyply_SOURCE_DIR}/source)
endif()
