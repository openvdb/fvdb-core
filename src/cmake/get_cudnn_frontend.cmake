# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME cudnn_frontend
    GITHUB_REPOSITORY NVIDIA/cudnn-frontend
    GIT_TAG v1.3.0
    DOWNLOAD_ONLY YES
    OPTIONS
        "CUDNN_FRONTEND_BUILD_SAMPLES OFF"
        "CUDNN_FRONTEND_BUILD_UNIT_TESTS OFF"
        "CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF"
)

# Create a header-only interface target to avoid installing cudnn_frontend
if(cudnn_frontend_ADDED)
    add_library(cudnn_frontend INTERFACE)
    target_include_directories(cudnn_frontend INTERFACE ${cudnn_frontend_SOURCE_DIR}/include)
endif()
