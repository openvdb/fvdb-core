# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Override the location with ./build.sh install -C cmake.define.CPM_nanovdb_editor_SOURCE=/path/to/nanovdb-editor
CPMAddPackage(
    NAME nanovdb_editor
    GITHUB_REPOSITORY openvdb/nanovdb-editor
    GIT_TAG 0634b32c5853694a4371c3170e03bcbb70edfe10
    DOWNLOAD_ONLY YES
)

if(nanovdb_editor_ADDED)
    message(STATUS "Building nanovdb_editor wheel...")
    execute_process(
        COMMAND bash -c "python -m pip install --force-reinstall ${nanovdb_editor_SOURCE_DIR}/pymodule"
        WORKING_DIRECTORY ${nanovdb_editor_SOURCE_DIR}/pymodule
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_error
    )
    if(NOT build_result EQUAL 0)
        message(STATUS ${build_output})
        message(FATAL_ERROR ${build_error})
    else()
        message(STATUS "nanovdb_editor wheel built and installed successfully.")
    endif()
endif()
