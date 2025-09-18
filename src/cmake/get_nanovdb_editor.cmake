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
    set(NANOVDB_EDITOR_WHEEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
    file(MAKE_DIRECTORY ${NANOVDB_EDITOR_WHEEL_DIR})

    message(STATUS "Building nanovdb_editor wheel to ${NANOVDB_EDITOR_WHEEL_DIR}...")
    execute_process(
        COMMAND bash -c "
        python -m pip wheel ${nanovdb_editor_SOURCE_DIR}/pymodule --wheel-dir ${NANOVDB_EDITOR_WHEEL_DIR}
        echo -- Installing nanovdb_editor wheel...
        python -m pip install --force-reinstall ${NANOVDB_EDITOR_WHEEL_DIR}/*.whl
        "
        WORKING_DIRECTORY ${NANOVDB_EDITOR_WHEEL_DIR}
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
