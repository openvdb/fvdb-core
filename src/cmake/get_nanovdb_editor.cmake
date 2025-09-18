# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Override the location with ./build.sh install -C cmake.define.CPM_nanovdb_editor_SOURCE=/path/to/nanovdb-editor
CPMAddPackage(
    NAME nanovdb_editor
    GITHUB_REPOSITORY openvdb/nanovdb-editor
    GIT_TAG main
    DOWNLOAD_ONLY YES
)

if(nanovdb_editor_ADDED)
    set(NANOVDB_EDITOR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/nanovdb_editor)
    file(MAKE_DIRECTORY ${NANOVDB_EDITOR_BUILD_DIR})

    message(STATUS "Building nanovdb_editor wheel to ${NANOVDB_EDITOR_BUILD_DIR}...")
    execute_process(
        COMMAND bash -c "
        echo Building nanovdb_editor wheel...
        python pip wheel "${nanovdb_editor_SOURCE_DIR}/pymodule" --wheel-dir "${NANOVDB_EDITOR_BUILD_DIR}"
        echo Installing nanovdb_editor wheel...
        pip install --force-reinstall ${NANOVDB_EDITOR_BUILD_DIR}/*.whl
        "
        WORKING_DIRECTORY ${NANOVDB_EDITOR_BUILD_DIR}
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_error
    )
    message(STATUS "${build_output}")
    message(STATUS "${build_error}")
endif()
