# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME nanovdb_editor
    # TODO when nanovdb_editor is available in a separate repo, update this
    # GITHUB_REPOSITORY openvdb/nanovdb-editor
    GIT_REPOSITORY https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/phapalova/openvdb.git
    # GIT_TAG feature/fvdb
    # SOURCE_DIR "/path/to/local/openvdb"
    DOWNLOAD_ONLY YES
)

# TODO only build nanovdb_editor if not already installed or the version has changed
if(nanovdb_editor_ADDED)
    set(NANOVDB_EDITOR_SOURCE_DIR "${nanovdb_editor_SOURCE_DIR}/nanovdb/nanovdb_editor")
    set(NANOVDB_EDITOR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/nanovdb_editor)
    file(MAKE_DIRECTORY ${NANOVDB_EDITOR_BUILD_DIR})

    message(STATUS "Building nanovdb_editor to ${NANOVDB_EDITOR_BUILD_DIR}...")
    execute_process(
        COMMAND bash -c "
        python -m build --wheel --outdir ${NANOVDB_EDITOR_BUILD_DIR} ${NANOVDB_EDITOR_SOURCE_DIR}/pymodule
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
