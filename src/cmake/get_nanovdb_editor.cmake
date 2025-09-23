# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

#   * To build with local repository, override the location with:
#       ./build.sh install -C cmake.define.CPM_nanovdb_editor_SOURCE=/path/to/nanovdb-editor
#   * To force rebuild, override the version check with:
#       ./build.sh install -C cmake.define.NANOVDB_EDITOR_FORCE=ON
#   * To skip nanovdb_editor wheel build:
#       ./build.sh install -C cmake.define.NANOVDB_EDITOR_SKIP=ON

CPMAddPackage(
    NAME nanovdb_editor
    GITHUB_REPOSITORY openvdb/nanovdb-editor
    GIT_TAG 49b2626b4b594feab7e6577e305ed05139aaa7d7
    DOWNLOAD_ONLY YES
)

if(nanovdb_editor_ADDED)
    # Check installed and downloaded nanovdb_editor versions and only proceed if newest is available
    set(NANOVDB_EDITOR_WHEEL_VERSION_FILE ${nanovdb_editor_SOURCE_DIR}/pymodule/VERSION.txt)
    file(READ ${NANOVDB_EDITOR_WHEEL_VERSION_FILE} NANOVDB_EDITOR_WHEEL_VERSION)
    message(STATUS "NanoVDB Editor latest wheel version: ${NANOVDB_EDITOR_WHEEL_VERSION}")

    # Get installed nanovdb_editor version from the active Python
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
[[
import sys
try:
    import importlib.metadata as md
except Exception:
    try:
        import importlib_metadata as md
    except Exception:
        md = None

version = ''
if md is not None:
    try:
        version = md.version('nanovdb_editor')
    except Exception:
        pass

print(version, end='')
]]
        OUTPUT_VARIABLE NANOVDB_EDITOR_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NANOVDB_EDITOR_VERSION STREQUAL "")
        message(STATUS "Installed nanovdb_editor version not found.")
    else()
        message(STATUS "Installed nanovdb_editor version: ${NANOVDB_EDITOR_VERSION}")
    endif()

    if(NANOVDB_EDITOR_SKIP)
        message(STATUS "NANOVDB_EDITOR_SKIP is set; skipping nanovdb_editor wheel build.")
        return()
    endif()

    if (NOT NANOVDB_EDITOR_FORCE)
        if(NANOVDB_EDITOR_VERSION VERSION_GREATER_EQUAL NANOVDB_EDITOR_WHEEL_VERSION)
            message(STATUS "Installed nanovdb_editor is up-to-date; skipping build.")
            return()
        endif()
    else()
        message(STATUS "NANOVDB_EDITOR_FORCE is set; rebuilding nanovdb_editor wheel.")
    endif()

    set(NANOVDB_EDITOR_WHEEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)

    message(STATUS "Removing existing nanovdb_editor wheels from ${NANOVDB_EDITOR_WHEEL_DIR}...")
    file(GLOB NANOVDB_EDITOR_WHEELS "${NANOVDB_EDITOR_WHEEL_DIR}/nanovdb_editor*.whl")
    foreach(wheel_file ${NANOVDB_EDITOR_WHEELS})
        file(REMOVE "${wheel_file}")
    endforeach()
    file(MAKE_DIRECTORY ${NANOVDB_EDITOR_WHEEL_DIR})
    message(STATUS "Building nanovdb_editor wheel version ${NANOVDB_EDITOR_WHEEL_VERSION} to ${NANOVDB_EDITOR_WHEEL_DIR}...")
    execute_process(
        COMMAND bash -c "
        python -m pip wheel ${nanovdb_editor_SOURCE_DIR}/pymodule --wheel-dir ${NANOVDB_EDITOR_WHEEL_DIR} -Cbuild-dir=${nanovdb_editor_BINARY_DIR}
        echo -- Installing nanovdb_editor wheel...
        python -m pip install --force-reinstall ${NANOVDB_EDITOR_WHEEL_DIR}/*.whl
        "
        WORKING_DIRECTORY ${nanovdb_editor_BINARY_DIR}
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_error
    )
    if(NOT build_result EQUAL 0)
        message(STATUS ${build_output})
        message(FATAL_ERROR ${build_error})
    else()
        message(STATUS "nanovdb_editor wheel version ${NANOVDB_EDITOR_WHEEL_VERSION} built and installed successfully.")
    endif()
endif()
