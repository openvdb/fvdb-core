# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

#   * To build with local repository, override the location with:
#       ./build.sh install -C cmake.define.CPM_nanovdb_editor_SOURCE=/path/to/nanovdb-editor
#   * To force rebuild, override the version check with:
#       ./build.sh install -C cmake.define.NANOVDB_EDITOR_FORCE=ON
#   * To skip nanovdb_editor wheel build:
#       ./build.sh install -C cmake.define.NANOVDB_EDITOR_SKIP=ON
#   IMPORTANT: variables are cached by cmake, so set them OFF to disable if not doing clean build

option(NANOVDB_EDITOR_FORCE "Force rebuild of nanovdb_editor wheel" OFF)
option(NANOVDB_EDITOR_SKIP "Skip nanovdb_editor wheel build" OFF)

CPMAddPackage(
    NAME nanovdb_editor
    GITHUB_REPOSITORY openvdb/nanovdb-editor
    GIT_TAG 64939c658f4b72b9c4ec905e50bfdda2bfbbd715
    VERSION 0.0.2
    DOWNLOAD_ONLY YES
)

if(nanovdb_editor_ADDED)
    # Check installed and downloaded nanovdb_editor versions and only proceed if newest is available
    file(READ ${nanovdb_editor_SOURCE_DIR}/pymodule/VERSION.txt NANOVDB_EDITOR_WHEEL_VERSION)
    string(STRIP "${NANOVDB_EDITOR_WHEEL_VERSION}" NANOVDB_EDITOR_WHEEL_VERSION)
    message(STATUS "Latest nanovdb_editor wheel version: ${NANOVDB_EDITOR_WHEEL_VERSION}")

    # Get installed nanovdb_editor version from the active Python
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
[[
import sys
try:
    import importlib.metadata as md
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
    # Ensure the build directory used by scikit-build exists; this is where the nested build writes
    file(MAKE_DIRECTORY ${nanovdb_editor_BINARY_DIR})

    message(STATUS "Removing existing nanovdb_editor wheel from ${NANOVDB_EDITOR_WHEEL_DIR}...")
    file(GLOB NANOVDB_EDITOR_WHEELS "${NANOVDB_EDITOR_WHEEL_DIR}/nanovdb_editor*.whl")
    foreach(wheel_file ${NANOVDB_EDITOR_WHEELS})
        file(REMOVE "${wheel_file}")
    endforeach()
    file(MAKE_DIRECTORY ${NANOVDB_EDITOR_WHEEL_DIR})
    message(STATUS "Building nanovdb_editor wheel version ${NANOVDB_EDITOR_WHEEL_VERSION} to ${NANOVDB_EDITOR_WHEEL_DIR}...")
    execute_process(
        COMMAND bash -lc "
        ${Python3_EXECUTABLE} -m pip wheel ${nanovdb_editor_SOURCE_DIR}/pymodule \
            --wheel-dir ${NANOVDB_EDITOR_WHEEL_DIR} \
            -Cbuild-dir=${nanovdb_editor_SOURCE_DIR}/../build \
            -Cbuild.verbose=false \
            -Clogging.level=WARNING \
            -Ccmake.define.NANOVDB_EDITOR_USE_GLFW=OFF \
            -Ccmake.define.NANOVDB_EDITOR_BUILD_TESTS=OFF \
            --config-settings=cmake.build-type=Release \
            -v \
            --no-build-isolation
        ${Python3_EXECUTABLE} -m pip install --force-reinstall ${NANOVDB_EDITOR_WHEEL_DIR}/nanovdb_editor*.whl
        "
        WORKING_DIRECTORY ${nanovdb_editor_BINARY_DIR}
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_error
    )
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR "nanovdb_editor wheel build failed.\nSTDOUT:\n${build_output}\n\nSTDERR:\n${build_error}")
    else()
        message(STATUS "${build_output}")
    endif()
endif()
