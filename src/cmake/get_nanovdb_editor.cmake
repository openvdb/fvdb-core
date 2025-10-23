# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

#   * To build with local repository, override the location with:
#       ./build.sh install -C cmake.define.CPM_nanovdb_editor_SOURCE=/path/to/nanovdb-editor
#       NOTE: variable is cached by cmake, to disable if not doing clean build:
#       ./build.sh install -C cmake.define.CPM_nanovdb_editor_SOURCE=
#   * To force rebuild, override the version check with:
#       ./build.sh install editor_force
#   * To skip nanovdb_editor wheel build:
#       ./build.sh install editor_skip

option(NANOVDB_EDITOR_FORCE "Force rebuild of nanovdb_editor wheel" OFF)
option(NANOVDB_EDITOR_SKIP "Skip nanovdb_editor wheel build" OFF)
set(NANOVDB_EDITOR_BUILD_TYPE "Release" CACHE STRING "Build type for nanovdb_editor (Release/Debug)")

# For fVDB main use nanovdb-editor main
set(NANOVDB_EDITOR_TAG fe2de1c140e84db248f19defe1a2b87642b49d39)
set(NANOVDB_EDITOR_VERSION 0.0.6)   # version at this commit

# If skip is set, get the latest tagged version to prevent unnecessary rebuilds each hash update
if(NANOVDB_EDITOR_SKIP)
    set(NANOVDB_EDITOR_VERSION 0.0.6)   # latest tagged version
    set(NANOVDB_EDITOR_TAG v${NANOVDB_EDITOR_VERSION})
endif()

CPMAddPackage(
    NAME nanovdb_editor
    GITHUB_REPOSITORY openvdb/nanovdb-editor
    GIT_TAG ${NANOVDB_EDITOR_TAG}
    VERSION ${NANOVDB_EDITOR_VERSION}
    DOWNLOAD_ONLY YES
)

if(NOT nanovdb_editor_ADDED)
    message(FATAL_ERROR "CPM failed to add nanovdb_editor package")
endif()

string(SUBSTRING "${NANOVDB_EDITOR_TAG}" 0 7 NANOVDB_EDITOR_TAG_SHORT)

if(NANOVDB_EDITOR_SKIP)
    if(NOT CPM_PACKAGE_nanovdb_editor_VERSION)
        message(STATUS "NANOVDB_EDITOR_SKIP is set; skipping nanovdb_editor wheel build, using the local repository for includes")
    else()
        message(STATUS "NANOVDB_EDITOR_SKIP is set; using the latest tagged version ${NANOVDB_EDITOR_TAG_SHORT} for includes")
    endif()
    set(NANOVDB_EDITOR_INCLUDE_DIR ${nanovdb_editor_SOURCE_DIR})
    message(STATUS "NANOVDB_EDITOR_INCLUDE_DIR: ${NANOVDB_EDITOR_INCLUDE_DIR}")
    return()
endif()

# Get nanovdb_editor site-packages directory
# Args:
#   NANOVDB_EDITOR_INCLUDE_DIR - output variable for include directory path
#   NANOVDB_EDITOR_INSTALLED - output variable indicating if nanovdb_editor is installed
function(get_installed_nanovdb_editor_dir NANOVDB_EDITOR_INCLUDE_DIR NANOVDB_EDITOR_INSTALLED)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${Python3_SITELIB} ${Python3_EXECUTABLE} -c
[[
import os
try:
    import nanovdb_editor
    print(os.path.dirname(nanovdb_editor.__file__))
except Exception:
    pass
]]
            OUTPUT_VARIABLE NANOVDB_EDITOR_PACKAGE_DIR
            RESULT_VARIABLE NANOVDB_EDITOR_IMPORTED
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NANOVDB_EDITOR_IMPORTED EQUAL 0)
        set(${NANOVDB_EDITOR_INSTALLED} ON PARENT_SCOPE)
        set(${NANOVDB_EDITOR_INCLUDE_DIR} ${NANOVDB_EDITOR_PACKAGE_DIR}/include PARENT_SCOPE)
    endif()
endfunction()

get_installed_nanovdb_editor_dir(NANOVDB_EDITOR_INCLUDE_DIR NANOVDB_EDITOR_INSTALLED)

# Get nanovdb_editor installed version
if(NANOVDB_EDITOR_INSTALLED)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${Python3_SITELIB} ${Python3_EXECUTABLE} -c
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
        OUTPUT_VARIABLE NANOVDB_EDITOR_INSTALLED_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NANOVDB_EDITOR_INSTALLED_VERSION STREQUAL "")
        message(STATUS "Installed nanovdb_editor version not found")
    else()
        message(STATUS "Installed nanovdb_editor version: ${NANOVDB_EDITOR_INSTALLED_VERSION}")
    endif()
else()
    message(STATUS "nanovdb_editor not installed")
endif()

# Check nanovdb_editor latest wheel version
set(VERSION_FILE ${nanovdb_editor_SOURCE_DIR}/pymodule/VERSION.txt)
if(NOT EXISTS ${VERSION_FILE})
    message(FATAL_ERROR "VERSION.txt file not found at ${VERSION_FILE}")
endif()

file(READ ${VERSION_FILE} NANOVDB_EDITOR_LATEST_VERSION)
string(STRIP ${NANOVDB_EDITOR_LATEST_VERSION} NANOVDB_EDITOR_LATEST_VERSION)
if(NOT NANOVDB_EDITOR_LATEST_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+")
    message(WARNING "Version format may be invalid: ${NANOVDB_EDITOR_LATEST_VERSION}")
endif()

message(STATUS "Latest nanovdb_editor version: ${NANOVDB_EDITOR_LATEST_VERSION}")

# Directory where locally built wheels are stored (project root /dist)
set(NANOVDB_EDITOR_WHEEL_DIR ${CMAKE_SOURCE_DIR}/dist)

# If not forcing a rebuild, check for installed version and compare with the latest wheel version on nanovdb-editor repo; skip build if up-to-date
if (NOT NANOVDB_EDITOR_FORCE)
    if(NANOVDB_EDITOR_INSTALLED_VERSION VERSION_GREATER_EQUAL NANOVDB_EDITOR_LATEST_VERSION)
        message(STATUS "Installed nanovdb_editor is up-to-date; skipping build")
        message(STATUS "NANOVDB_EDITOR_INCLUDE_DIR: ${NANOVDB_EDITOR_INCLUDE_DIR}")
        return()
    endif()
else()
    message(STATUS "NANOVDB_EDITOR_FORCE is set; rebuilding nanovdb_editor wheel")
endif()

# If not forcing a rebuild, check for the latest version wheel in /dist; build when version empty (local repo) or not found
if (NOT NANOVDB_EDITOR_FORCE)
    if(NOT CPM_PACKAGE_nanovdb_editor_VERSION)
        message(STATUS "Using local nanovdb_editor repository ${CPM_PACKAGE_nanovdb_editor_SOURCE_DIR}; will build wheel")
    else()
        file(GLOB LATEST_WHEELS "${NANOVDB_EDITOR_WHEEL_DIR}/nanovdb_editor-*${NANOVDB_EDITOR_LATEST_VERSION}*.whl")
        list(LENGTH LATEST_WHEELS NUM_LATEST_WHEELS)
        if(NUM_LATEST_WHEELS GREATER 0)
            list(GET LATEST_WHEELS 0 LATEST_WHEEL)
            message(STATUS "Found wheel in dist for the latest version ${NANOVDB_EDITOR_LATEST_VERSION}: ${LATEST_WHEEL}")
            execute_process(
                COMMAND bash -lc "
                ${Python3_EXECUTABLE} -m pip install --force-reinstall ${LATEST_WHEEL}
                "
                WORKING_DIRECTORY ${nanovdb_editor_BINARY_DIR}
                RESULT_VARIABLE install_result
                OUTPUT_VARIABLE install_output
                ERROR_VARIABLE install_error
            )
            if(NOT install_result EQUAL 0)
                message(FATAL_ERROR "nanovdb_editor wheel install failed.\nSTDOUT:\n${install_output}\n\nSTDERR:\n${install_error}")
            else()
                message(STATUS "Successfully installed: ${install_output}")

                get_installed_nanovdb_editor_dir(NANOVDB_EDITOR_INCLUDE_DIR NANOVDB_EDITOR_INSTALLED)
                if(NOT NANOVDB_EDITOR_INSTALLED)
                    message(FATAL_ERROR "nanovdb_editor installation verification failed")
                endif()
                message(STATUS "NANOVDB_EDITOR_INCLUDE_DIR: ${NANOVDB_EDITOR_INCLUDE_DIR}")
                return()
            endif()
        else()
            message(STATUS "No wheel found in dist for the latest version ${NANOVDB_EDITOR_LATEST_VERSION}; will build wheel")
        endif()
    endif()
endif()

# Build and install nanovdb_editor wheel
message(STATUS "Removing existing nanovdb_editor wheel from ${NANOVDB_EDITOR_WHEEL_DIR}...")
file(GLOB NANOVDB_EDITOR_WHEELS "${NANOVDB_EDITOR_WHEEL_DIR}/nanovdb_editor*.whl")
foreach(wheel_file ${NANOVDB_EDITOR_WHEELS})
    file(REMOVE "${wheel_file}")
endforeach()
# Ensure the wheel directory exists
file(MAKE_DIRECTORY ${NANOVDB_EDITOR_WHEEL_DIR})
# Ensure the build directory used by scikit-build exists; this is where the nested build writes
file(MAKE_DIRECTORY ${nanovdb_editor_BINARY_DIR})

find_package(Git QUIET)
if(NOT CPM_PACKAGE_nanovdb_editor_VERSION)
    message(STATUS "Using local nanovdb_editor repository: ${nanovdb_editor_SOURCE_DIR}")
    set(NANOVDB_EDITOR_COMMIT_HASH "unknown")
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${nanovdb_editor_SOURCE_DIR} rev-parse --short HEAD
            OUTPUT_VARIABLE NANOVDB_EDITOR_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _nanovdb_rev_result
        )
        if(NOT _nanovdb_rev_result EQUAL 0)
            set(NANOVDB_EDITOR_COMMIT_HASH "unknown")
        endif()
    endif()
    if(NANOVDB_EDITOR_COMMIT_HASH STREQUAL "unknown" AND DEFINED NANOVDB_EDITOR_TAG_SHORT)
        set(NANOVDB_EDITOR_COMMIT_HASH ${NANOVDB_EDITOR_TAG_SHORT})
    endif()
    message(STATUS "NanoVDB Editor build: ${NANOVDB_EDITOR_COMMIT_HASH}")
else()
    message(STATUS "Using nanovdb_editor version: ${CPM_PACKAGE_nanovdb_editor_VERSION} from ${nanovdb_editor_SOURCE_DIR}")
    set(NANOVDB_EDITOR_COMMIT_HASH ${NANOVDB_EDITOR_TAG_SHORT})
endif()

set(FVDB_COMMIT_HASH "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_SOURCE_DIR} rev-parse --short HEAD
        OUTPUT_VARIABLE FVDB_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _fvdb_rev_result
    )
    if(NOT _fvdb_rev_result EQUAL 0)
        set(FVDB_COMMIT_HASH "unknown")
    endif()
endif()

message(STATUS "Building nanovdb_editor wheel version ${NANOVDB_EDITOR_LATEST_VERSION} to ${NANOVDB_EDITOR_WHEEL_DIR}...")
execute_process(
    COMMAND bash -lc "
    ${Python3_EXECUTABLE} -m pip wheel ${nanovdb_editor_SOURCE_DIR}/pymodule \
        --wheel-dir ${NANOVDB_EDITOR_WHEEL_DIR} \
        -Cbuild-dir=${nanovdb_editor_SOURCE_DIR}/../build \
        -Cbuild.verbose=false \
        -Clogging.level=WARNING \
        -Ccmake.define.NANOVDB_EDITOR_USE_GLFW=OFF \
        -Ccmake.define.NANOVDB_EDITOR_BUILD_TESTS=OFF \
        -Ccmake.define.NANOVDB_EDITOR_COMMIT_HASH=${NANOVDB_EDITOR_COMMIT_HASH} \
        -Ccmake.define.NANOVDB_EDITOR_FVDB_COMMIT_HASH=${FVDB_COMMIT_HASH} \
        --config-settings=cmake.build-type=${NANOVDB_EDITOR_BUILD_TYPE} \
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

    get_installed_nanovdb_editor_dir(NANOVDB_EDITOR_INCLUDE_DIR NANOVDB_EDITOR_INSTALLED)
    if(NOT NANOVDB_EDITOR_INSTALLED)
        message(FATAL_ERROR "nanovdb_editor installation verification failed after build")
    endif()
    if(NOT EXISTS ${NANOVDB_EDITOR_INCLUDE_DIR})
        message(FATAL_ERROR "nanovdb_editor include directory not found: ${NANOVDB_EDITOR_INCLUDE_DIR}")
    endif()
    message(STATUS "NANOVDB_EDITOR_INCLUDE_DIR: ${NANOVDB_EDITOR_INCLUDE_DIR}")
endif()
