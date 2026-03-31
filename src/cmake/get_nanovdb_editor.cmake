# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

#   * fVDB always uses CPM-pinned NanoVDB Editor headers during builds.
#   * Viewer/runtime binaries are optional and come from the installed
#     nanovdb-editor pip package unless a local source checkout is provided.
#   * When CPM_nanovdb_editor_SOURCE points at a local repository, fVDB builds
#     and installs nanovdb_editor from that source checkout instead.
set(NANOVDB_EDITOR_BUILD_TYPE "Release" CACHE STRING "Build type for nanovdb_editor (Release/Debug)")

# fVDB pins NanoVDB Editor headers to an exact source commit via CPM
set(NANOVDB_EDITOR_TAG 62861a3b7f0fe2d4d61e7025b7f5b872086e965c)
set(NANOVDB_EDITOR_VERSION 0.0.23)   # version at this commit

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
set(NANOVDB_EDITOR_INCLUDE_DIR ${nanovdb_editor_SOURCE_DIR})
message(STATUS "Using NanoVDB Editor headers from ${NANOVDB_EDITOR_INCLUDE_DIR} (tag ${NANOVDB_EDITOR_TAG_SHORT})")

set(NANOVDB_EDITOR_BUILD_FROM_SOURCE OFF)
if(DEFINED CPM_nanovdb_editor_SOURCE AND NOT "${CPM_nanovdb_editor_SOURCE}" STREQUAL "")
    set(NANOVDB_EDITOR_BUILD_FROM_SOURCE ON)
    message(STATUS "CPM_nanovdb_editor_SOURCE is set; building nanovdb_editor from local source at ${CPM_nanovdb_editor_SOURCE}")
endif()

# Get nanovdb_editor site-packages directory
# Args:
#   NANOVDB_EDITOR_PACKAGE_DIR - output variable for package directory path
#   NANOVDB_EDITOR_INSTALLED - output variable indicating if nanovdb_editor is installed
function(get_installed_nanovdb_editor_dir NANOVDB_EDITOR_PACKAGE_DIR NANOVDB_EDITOR_INSTALLED)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${Python3_SITELIB} ${Python3_EXECUTABLE} -c
[[import os
try:
    import nanovdb_editor
    print(os.path.dirname(nanovdb_editor.__file__))
except Exception:
    pass
]]
        OUTPUT_VARIABLE _NANOVDB_EDITOR_PACKAGE_DIR
        RESULT_VARIABLE NANOVDB_EDITOR_IMPORTED
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NANOVDB_EDITOR_IMPORTED EQUAL 0)
        set(${NANOVDB_EDITOR_INSTALLED} ON PARENT_SCOPE)
        set(${NANOVDB_EDITOR_PACKAGE_DIR} ${_NANOVDB_EDITOR_PACKAGE_DIR} PARENT_SCOPE)
    endif()
endfunction()

function(get_installed_nanovdb_editor_version NANOVDB_EDITOR_INSTALLED_VERSION)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${Python3_SITELIB} ${Python3_EXECUTABLE} -c
[[try:
    import importlib.metadata as md
    print(md.version('nanovdb-editor'), end='')
except Exception:
    pass
]]
        OUTPUT_VARIABLE _NANOVDB_EDITOR_INSTALLED_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${NANOVDB_EDITOR_INSTALLED_VERSION} ${_NANOVDB_EDITOR_INSTALLED_VERSION} PARENT_SCOPE)
endfunction()

get_installed_nanovdb_editor_dir(NANOVDB_EDITOR_PACKAGE_DIR NANOVDB_EDITOR_INSTALLED)
if(NANOVDB_EDITOR_INSTALLED)
    get_installed_nanovdb_editor_version(NANOVDB_EDITOR_INSTALLED_VERSION)
    if(NANOVDB_EDITOR_INSTALLED_VERSION STREQUAL "")
        message(STATUS "Installed nanovdb_editor version not found")
    else()
        message(STATUS "Using installed nanovdb_editor binaries version ${NANOVDB_EDITOR_INSTALLED_VERSION} from ${NANOVDB_EDITOR_PACKAGE_DIR}")
    endif()
else()
    message(STATUS
        "nanovdb_editor is not installed. This is fine for core builds; "
        "viewer runtime features require the optional nanovdb-editor package "
        "or a local CPM_nanovdb_editor_SOURCE checkout.")
endif()

if(NOT NANOVDB_EDITOR_BUILD_FROM_SOURCE)
    return()
endif()

# Directory where locally built wheels are stored (project root /dist)
set(NANOVDB_EDITOR_WHEEL_DIR ${CMAKE_SOURCE_DIR}/dist)

set(VERSION_FILE ${nanovdb_editor_SOURCE_DIR}/pymodule/VERSION.txt)
if(NOT EXISTS ${VERSION_FILE})
    message(FATAL_ERROR "VERSION.txt file not found at ${VERSION_FILE}")
endif()

file(READ ${VERSION_FILE} NANOVDB_EDITOR_SOURCE_VERSION)
string(STRIP ${NANOVDB_EDITOR_SOURCE_VERSION} NANOVDB_EDITOR_SOURCE_VERSION)
if(NOT NANOVDB_EDITOR_SOURCE_VERSION STREQUAL NANOVDB_EDITOR_VERSION)
    message(WARNING
        "NanoVDB Editor source version ${NANOVDB_EDITOR_SOURCE_VERSION} does not match "
        "the pinned fVDB version ${NANOVDB_EDITOR_VERSION}")
endif()

# Build and install nanovdb_editor wheel
message(STATUS "Removing existing nanovdb_editor wheel from ${NANOVDB_EDITOR_WHEEL_DIR}...")
file(GLOB NANOVDB_EDITOR_WHEELS "${NANOVDB_EDITOR_WHEEL_DIR}/nanovdb_editor*.whl")
foreach(wheel_file ${NANOVDB_EDITOR_WHEELS})
    file(REMOVE "${wheel_file}")
endforeach()
file(MAKE_DIRECTORY ${NANOVDB_EDITOR_WHEEL_DIR})
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

message(STATUS "Building nanovdb_editor wheel version ${NANOVDB_EDITOR_SOURCE_VERSION} to ${NANOVDB_EDITOR_WHEEL_DIR}...")
execute_process(
    COMMAND bash -lc "
    ${Python3_EXECUTABLE} -m pip wheel ${nanovdb_editor_SOURCE_DIR}/pymodule \
        --wheel-dir ${NANOVDB_EDITOR_WHEEL_DIR} \
        -Cbuild-dir=${nanovdb_editor_SOURCE_DIR}/../build \
        -Cbuild.verbose=false \
        -Clogging.level=WARNING \
        -Ccmake.define.CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} \
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

    get_installed_nanovdb_editor_dir(NANOVDB_EDITOR_PACKAGE_DIR NANOVDB_EDITOR_INSTALLED)
    if(NOT NANOVDB_EDITOR_INSTALLED)
        message(FATAL_ERROR "nanovdb_editor installation verification failed after build")
    endif()
    message(STATUS "Installed nanovdb_editor binaries from ${NANOVDB_EDITOR_PACKAGE_DIR}")
    message(STATUS "NANOVDB_EDITOR_INCLUDE_DIR: ${NANOVDB_EDITOR_INCLUDE_DIR}")
endif()
