# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

message(STATUS "CPM_SOURCE_CACHE: $ENV{CPM_SOURCE_CACHE}")

if (DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCECODE_DIR $ENV{CPM_SOURCE_CACHE})
else()
  set(CPM_SOURCECODE_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Ensure the cache directory exists
file(MAKE_DIRECTORY "${CPM_SOURCECODE_DIR}")

# Download CPM.cmake to the cache directory if it doesn't exist
if(NOT EXISTS "${CPM_SOURCECODE_DIR}/CPM.cmake")
  file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake
    "${CPM_SOURCECODE_DIR}/CPM.cmake"
    EXPECTED_HASH SHA256=c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d
    TIMEOUT 600
    INACTIVITY_TIMEOUT 60
    STATUS _cpm_dl_status
  )
  list(GET _cpm_dl_status 0 _cpm_dl_code)
  if(NOT _cpm_dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download CPM.cmake to ${CPM_SOURCECODE_DIR}: ${_cpm_dl_status}")
  endif()
endif()

# Include CPM.cmake from the cache
include("${CPM_SOURCECODE_DIR}/CPM.cmake")

# Sanity check that CPM loaded
if(NOT COMMAND CPMAddPackage)
  message(FATAL_ERROR "CPMAddPackage not available from ${CPM_SOURCECODE_DIR}/CPM.cmake")
endif()
