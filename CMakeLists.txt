# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Set global download timeout to 600 seconds (10 minutes)
set(CMAKE_DOWNLOAD_TIMEOUT 600 CACHE STRING "Time allowed for downloads in seconds")

option(FVDB_STRIP_SYMBOLS "Strip symbols from the build" OFF)
message(STATUS "FVDB_STRIP_SYMBOLS: ${FVDB_STRIP_SYMBOLS}")

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

find_package(Python3 COMPONENTS Development Interpreter)

message(STATUS "CMAKE_BUILD_PARALLEL_LEVEL: $ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "NVCC_THREADS: $ENV{NVCC_THREADS}")

# ----------------------------------------------------------------------------
# Compiler caching (ccache / sccache)
#
# Wraps every CXX and CUDA invocation so unchanged TUs are pulled from cache.
# Set FVDB_ENABLE_CCACHE=OFF to opt out (e.g. for CI nodes without persistent
# cache storage). The launcher variables are CACHE entries so they propagate
# into the src/ subproject's separate project() scope.
#
# Cold builds are unaffected; subsequent rebuilds and most touched-header
# scenarios drop from minutes to seconds. ccache 4.6+ recognizes nvcc out of
# the box.
# ----------------------------------------------------------------------------
option(FVDB_ENABLE_CCACHE "Enable ccache/sccache compiler caching" ON)
# We point ccache at a project-scoped CCACHE_DIR and apply CCACHE_MAXSIZE
# against *that* directory, so the size cap can't evict entries belonging to
# other projects sharing the user's global ~/.ccache. The directory is stable
# across build trees (under XDG_CACHE_HOME / ~/.cache) so switching branches
# or wiping `build/` still hits the cache. Set FVDB_CCACHE_DIR to "" (or "0")
# to fall back to the user's global ccache; in that mode we also skip the
# CCACHE_MAXSIZE override to avoid resizing a cache we don't own.
#
# One full multi-arch wheel build is ~250 MB compressed; 10 GB retains
# roughly 15-40 builds, comfortably covering branch switches and a day's
# worth of bisecting. Set FVDB_CCACHE_MAXSIZE to "0" (or "") to inherit the
# user's personal ccache.conf max_size for the project-local dir.
if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_FVDB_DEFAULT_CCACHE_DIR "$ENV{XDG_CACHE_HOME}/fvdb/ccache")
elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_FVDB_DEFAULT_CCACHE_DIR "$ENV{HOME}/.cache/fvdb/ccache")
else()
    set(_FVDB_DEFAULT_CCACHE_DIR "")
endif()
set(FVDB_CCACHE_DIR "${_FVDB_DEFAULT_CCACHE_DIR}" CACHE PATH
    "Project-scoped CCACHE_DIR. '' or '0' = use the user's global ccache (and skip MAXSIZE override).")
set(FVDB_CCACHE_MAXSIZE "10G" CACHE STRING
    "Per-invocation CCACHE_MAXSIZE override (e.g. '20G'). '0' or '' = inherit user config.")

if(FVDB_ENABLE_CCACHE)
    find_program(FVDB_COMPILER_CACHE NAMES ccache sccache)
    if(FVDB_COMPILER_CACHE)
        set(_fvdb_ccache_dir_active FALSE)
        if(FVDB_CCACHE_DIR AND NOT FVDB_CCACHE_DIR STREQUAL "0")
            set(_fvdb_ccache_dir_active TRUE)
        endif()

        # PCH-aware launcher.
        #
        # ccache refuses direct-mode hits on TUs that include a PCH unless
        # `time_macros` sloppiness is set (the PCH was preprocessed at build-PCH
        # time, so __TIME__/__DATE__ would always differ from the consuming TU).
        # `pch_defines` lets us survive minor -D flag changes that don't affect
        # the PCH contents. We bake both into the launcher via `cmake -E env`
        # so users get cache hits without needing a personal ccache.conf.
        set(_FVDB_CCACHE_LAUNCHER
            "${CMAKE_COMMAND}" "-E" "env"
            "CCACHE_SLOPPINESS=pch_defines,time_macros")
        if(_fvdb_ccache_dir_active)
            file(MAKE_DIRECTORY "${FVDB_CCACHE_DIR}")
            list(APPEND _FVDB_CCACHE_LAUNCHER
                "CCACHE_DIR=${FVDB_CCACHE_DIR}")
            # Only apply MAXSIZE when we own the cache directory; otherwise
            # we'd be resizing the user's shared global cache.
            if(FVDB_CCACHE_MAXSIZE AND NOT FVDB_CCACHE_MAXSIZE STREQUAL "0")
                list(APPEND _FVDB_CCACHE_LAUNCHER
                    "CCACHE_MAXSIZE=${FVDB_CCACHE_MAXSIZE}")
            endif()
        endif()
        list(APPEND _FVDB_CCACHE_LAUNCHER "${FVDB_COMPILER_CACHE}")

        if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
            set(CMAKE_CXX_COMPILER_LAUNCHER "${_FVDB_CCACHE_LAUNCHER}"
                CACHE STRING "Compiler launcher for CXX (auto-detected)")
        endif()
        if(NOT DEFINED CMAKE_CUDA_COMPILER_LAUNCHER)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${_FVDB_CCACHE_LAUNCHER}"
                CACHE STRING "Compiler launcher for CUDA (auto-detected)")
        endif()

        if(_fvdb_ccache_dir_active)
            if(FVDB_CCACHE_MAXSIZE AND NOT FVDB_CCACHE_MAXSIZE STREQUAL "0")
                message(STATUS "FVDB: compiler caching enabled via ${FVDB_COMPILER_CACHE} "
                               "(dir=${FVDB_CCACHE_DIR}, "
                               "max_size=${FVDB_CCACHE_MAXSIZE}, "
                               "sloppiness=pch_defines,time_macros)")
            else()
                message(STATUS "FVDB: compiler caching enabled via ${FVDB_COMPILER_CACHE} "
                               "(dir=${FVDB_CCACHE_DIR}, "
                               "max_size=user ccache.conf, "
                               "sloppiness=pch_defines,time_macros)")
            endif()
        else()
            message(STATUS "FVDB: compiler caching enabled via ${FVDB_COMPILER_CACHE} "
                           "(dir=user global, max_size=user ccache.conf, "
                           "sloppiness=pch_defines,time_macros)")
        endif()
    else()
        message(STATUS "FVDB: ccache/sccache not found on PATH; "
                       "install one to speed up incremental builds")
    endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_pybind11.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_nanovdb.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/configure_torch_pybind11.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_nanovdb_editor.cmake)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure libfvdb
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/)

# Source files
set(FVDB_BINDINGS_CPP_FILES
    src/python/Bindings.cpp
    src/python/FusedSSIMBinding.cpp
    src/python/GaussianSplatOps.cpp
    src/python/GridBatchDataBinding.cpp
    src/python/GridBatchOps.cpp
    src/python/JaggedTensorBinding.cpp
    src/python/PersistentTSDFStateBinding.cpp
    src/python/ViewerBinding.cpp)

# Build library
Python3_add_library(_fvdb_cpp ${FVDB_BINDINGS_CPP_FILES} MODULE)
if (DEFINED TORCH_PYBIND11_INCLUDE_DIR)
    target_link_libraries(_fvdb_cpp PUBLIC torch_pybind11_headers)
else()
    target_link_libraries(_fvdb_cpp PUBLIC pybind11::headers)
endif()
set_target_properties(_fvdb_cpp PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

target_link_options(_fvdb_cpp PRIVATE "-Wl,--no-as-needed")

set_target_properties(_fvdb_cpp PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    POSITION_INDEPENDENT_CODE ON
    INTERFACE_POSITION_INDEPENDENT_CODE ON)

target_include_directories(_fvdb_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    # Overrides directory must come BEFORE the upstream nanoVDB source so that
    # forked headers (e.g. nanovdb/cuda/DeviceBuffer.h) win the include search.
    # Mirror the ordering used in src/CMakeLists.txt for the fvdb target.
    # See src/fvdb/nanovdb_overrides/README.md.
    ${FVDB_NANOVDB_OVERRIDES_DIR}
    ${nanovdb_SOURCE_DIR}/nanovdb
    ${NANOVDB_EDITOR_INCLUDE_DIR})
target_link_libraries(_fvdb_cpp PRIVATE
    ${TORCH_PYTHON_LIBRARY}
    ${TORCH_LIBRARIES}
    fvdb)

target_compile_options(_fvdb_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
    #"-Wno-unused-variable"
    "-fvisibility=hidden"
    "-fvisibility-inlines-hidden"
    "-Wall"
    "-Werror"
    "${TORCH_CXX_FLAGS}"
    >)

target_compile_definitions(_fvdb_cpp PRIVATE
    TORCH_EXTENSION_NAME=_fvdb_cpp
    TORCH_API_INCLUDE_EXTENSION_H)

# Configure the pybind11 module with PyTorch compatibility settings
configure_torch_pybind11(_fvdb_cpp)

# All bindings TUs include <torch/types.h> + pybind11. Pre-compile both so the
# 30+ s ATen+pybind parse cost is paid once. Honors the same opt-in flag as
# the fvdb library above (FVDB_ENABLE_PCH, default ON).
if(FVDB_ENABLE_PCH)
    target_precompile_headers(_fvdb_cpp PRIVATE
        <torch/types.h>
        <pybind11/pybind11.h>
        <pybind11/stl.h>
        <vector>
        <memory>
        <optional>
        <string>)
endif()

# Add linker flags for Debug builds to preserve symbols
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_link_options(_fvdb_cpp PRIVATE "-rdynamic")
endif()

# Strip symbols in non-Debug builds to reduce wheel size
if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo") AND FVDB_STRIP_SYMBOLS)
    target_link_options(_fvdb_cpp PRIVATE "-s")
endif()

# Only install the Python package artifacts; the core lib is installed as part of the Python module
install(TARGETS _fvdb_cpp fvdb
    LIBRARY DESTINATION fvdb COMPONENT PythonModule
    RUNTIME DESTINATION fvdb COMPONENT PythonModule)

# Public headers for downstream C++ consumers (bundled in wheel like torch/include/)
file(GLOB FVDB_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/fvdb/*.h")
install(FILES ${FVDB_PUBLIC_HEADERS}
    DESTINATION fvdb/include/fvdb
    COMPONENT PythonModule)

install(DIRECTORY ${nanovdb_SOURCE_DIR}/nanovdb/nanovdb/
    DESTINATION fvdb/include/nanovdb
    COMPONENT PythonModule
    FILES_MATCHING PATTERN "*.h")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/dispatch/dispatch/macros.h
    DESTINATION fvdb/include/dispatch/dispatch
    COMPONENT PythonModule)

# CMake config for find_package(fvdb) against the wheel install
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/fvdbWheelConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fvdbConfig.cmake
    @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fvdbConfig.cmake
    DESTINATION fvdb/share/cmake/fvdb
    COMPONENT PythonModule)

# Capture the git short SHA for embedding in version.py
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE FVDB_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT FVDB_GIT_SHA)
    set(FVDB_GIT_SHA "Unknown")
endif()

set(VERSION_PY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/fvdb/version.py")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/fvdb/version.py.in" "${VERSION_PY_PATH}" @ONLY)

# Make sure version.py is included in the installation
install(FILES "${VERSION_PY_PATH}" DESTINATION fvdb COMPONENT PythonModule)
