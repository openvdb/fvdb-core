# Find and use the exact pybind11 version that PyTorch is using
# Use CPM for consistency with other project dependencies

# First make sure Python is found
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get PyTorch's pybind11 version. Sets TORCH_PYBIND11_INCLUDE_DIR and
# PYBIND11_VERSION in the parent scope when pytorch bundles its own pybind11
# (pip wheels do; conda-forge dropped this several versions ago). Leaves them
# unset otherwise, so the caller falls back to find_package(pybind11).
function(detect_torch_pybind11_version)
    set(_candidates)
    # pip pytorch bundles pybind11 under TORCH_PACKAGE_DIR/include (set by
    # get_torch.cmake). TORCH_INCLUDE_DIRS may or may not cover the same dir
    # depending on which branch of TorchConfig fires, so check it too.
    if(TORCH_PACKAGE_DIR)
        list(APPEND _candidates "${TORCH_PACKAGE_DIR}/include")
    endif()
    list(APPEND _candidates ${TORCH_INCLUDE_DIRS})

    foreach(dir ${_candidates})
        if(EXISTS "${dir}/pybind11/detail/common.h")
            set(PYBIND11_HEADER "${dir}/pybind11/detail/common.h")
            set(TORCH_PYBIND11_INCLUDE_DIR "${dir}" PARENT_SCOPE)
            break()
        endif()
    endforeach()

    if(NOT PYBIND11_HEADER)
        message(STATUS
            "PyTorch does not bundle pybind11 in its include tree; "
            "falling back to find_package(pybind11) for a compatible version.")
        return()
    endif()

    # Extract the version from the header by reading the file content
    file(READ "${PYBIND11_HEADER}" header_content)

    # First try standard version defines
    string(REGEX MATCH "#define PYBIND11_VERSION_MAJOR[ \t]+([0-9]+)" _ "${header_content}")
    set(MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define PYBIND11_VERSION_MINOR[ \t]+([0-9]+)" _ "${header_content}")
    set(MINOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define PYBIND11_VERSION_PATCH[ \t]+([0-9]+)" _ "${header_content}")
    set(PATCH "${CMAKE_MATCH_1}")

    # string(REGEX MATCH) sets CMAKE_MATCH_1 to "" on no match, which DEFINED
    # would still consider set — so check for non-empty instead.
    if(MAJOR AND MINOR AND PATCH)
        set(PYBIND11_VERSION_MAJOR "${MAJOR}" PARENT_SCOPE)
        set(PYBIND11_VERSION_MINOR "${MINOR}" PARENT_SCOPE)
        set(PYBIND11_VERSION_PATCH "${PATCH}" PARENT_SCOPE)
        set(PYBIND11_VERSION "${MAJOR}.${MINOR}.${PATCH}" PARENT_SCOPE)
        message(STATUS "Detected PyTorch's pybind11 version: ${MAJOR}.${MINOR}.${PATCH}")
    else()
        message(FATAL_ERROR "Could not detect PyTorch's pybind11 version from ${PYBIND11_HEADER}")
    endif()
endfunction()

function(check_pybind11_differences)
    message(STATUS "pybind11 include dir: ${pybind11_SOURCE_DIR}/include/pybind11")
    message(STATUS "TORCH_PYBIND11_INCLUDE_DIR: ${TORCH_PYBIND11_INCLUDE_DIR}")

    execute_process(
        COMMAND diff -r "${TORCH_PYBIND11_INCLUDE_DIR}/pybind11" "${pybind11_SOURCE_DIR}/include/pybind11"
        RESULT_VARIABLE diff_result
        OUTPUT_VARIABLE diff_output
        ERROR_VARIABLE diff_error
    )

    if(diff_result EQUAL 0)
        message(STATUS "No differences found between the pybind11 directories")
    else()
        message(WARNING "Differences found between the pybind11 directories:")
        message(STATUS "${diff_output}")
        if(diff_error)
            message(STATUS "Diff errors: ${diff_error}")
        endif()
    endif()
endfunction()

detect_torch_pybind11_version()
if (DEFINED TORCH_PYBIND11_INCLUDE_DIR)
    add_library(torch_pybind11_headers INTERFACE)
    target_include_directories(torch_pybind11_headers INTERFACE ${TORCH_PYBIND11_INCLUDE_DIR})
else()
    find_package(pybind11)
    if (pybind11_FOUND)
        message(STATUS "pybind11 found: ${pybind11_INCLUDE_DIRS}")
    else()
        # Set variables needed by pybind11
        set(PYBIND11_NEWPYTHON ON)
        set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
        set(PYTHON_LIBRARIES ${Python3_LIBRARIES})

        # Use CPM to fetch pybind11
        CPMAddPackage(
            NAME pybind11
            GITHUB_REPOSITORY pybind/pybind11
            VERSION ${PYBIND11_VERSION}
            GIT_TAG v${PYBIND11_VERSION}
            OPTIONS
            "PYBIND11_INSTALL ON"
            "PYBIND11_TEST OFF"
        )

        check_pybind11_differences()
    endif()
endif()
