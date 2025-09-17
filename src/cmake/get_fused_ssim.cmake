# Fetch fused-ssim sources via CPM and build a CMake target from its ext.cpp and CUDA sources

CPMAddPackage(
    NAME fused_ssim
    GITHUB_REPOSITORY rahul-goel/fused-ssim
    GIT_TAG b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3 # Most recent as of 2025-09-17
    OPTIONS
        "BUILD_TESTS OFF"
)

if(fused_ssim_ADDED OR fused_ssim_SOURCE_DIR)
    message(STATUS "fused-ssim downloaded via CPM: ${fused_ssim_SOURCE_DIR}")

    set(FUSED_SSIM_CPP_SOURCES "${fused_ssim_SOURCE_DIR}/ext.cpp")
    set(FUSED_SSIM_CUDA_SOURCES "${fused_ssim_SOURCE_DIR}/ssim.cu")

    enable_language(CUDA)

    # Generate the fvdb.metrics.ssim interface from fused-ssim's __init__.py
    set(FUSED_SSIM_INIT_PY "${fused_ssim_SOURCE_DIR}/fused_ssim/__init__.py")
    if(EXISTS "${FUSED_SSIM_INIT_PY}")
        file(READ "${FUSED_SSIM_INIT_PY}" FUSED_SSIM_INIT_PY_CONTENT)
    else()
        message(WARNING "Could not find fused_ssim/__init__.py at ${FUSED_SSIM_INIT_PY}")
        set(FUSED_SSIM_INIT_PY_CONTENT "")
    endif()

    # Make sure the output dir exists in the build tree
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fvdb/utils/metrics")

    # Generate into the build tree
    set(GENERATED_SSIM_PY "${CMAKE_CURRENT_BINARY_DIR}/fvdb/utils/metrics/ssim.py")
    configure_file(
        "${CMAKE_SOURCE_DIR}/fvdb/utils/metrics/ssim.py.in"
        "${GENERATED_SSIM_PY}"
        @ONLY
    )

    # Install into the Python package inside the wheel
    install(FILES "${GENERATED_SSIM_PY}" DESTINATION fvdb/utils/metrics)

    # Build a standalone Python extension module fused_ssim_cuda for compatibility with upstream imports
    # This ensures imports like `from fused_ssim_cuda import fusedssim, fusedssim_backward` work unmodified.
    add_library(fused_ssim_cuda MODULE ${FUSED_SSIM_CPP_SOURCES} ${FUSED_SSIM_CUDA_SOURCES})
    target_include_directories(fused_ssim_cuda PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
    target_compile_definitions(fused_ssim_cuda PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
    if(TORCH_CXX_FLAGS)
        separate_arguments(FUSED_TORCH_CXX_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(fused_ssim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FUSED_TORCH_CXX_FLAGS}>)
    endif()
    configure_torch_pybind11(fused_ssim_cuda CUDA)
    target_link_libraries(fused_ssim_cuda PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    set_target_properties(fused_ssim_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        PREFIX ""
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH FALSE)
    # Install at the wheel root (platlib) so `import fused_ssim_cuda` works
    install(TARGETS fused_ssim_cuda
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
else()
    message(FATAL_ERROR "Failed to fetch fused-ssim using CPM")
endif()
