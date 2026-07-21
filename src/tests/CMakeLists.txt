# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Enable testing
enable_testing()

# Get GTest via CPM
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_google_test.cmake)
include(GoogleTest)

# Find libpng
find_package(PNG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_test_data.cmake)

# --- Define Test Utilities Library ---
add_library(fvdb_test_utils STATIC
  utils/ImageUtils.cpp
  utils/Tensor.cpp
)

set_target_properties(fvdb_test_utils
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# Test util headers (utils/Tensor.h, utils/ImageUtils.h) include torch headers,
# so torch must be a PUBLIC dependency. Test executables that include these
# headers inherit torch's usage requirements transitively.
target_include_directories(fvdb_test_utils PUBLIC
  ${CMAKE_SOURCE_DIR}/src # To find <tests/utils/ImageUtils.h>
)
target_include_directories(fvdb_test_utils PRIVATE
  ${PNG_INCLUDE_DIRS}
)

target_link_libraries(fvdb_test_utils
  PUBLIC torch
  PRIVATE PNG::PNG
)
# --- End Test Utilities Library Definition ---

# output directory
set(TEST_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gtests>")

# This function takes in a unit test name and test source and handles setting all of the
# associated properties and linking to build the test binary
function(ConfigureTest CMAKE_TEST_NAME)
  # Create object library for the test
  # This allows us to separate the compilation dependencies from the link dependencies
  # So that we don't recompile all the test source files when  we only need to relink
  add_library(${CMAKE_TEST_NAME}_obj OBJECT ${ARGN})

  set_target_properties(
    ${CMAKE_TEST_NAME}_obj
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  # Include directories needed to COMPILE the test object library
  # Test sources need to find <tests/utils/ImageUtils.h> etc.
  target_include_directories(${CMAKE_TEST_NAME}_obj PRIVATE
    ${CMAKE_SOURCE_DIR}/src # Include src for finding <tests/utils/...>
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
  )

  # Propagate usage requirements (include dirs, defines, SYSTEM marker) from
  # fvdb and gtest into the OBJECT library's compilation. linking an OBJECT lib
  # does not actually link — it only carries interface properties forward.
  target_link_libraries(${CMAKE_TEST_NAME}_obj PRIVATE
    fvdb
    GTest::gtest
    GTest::gtest_main
  )

  target_compile_options(${CMAKE_TEST_NAME}_obj PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -Xcompiler=-O0>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0>
    $<$<COMPILE_LANGUAGE:CXX>:
    "-Wall"
    "-Werror"
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
    "--extended-lambda"
    "-Xfatbin=-compress-all"
    "-Werror=all-warnings"
    "-Xcompiler=-Wall,-Werror"
    "-diag-suppress=3189" # Suppress warnings about module keyword in PyTorch code
    >)

  target_compile_definitions(${CMAKE_TEST_NAME}_obj PRIVATE
    "FVDB_EXTERNAL_TEST_DATA_PATH=\"${FVDB_TEST_DATA_DIR}\"")

  # Create the actual test executable using the object files
  add_executable(${CMAKE_TEST_NAME} $<TARGET_OBJECTS:${CMAKE_TEST_NAME}_obj>)

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIRECTORY}
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )

  # Add linker flags for Debug builds to preserve symbols
  target_link_options(${CMAKE_TEST_NAME} PRIVATE
    $<$<CONFIG:Debug>:-rdynamic>
  )

  # Link the FINAL executable against necessary libraries
  target_link_libraries(
    ${CMAKE_TEST_NAME}
    fvdb
    fvdb_test_utils # Link test executable against test utils library
    GTest::gtest
    GTest::gtest_main
    $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  # Register with CTest
  add_test(NAME ${CMAKE_TEST_NAME}
           COMMAND ${CMAKE_TEST_NAME}
           WORKING_DIRECTORY ${TEST_BINARY_DIRECTORY})

  # Set test properties
  set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES
    ENVIRONMENT "FVDB_EXTERNAL_TEST_DATA_PATH=${FVDB_TEST_DATA_DIR}"
  )

  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT testing
    DESTINATION bin/gtests/fvdb
    EXCLUDE_FROM_ALL
  )
endfunction()

# Configure an example test
ConfigureTest(ExampleTest "ExampleTest.cpp")

# Configure unit tests
ConfigureTest(HDDAIteratorsTest "HDDAIteratorsTest.cpp")
ConfigureTest(PrefetchTest "PrefetchTest.cpp")
ConfigureTest(LoadNanovdbTest "LoadNanovdbTest.cpp")
ConfigureTest(JaggedTensorTest "JaggedTensorTest.cpp")
ConfigureTest(PackedJaggedAccessorTest "PackedJaggedAccessorTest.cu")
ConfigureTest(GaussianComputeSparseInfoTest "GaussianComputeSparseInfoTest.cpp")
ConfigureTest(GaussianTileIntersectionTest "GaussianTileIntersectionTest.cpp")
ConfigureTest(GaussianComputeNanInfMaskTest "GaussianComputeNanInfMaskTest.cpp")
ConfigureTest(GaussianRasterizeBackwardTest "GaussianRasterizeBackwardTest.cpp")
ConfigureTest(GaussianRasterizeForwardTest "GaussianRasterizeForwardTest.cpp")
ConfigureTest(GaussianRasterizeWorldSpaceTest "GaussianRasterizeWorldSpaceTest.cpp")
ConfigureTest(GaussianSphericalHarmonicsForwardTest "GaussianSphericalHarmonicsForwardTest.cpp")
ConfigureTest(GaussianSphericalHarmonicsBackwardTest "GaussianSphericalHarmonicsBackwardTest.cpp")
ConfigureTest(GaussianProjectionForwardTest "GaussianProjectionForwardTest.cpp")
ConfigureTest(GaussianProjectionBackwardTest "GaussianProjectionBackwardTest.cpp")
ConfigureTest(GaussianProjectionUTTest "GaussianProjectionUTTest.cpp")
ConfigureTest(GaussianProjectionJaggedTest "GaussianProjectionJaggedTest.cpp")
ConfigureTest(GaussianPlyIOTest "GaussianPlyIOTest.cpp")
ConfigureTest(GaussianCamerasTest "GaussianCamerasTest.cu")
ConfigureTest(GaussianUtilsTest "GaussianUtilsTest.cu")
ConfigureTest(GaussianRasterizeTopContributorsTest "GaussianRasterizeTopContributorsTest.cpp")
ConfigureTest(GaussianRasterizeContributingGaussianIdsTest "GaussianRasterizeContributingGaussianIdsTest.cpp")
ConfigureTest(GaussianMCMCAddNoiseTest "GaussianMCMCAddNoiseTest.cpp")
ConfigureTest(GaussianMCMCRelocationTest "GaussianMCMCRelocationTest.cpp")
ConfigureTest(GatherScatterDefaultConvTest "GatherScatterDefaultConvTest.cu")
ConfigureTest(PredGatherIGemmTest "PredGatherIGemmTest.cu")
if(NANOVDB_EDITOR_INSTALLED)
  #ConfigureTest(ViewerTest "ViewerTest.cpp")
endif()
