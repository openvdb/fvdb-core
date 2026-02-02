# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# Dispatch Interface Library
# ============================================================================
# This is a header-only interface library that provides dispatch utilities.
# It inherits CUDA and C++20 settings from the parent build.

add_library(dispatch INTERFACE)

target_include_directories(dispatch INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ============================================================================
# Examples Library (built when tests or benchmarks are enabled)
# ============================================================================
# Examples are compiled as a library that tests and benchmarks link against.

if(FVDB_BUILD_TESTS OR FVDB_BUILD_BENCHMARKS)
    set(DISPATCH_EXAMPLE_SOURCES
        examples/functional.cu
        examples/op.cu
        examples/relu.cu
        examples/gelu_for_each.cu
        examples/scan_lib.cpp
        examples/scan_lib.cu
    )

    add_library(dispatch_examples STATIC ${DISPATCH_EXAMPLE_SOURCES})

    set_target_properties(dispatch_examples
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(dispatch_examples PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(dispatch_examples PUBLIC
        dispatch
        ${TORCH_LIBRARIES}
    )

    target_compile_options(dispatch_examples PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -Xcompiler=-O0>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0>
        $<$<COMPILE_LANGUAGE:CXX>:
            "-Wall"
            "-Werror"
            "-fdiagnostics-color=always"
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            "--extended-lambda"
            "-Xfatbin=-compress-all"
            "-Werror=all-warnings"
            "-Xcompiler=-Wall,-Werror"
            ${TORCH_CUDA_COMMON_FLAGS}
        >
    )
endif()

# ============================================================================
# Tests (only built when tests are enabled)
# ============================================================================
if(FVDB_BUILD_TESTS)
    # Get GTest via CPM (same as main tests)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_google_test.cmake)

    enable_testing()
    include(GoogleTest)

    set(TEST_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gtests/dispatch>")

    function(ConfigureDispatchTest CMAKE_TEST_NAME)
        add_library(${CMAKE_TEST_NAME}_obj OBJECT ${ARGN})

        set_target_properties(${CMAKE_TEST_NAME}_obj
            PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )

        target_include_directories(${CMAKE_TEST_NAME}_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            $<TARGET_PROPERTY:dispatch,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:dispatch_examples,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:GTest::gtest_main,INTERFACE_INCLUDE_DIRECTORIES>
            ${TORCH_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}/cccl
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
                "-diag-suppress=3189"
                ${TORCH_CUDA_COMMON_FLAGS}
            >
        )

        add_executable(${CMAKE_TEST_NAME} $<TARGET_OBJECTS:${CMAKE_TEST_NAME}_obj>)

        set_target_properties(${CMAKE_TEST_NAME}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIRECTORY}
            INSTALL_RPATH "\$ORIGIN/../../../lib"
        )

        target_link_options(${CMAKE_TEST_NAME} PRIVATE
            $<$<CONFIG:Debug>:-rdynamic>
        )

        target_link_libraries(${CMAKE_TEST_NAME}
            dispatch
            dispatch_examples
            ${TORCH_LIBRARIES}
            GTest::gtest
            GTest::gtest_main
            $<TARGET_NAME_IF_EXISTS:conda_env>
        )

        add_test(NAME ${CMAKE_TEST_NAME}
                 COMMAND ${CMAKE_TEST_NAME}
                 WORKING_DIRECTORY ${TEST_BINARY_DIRECTORY})

        install(
            TARGETS ${CMAKE_TEST_NAME}
            COMPONENT testing
            DESTINATION bin/gtests/dispatch
            EXCLUDE_FROM_ALL
        )
    endfunction()

    # Configure dispatch tests
    ConfigureDispatchTest(DispatchTypesTest tests/types_test.cpp)
    ConfigureDispatchTest(DispatchAxesMapTest tests/axes_map_test.cpp)
    ConfigureDispatchTest(DispatchTableTest tests/dispatch_table_test.cpp)
    ConfigureDispatchTest(DispatchVisitSpacesTest tests/visit_spaces_test.cpp)
    ConfigureDispatchTest(DispatchTorchTest tests/torch_test.cpp)
    ConfigureDispatchTest(DispatchNvccSmokeTest tests/nvcc_smoke_test.cu)
    ConfigureDispatchTest(DispatchScanLibHostTest tests/scan_lib_host_test.cpp)
    ConfigureDispatchTest(DispatchScanLibCudaTest tests/scan_lib_cuda_test.cu)
    ConfigureDispatchTest(DispatchInclusiveScanExamplesTest tests/inclusive_scan_examples_test.cpp)
    ConfigureDispatchTest(DispatchReLUTest tests/relu_test.cpp)
    ConfigureDispatchTest(DispatchGeLUForEachTest tests/gelu_for_each_test.cpp)
    ConfigureDispatchTest(DispatchForEachTest tests/for_each_test.cu)
    ConfigureDispatchTest(DispatchGatherScatterTest tests/gather_scatter_test.cu)
    ConfigureDispatchTest(DispatchViewsTest tests/views_test.cu)

    # ============================================================================
    # Negative Compilation Tests (Expected Failures)
    # ============================================================================

    function(ConfigureCompileFailTest TEST_NAME DEFINE_NAME)
        set(FULL_NAME "DispatchCompileFail_${TEST_NAME}")

        add_library(${FULL_NAME}_obj OBJECT EXCLUDE_FROM_ALL
            tests/compile_errors/dispatch_compile_errors.cpp
        )

        target_compile_definitions(${FULL_NAME}_obj PRIVATE ${DEFINE_NAME})

        set_target_properties(${FULL_NAME}_obj PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )

        target_include_directories(${FULL_NAME}_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )

        add_test(
            NAME ${FULL_NAME}
            COMMAND ${CMAKE_COMMAND}
                --build ${CMAKE_BINARY_DIR}
                --target ${FULL_NAME}_obj
                --config $<CONFIG>
        )

        set_tests_properties(${FULL_NAME} PROPERTIES
            WILL_FAIL TRUE
            LABELS "compile_fail"
        )
    endfunction()

    ConfigureCompileFailTest(MixedAxisTypes      TEST_MIXED_AXIS_TYPES)
    ConfigureCompileFailTest(SubspaceNotWithin   TEST_SUBSPACE_NOT_WITHIN)
    ConfigureCompileFailTest(OpMissingOverload   TEST_OP_MISSING_OVERLOAD)
    ConfigureCompileFailTest(WrongTupleType      TEST_WRONG_TUPLE_TYPE)
endif()
