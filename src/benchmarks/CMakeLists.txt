# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

add_custom_command(
  OUTPUT FVDB_BENCHMARKS
  COMMAND echo Running benchmarks
  #COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running fvdb benchmarks."
  USES_TERMINAL
)

set(BENCHMARKS_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gbenchmarks>")

# ============================================================================
# Common benchmark configuration
# ============================================================================
# All dispatch benchmarks share this setup. Link dispatch (PUBLIC OpenMP flags
# propagate automatically) and local headers for retired pool baselines.

function(ConfigureDispatchBench CMAKE_BENCH_NAME)
    add_executable(${CMAKE_BENCH_NAME} ${ARGN})

    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BENCHMARKS_BINARY_DIRECTORY}
        INSTALL_RPATH "\$ORIGIN/../../../lib"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(${CMAKE_BENCH_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatch
    )

    target_link_libraries(${CMAKE_BENCH_NAME}
        dispatch
        fvdb
        ${TORCH_LIBRARIES}
        benchmark::benchmark_main
        $<TARGET_NAME_IF_EXISTS:conda_env>
    )

    target_compile_options(${CMAKE_BENCH_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            "-Wall"
            "-Werror"
            "-Wno-unknown-pragmas"
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

    add_custom_command(
        OUTPUT FVDB_BENCHMARKS
        COMMAND ${CMAKE_BENCH_NAME} --benchmark_out_format=json
                --benchmark_out=results/${CMAKE_BENCH_NAME}.json
        APPEND
        COMMENT "Adding ${CMAKE_BENCH_NAME}"
    )

    install(
        TARGETS ${CMAKE_BENCH_NAME}
        COMPONENT testing
        DESTINATION bin/benchmarks/fvdb
        EXCLUDE_FROM_ALL
    )
endfunction()

# ============================================================================
# Legacy fvdb benchmarks (simple)
# ============================================================================

function(ConfigureBench CMAKE_BENCH_NAME)
    add_executable(${CMAKE_BENCH_NAME} ${ARGN})
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BENCHMARKS_BINARY_DIRECTORY}
        INSTALL_RPATH "\$ORIGIN/../../../lib"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(${CMAKE_BENCH_NAME}
        fvdb
        ${TORCH_LIBRARIES}
        benchmark::benchmark_main
        $<TARGET_NAME_IF_EXISTS:conda_env>
    )
    add_custom_command(
        OUTPUT FVDB_BENCHMARKS
        COMMAND ${CMAKE_BENCH_NAME} --benchmark_out_format=json
                --benchmark_out=results/${CMAKE_BENCH_NAME}.json
        APPEND
        COMMENT "Adding ${CMAKE_BENCH_NAME}"
    )
    install(
        TARGETS ${CMAKE_BENCH_NAME}
        COMPONENT testing
        DESTINATION bin/benchmarks/fvdb
        EXCLUDE_FROM_ALL
    )
endfunction()

ConfigureBench(simple "simple/simple.cpp")

# ============================================================================
# Dispatch benchmarks
# ============================================================================

# CPU pool comparison — all pool variants with SIMD GELU workload
ConfigureDispatchBench(cpu_pool_comparison
    dispatch/cpu_pool_comparison.cpp
    dispatch/omp_gelu.cpp
)

# Synthetic voxel benchmark — uniform and unbalanced compute-bound workloads
ConfigureDispatchBench(synthetic_voxel_benchmark
    dispatch/synthetic_voxel_benchmark.cpp
)

# for_each + views benchmark — contiguous/strided x CPU/CUDA x float32/float64
ConfigureDispatchBench(for_each_benchmark
    dispatch/for_each_benchmark.cu
)
