# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(fvdb LANGUAGES CXX CUDA)

# Build options - must be defined before subdirectories that use them
option(FVDB_BUILD_TESTS "Configure CMake to build tests" ON)
option(FVDB_BUILD_BENCHMARKS "Configure CMake to build (google & nvbench) benchmarks" OFF)
option(FVDB_STRIP_SYMBOLS "Strip symbols from the build" OFF)
option(FVDB_LINEINFO "Enable lineinfo in the build" OFF)

message(STATUS "FVDB: Configure CMake to build tests: ${FVDB_BUILD_TESTS}")
message(STATUS "FVDB: Configure CMake to build (google & nvbench) benchmarks: ${FVDB_BUILD_BENCHMARKS}")
message(STATUS "FVDB_STRIP_SYMBOLS: ${FVDB_STRIP_SYMBOLS}")

# Get dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_tinyply.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_cutlass.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_nanovdb.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_blosc.cmake)

# zlib is required for NANOVDB_USE_ZIP (reading ZIP-compressed .nvdb files).
find_package(ZLIB REQUIRED)

# Enable testing at this level so CTest discovers tests from all subdirectories
enable_testing()

# Add dispatch interface library (header-only)
add_subdirectory(dispatch)


# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink the latest compile_commands.json from the build dir to the source dir for clangd
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

# Source files
set(FVDB_CPP_FILES
    fvdb/Config.cpp
    fvdb/detail/autograd/VolumeRender.cpp
    fvdb/detail/io/GaussianPlyIO.cpp
    fvdb/detail/io/LoadNanovdb.cpp
    fvdb/detail/utils/cuda/Prefetch.cpp
    fvdb/detail/io/SaveNanoVDB.cpp
    fvdb/detail/ops/jagged/JaggedReductions.cpp
    fvdb/TorchDeviceBuffer.cpp
    fvdb/detail/viewer/GaussianSplat3dView.cpp
    fvdb/detail/viewer/Viewer.cpp
    fvdb/FVDB.cpp
    fvdb/JaggedTensor.cpp
)

set(FVDB_CU_FILES
    fvdb/GridBatchData.cu
    fvdb/detail/GridBatchDataFactory.cu
    fvdb/detail/ops/ActiveGridCoords.cu
    fvdb/detail/ops/ActiveVoxelsInBoundsMask.cu
    fvdb/detail/ops/BuildCoarseGridFromFine.cu
    fvdb/detail/ops/BuildDenseGrid.cu
    fvdb/detail/ops/BuildDilatedGrid.cu
    fvdb/detail/ops/BuildFineGridFromCoarse.cu
    fvdb/detail/ops/BuildGridForConv.cu
    fvdb/detail/ops/BuildGridForConvTranspose.cu
    fvdb/detail/ops/BuildGridFromIjk.cu
    fvdb/detail/ops/BuildGridFromMesh.cu
    fvdb/detail/ops/BuildGridFromNearestVoxelsToPoints.cu
    fvdb/detail/ops/BuildGridFromPoints.cu
    fvdb/detail/ops/BuildMergedGrids.cu
    fvdb/detail/ops/BuildPaddedGrid.cu
    fvdb/detail/ops/BuildPrunedGrid.cu
    fvdb/detail/ops/ClipGrid.cu
    fvdb/detail/ops/CloneGrid.cu
    fvdb/detail/ops/ConcatenateGrids.cu
    fvdb/detail/ops/IndexGrid.cu
    fvdb/detail/ops/MakeContiguous.cu
    fvdb/detail/ops/SerializeGrid.cu
    fvdb/detail/ops/CoarseIjkForFineGrid.cu
    fvdb/detail/ops/convolution/GatherScatterDefault.cu
    fvdb/detail/ops/convolution/PredGatherIGemm.cu
    fvdb/detail/ops/CoordsInGrid.cu
    fvdb/detail/ops/CubesInGrid.cu
    fvdb/detail/ops/AvgPool.cu
    fvdb/detail/ops/MaxPool.cu
    fvdb/detail/ops/GridEdgeNetwork.cu
    fvdb/detail/ops/gsplat/FusedSSIM.cu
    fvdb/detail/ops/gsplat/ComputeGaussianNanInfMask.cu
    fvdb/detail/ops/gsplat/AddNoiseToGaussianMeans.cu
    fvdb/detail/ops/gsplat/RelocateGaussians.cu
    fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticBackward.cu
    fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticForward.cu
    fvdb/detail/ops/gsplat/ProjectGaussiansUnscentedForward.cu
    fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedBackward.cu
    fvdb/detail/ops/gsplat/ProjectGaussiansAnalyticJaggedForward.cu
    fvdb/detail/ops/gsplat/RasterizeScreenSpaceGaussiansBackward.cu
    fvdb/detail/ops/gsplat/RasterizeScreenSpaceGaussiansForward.cu
    fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansBackward.cu
    fvdb/detail/ops/gsplat/RasterizeWorldSpaceGaussiansForward.cu
    fvdb/detail/ops/gsplat/CountContributingGaussians.cu
    fvdb/detail/ops/gsplat/IdentifyTopContributingGaussians.cu
    fvdb/detail/ops/gsplat/IdentifyContributingGaussians.cu
    fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsBackward.cu
    fvdb/detail/ops/gsplat/EvaluateSphericalHarmonicsForward.cu
    fvdb/detail/ops/gsplat/BuildSparseGaussianTileLayout.cu
    fvdb/detail/ops/gsplat/IntersectGaussianTiles.cu
    fvdb/detail/ops/IjkForMesh.cu
    fvdb/detail/ops/IjkToIndex.cu
    fvdb/detail/ops/IjkToInvIndex.cu
    fvdb/detail/ops/Inject.cu
    fvdb/detail/ops/IntegrateTSDF.cu
    fvdb/detail/ops/jagged/JaggedSort.cu
    fvdb/detail/ops/JaggedTensorIndex.cu
    fvdb/detail/ops/JCat0.cu
    fvdb/detail/ops/JIdxForGrid.cu
    fvdb/detail/ops/JIdxForJOffsets.cu
    fvdb/detail/ops/JOffsetsFromJIdx.cu
    fvdb/detail/ops/MarchingCubes.cu
    fvdb/detail/ops/MortonHilbertFromIjk.cu
    fvdb/detail/ops/NearestIjkForPoints.cu
    fvdb/detail/ops/PointsInGrid.cu
    fvdb/detail/ops/PopulateGridMetadata.cu
    fvdb/detail/ops/RayImplicitIntersection.cu
    fvdb/detail/ops/InjectFromDense.cu
    fvdb/detail/ops/InjectToDense.cu
    fvdb/detail/ops/SampleBezier.cu
    fvdb/detail/ops/SampleBezierWithGrad.cu
    fvdb/detail/ops/SampleBezierWithGradBackward.cu
    fvdb/detail/ops/SampleNearest.cu
    fvdb/detail/ops/SampleTrilinear.cu
    fvdb/detail/ops/SampleTrilinearWithGrad.cu
    fvdb/detail/ops/SampleTrilinearWithGradBackward.cu
    fvdb/detail/ops/SampleRaysUniform.cu
    fvdb/detail/ops/SegmentsAlongRays.cu
    fvdb/detail/ops/SerializeEncode.cu
    fvdb/detail/ops/SplatBezier.cu
    fvdb/detail/ops/SplatTrilinear.cu
    fvdb/detail/ops/VoxelToWorld.cu
    fvdb/detail/ops/Refine.cu
    fvdb/detail/ops/VolumeRender.cu
    fvdb/detail/ops/NeighborIndexes.cu
    fvdb/detail/ops/VoxelsAlongRays.cu
    fvdb/detail/utils/cuda/ForEachCUDA.cu
)

# Build library
add_library(fvdb SHARED ${FVDB_CPP_FILES} ${FVDB_CU_FILES})

# Target properties
set_target_properties(
    fvdb
    PROPERTIES BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    INTERFACE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "fvdb: TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

target_include_directories(fvdb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${nanovdb_SOURCE_DIR}/nanovdb>
    $<BUILD_INTERFACE:${nvtx3_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${NANOVDB_EDITOR_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)
foreach(_torch_inc_dir IN LISTS TORCH_INCLUDE_DIRS)
    target_include_directories(fvdb SYSTEM PUBLIC
        $<BUILD_INTERFACE:${_torch_inc_dir}>)
endforeach()
target_include_directories(fvdb PRIVATE
    ${TORCH_SOURCE_INCLUDE_DIRS}
    ${tinyply_SOURCE_DIR}/source)

# Needed by Torch for CUDA extensions
set(TORCH_CUDA_COMMON_FLAGS
    "-D__CUDA_NO_HALF_OPERATORS__"
    "-D__CUDA_NO_HALF_CONVERSIONS__"
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
    "-D__CUDA_NO_HALF2_OPERATORS__"
    "--expt-relaxed-constexpr")

target_compile_options(fvdb PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -Xcompiler=-O0>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0>
    $<$<COMPILE_LANGUAGE:CXX>:
    "-Wno-unknown-pragmas"
    "-Wno-class-memaccess"
    "-Wno-array-bounds"
    "-Wno-stringop-overflow"
    "-Wno-maybe-uninitialized"
    "-fdiagnostics-color=always"
    "-DNANOVDB_USE_BLOSC"
    "-DNANOVDB_USE_ZIP"
    "-fvisibility=default"
    "-Wall"
    "-Werror"
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
    "-Wno-unknown-pragmas"
    "--extended-lambda"
    "-Xfatbin=-compress-all"
    "-fvisibility=default"
    "-Werror=all-warnings"
    "--threads=$ENV{NVCC_THREADS}"
    "-Xcompiler=-Wall,-Werror,-Wno-maybe-uninitialized"
    ${TORCH_CUDA_COMMON_FLAGS}
    >
    $<$<AND:$<BOOL:${FVDB_LINEINFO}>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

target_link_libraries(fvdb PRIVATE
    ${TORCH_LIBRARIES} tinyply cutlass blosc::blosc ZLIB::ZLIB dispatch)

# Add linker flags for Debug builds to preserve symbols
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_link_options(fvdb PRIVATE "-rdynamic")
endif()

# Strip symbols in non-Debug builds to reduce wheel size
if(NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo") AND FVDB_STRIP_SYMBOLS)
    target_link_options(fvdb PRIVATE "-s")
endif()

# Install rules
# Install of development artifacts (headers and system-level lib) is optional
option(FVDB_INSTALL_DEV "Install headers and system-level lib for development" ON)

if(FVDB_INSTALL_DEV)
    include(CMakePackageConfigHelpers)

    install(TARGETS fvdb
        EXPORT fvdb-targets
        LIBRARY DESTINATION lib)

    file(GLOB _FVDB_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/fvdb/*.h")
    install(FILES ${_FVDB_PUBLIC_HEADERS}
        DESTINATION include/fvdb)

    install(DIRECTORY ${nanovdb_SOURCE_DIR}/nanovdb/nanovdb/
        DESTINATION include/nanovdb
        FILES_MATCHING PATTERN "*.h")

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/dispatch/macros.h
        DESTINATION include/dispatch/dispatch)

    install(EXPORT fvdb-targets
        FILE fvdbTargets.cmake
        NAMESPACE fvdb::
        DESTINATION lib/cmake/fvdb)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fvdbConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/fvdbConfig.cmake
        INSTALL_DESTINATION lib/cmake/fvdb)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fvdbConfig.cmake
        DESTINATION lib/cmake/fvdb)
endif()

# Add tests subdirectory if tests are enabled
if(FVDB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Add benchmarks
if(FVDB_BUILD_BENCHMARKS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_google_benchmark.cmake)
    add_subdirectory(benchmarks)
endif()
