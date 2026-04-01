// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/VoxelsAlongRays.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {
namespace {

template <typename ScalarType>
__global__ void countAcousticRaySamplesKernel(
    BatchGridAccessor batchAccessor,
    const JaggedAccessor<ScalarType, 2> rayOrigins,
    const JaggedAccessor<ScalarType, 2> rayDirections,
    const JaggedAccessor<ScalarType, 1> soundSpeeds,
    float stepSize,
    fvdb::TorchRAcc64<int32_t, 1> outCounts) {

        // Kernel parallelizes over rays.
        // For each ray, we count the number of samples we will take.
}

template <typename ScalarType>
__global__ void accousticRaySamplesKernel(
    BatchGridAccessor batchAccessor,
    const JaggedAccessor<ScalarType, 2> rayOrigins,
    const JaggedAccessor<ScalarType, 2> rayDirections,
    const JaggedAccessor<ScalarType, 1> soundSpeeds,
    float stepSize,
    JaggedAccessor<ScalarType, 2> outSamples) {

        // Kernel parallelizes over rays.
        // For each ray, we take a step, query the sound speeds, write a value, and bend the ray according to Snell's law.
}

} // anonymous namespace

// Generate samples along each ray with a given step size where samples are taken
// by bending the ray according to Snell's law.
std::vector<JaggedTensor> acousticRaySamples(
    const GridBatchImpl &batchHdl,
    const JaggedTensor &rayOrigins,
    const JaggedTensor &rayDirections,
    const JaggedTensor &soundSpeeds,
    float stepSize) {

    const auto gridBatchAcc = gridBatchAccessor<DeviceTag>(batchHdl);
    const auto rayOriginsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayOrigins);
    const auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);
    const auto soundSpeedsAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(soundSpeeds);

    const int64_t N = rayOrigins.jdata().size(0);
    auto outSamplesData = torch::empty({N, 2}, rayOrigins.jdata().options());
    auto outPositionsData = torch::empty({N, 3}, rayOrigins.jdata().options());
    auto outTimesData = torch::empty({N}, rayOrigins.jdata().options());
    fvdb::JaggedTensor outSamples = rayOrigins.jagged_like(outSamplesData);
    fvdb::JaggedTensor outPositions = rayOrigins.jagged_like(outPositionsData);
    fvdb::JaggedTensor outTimes = rayOrigins.jagged_like(outTimesData);

    // 1. Count the number of samples we will take for each ray.

    // 2. Generate the samples.

    return {outSamples, outPositions, outTimes};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
