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
    JaggedAccessor<int32_t, 1> outCounts) {

        // Kernel parallelizes over rays.
        // For each ray, we count the number of samples we will take.
    }

} // anonymous namespace

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

    const JaggedTensor outputSamples = JaggedTensor::empty({rayOrigins.size(0), rayOrigins.size(1), 2}, rayOrigins.options());
    const JaggedTensor outputPositions = JaggedTensor::empty({rayOrigins.size(0), rayOrigins.size(1), 3}, rayOrigins.options());
    const JaggedTensor outputTimes = JaggedTensor::empty({rayOrigins.size(0), rayOrigins.size(1)}, rayOrigins.options());

    // 1. Count the number of samples we will take for each ray.

    // 2. Generate the samples.

    return {outputSamples, outputPositions, outputTimes};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
