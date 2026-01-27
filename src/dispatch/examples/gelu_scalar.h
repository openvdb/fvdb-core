// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DISPATCH_EXAMPLES_GELU_SCALAR_H
#define DISPATCH_EXAMPLES_GELU_SCALAR_H

#include "examples/common.h"

#include <torch/types.h>

#include <cmath>
#include <numbers>

namespace dispatch_examples {

// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))

__hostdev__ inline float
gelu_scalar(float x) {
#ifdef __CUDA_ARCH__
    return 0.5f * x * (1.0f + ::erff(x / std::numbers::sqrt2_v<float>));
#else
    return 0.5f * x * (1.0f + std::erf(x / std::numbers::sqrt2_v<float>));
#endif
}

__hostdev__ inline double
gelu_scalar(double x) {
#ifdef __CUDA_ARCH__
    return 0.5 * x * (1.0 + ::erf(x / std::numbers::sqrt2));
#else
    return 0.5 * x * (1.0 + std::erf(x / std::numbers::sqrt2));
#endif
}

__hostdev__ inline at::Half
gelu_scalar(at::Half x) {
    return at::Half(gelu_scalar(static_cast<float>(x)));
}

__hostdev__ inline at::BFloat16
gelu_scalar(at::BFloat16 x) {
    return at::BFloat16(gelu_scalar(static_cast<float>(x)));
}

} // namespace dispatch_examples

#endif // DISPATCH_EXAMPLES_GELU_SCALAR_H
