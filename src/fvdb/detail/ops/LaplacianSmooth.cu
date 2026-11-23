// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/VbmCache.h>
#include <fvdb/detail/ops/LaplacianSmooth.h>
#include <fvdb/detail/utils/cuda/ForEachVbmCUDA.cuh>

#include <nanovdb/NanoVDB.h>

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <vector>

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Batch-wide row of a face neighbour, or -1 if it is inactive. `valueIndex` is the neighbour's
// 1-based value index in its grid (0 = inactive); `rowBase` is the grid's voxel offset minus one.
__device__ inline int64_t
faceRow(uint64_t valueIndex, int64_t rowBase) {
    return valueIndex ? rowBase + int64_t(valueIndex) : int64_t(-1);
}

// One umbrella-Laplacian pass, out = in + weight * (faceMean - in), per channel. Double-buffered
// (in != out) so neighbour reads see the pre-pass field.
template <typename ScalarT> struct LaplacianPass {
    BatchVbmLayout layout;
    const ScalarT *in;
    ScalarT *out;
    int64_t channels;
    ScalarT weight;
    const ScalarT *boundary; // one value per grid (sign continuation), or nullptr (zero flux)

    __device__ void
    perActiveVoxel(uint32_t gridIdx, const nanovdb::Coord &ijk, int64_t featureIdx) const {
        auto accessor               = layout.acc.grid(gridIdx)->getAccessor();
        const int64_t base          = layout.voxelOffsets[gridIdx] - 1;
        const int64_t row[6]        = {faceRow(accessor.getValue(ijk.offsetBy(-1, 0, 0)), base),
                                       faceRow(accessor.getValue(ijk.offsetBy(1, 0, 0)), base),
                                       faceRow(accessor.getValue(ijk.offsetBy(0, -1, 0)), base),
                                       faceRow(accessor.getValue(ijk.offsetBy(0, 1, 0)), base),
                                       faceRow(accessor.getValue(ijk.offsetBy(0, 0, -1)), base),
                                       faceRow(accessor.getValue(ijk.offsetBy(0, 0, 1)), base)};
        const ScalarT boundaryValue = boundary ? boundary[gridIdx] : ScalarT(0);
        for (int64_t c = 0; c < channels; ++c) {
            const ScalarT center = in[featureIdx * channels + c];
            const ScalarT inactive =
                boundary ? (center < ScalarT(0) ? -boundaryValue : boundaryValue) : center;
            // Summed in face order (-x,+x,-y,+y,-z,+z), then scaled by 1/6.
            ScalarT sum = ScalarT(0);
            for (int k = 0; k < 6; ++k)
                sum += row[k] >= 0 ? in[row[k] * channels + c] : inactive;
            const ScalarT faceMean         = sum * (ScalarT(1) / ScalarT(6));
            out[featureIdx * channels + c] = center + weight * (faceMean - center);
        }
    }
};

} // namespace

JaggedTensor
laplacianSmooth(const GridBatchData &batchHdl,
                const JaggedTensor &field,
                int iterations,
                SmoothingMode mode,
                const std::optional<torch::Tensor> &boundaryValues) {
    TORCH_CHECK_VALUE(
        field.ldim() == 1,
        "Expected field to have 1 list dimension (a single list of per-voxel values)");
    TORCH_CHECK_TYPE(field.is_floating_point(), "field must have a floating point type");
    const torch::Tensor &data = field.jdata();
    TORCH_CHECK_VALUE(data.dim() == 1 || data.dim() == 2,
                      "field must have shape (N,) or (N, C), got ",
                      data.sizes());
    TORCH_CHECK_VALUE(data.size(0) == batchHdl.totalVoxels(),
                      "field must have one row per voxel, got ",
                      data.size(0),
                      " rows for ",
                      batchHdl.totalVoxels(),
                      " voxels");
    TORCH_CHECK_VALUE(field.num_outer_lists() == batchHdl.batchSize(),
                      "field batch size does not match the grid batch size");
    TORCH_CHECK_VALUE(iterations >= 0, "iterations must be >= 0");
    batchHdl.checkDevice(field);
    TORCH_CHECK(field.device().is_cuda(), "laplacian_smooth currently requires a CUDA device");
    TORCH_CHECK_TYPE(data.scalar_type() == torch::kFloat32 || data.scalar_type() == torch::kFloat64,
                     "laplacian_smooth supports float32 or float64 fields");

    torch::Tensor boundary;
    if (boundaryValues.has_value()) {
        const torch::Tensor &values = boundaryValues.value();
        TORCH_CHECK_VALUE(values.dim() == 1 && values.size(0) == batchHdl.batchSize(),
                          "boundary values must be a 1-D tensor with one value per grid, got "
                          "shape ",
                          values.sizes(),
                          " for ",
                          batchHdl.batchSize(),
                          " grids");
        TORCH_CHECK_VALUE(values.is_floating_point(), "boundary values must be floating point");
        if (values.device().is_cpu()) {
            // Validate on the host and upload asynchronously from pinned memory, so the common
            // case (values passed from Python) adds no host-device synchronization.
            const torch::Tensor host = values.to(data.scalar_type()).contiguous();
            TORCH_CHECK_VALUE((torch::isfinite(host) & (host >= 0)).all().item<bool>(),
                              "boundary values must be finite and non-negative");
            boundary = host.pin_memory().to(data.device(), /*non_blocking=*/true);
        } else {
            TORCH_CHECK_VALUE(values.device() == data.device(),
                              "boundary values must be on the host or the field's device");
            boundary = values.to(data.scalar_type()).contiguous();
            if (boundary.numel() > 0) {
                TORCH_CHECK_VALUE((torch::isfinite(boundary) & (boundary >= 0)).all().item<bool>(),
                                  "boundary values must be finite and non-negative");
            }
        }
    }

    // Per-pass weights, applied in order.
    std::vector<double> weights;
    for (int i = 0; i < iterations; ++i) {
        if (mode == SmoothingMode::TAUBIN) {
            weights.push_back(0.5);   // shrink
            weights.push_back(-0.53); // inflate slightly more, so volume is preserved
        } else {
            weights.push_back(1.0);
        }
    }

    const torch::Tensor input = data.contiguous();
    torch::Tensor out         = torch::empty_like(input);
    const int64_t channels    = input.dim() == 2 ? input.size(1) : 1;
    if (weights.empty() || input.numel() == 0) {
        out.copy_(input);
        return field.jagged_like(out);
    }

    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    // Ping-pong between `out` and one scratch buffer, choosing the first destination so the last
    // pass lands in `out`. The first pass reads the input directly, so a single pass needs no
    // scratch.
    const int64_t numPasses     = int64_t(weights.size());
    torch::Tensor scratch       = numPasses > 1 ? torch::empty_like(input) : torch::Tensor();
    const BatchVbmLayout layout = BatchVbmLayout::of(batchHdl);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "laplacianSmooth", [&] {
        const scalar_t *boundaryPtr = boundary.defined() ? boundary.data_ptr<scalar_t>() : nullptr;
        const scalar_t *source      = input.data_ptr<scalar_t>();
        for (int64_t pass = 0; pass < numPasses; ++pass) {
            scalar_t *destination = ((numPasses - 1 - pass) % 2 == 0)
                                        ? out.data_ptr<scalar_t>()
                                        : scratch.data_ptr<scalar_t>();
            forEachActiveVoxelInGridVbmCUDA(
                batchHdl,
                LaplacianPass<scalar_t>{
                    layout, source, destination, channels, scalar_t(weights[pass]), boundaryPtr});
            source = destination;
        }
    });
    return field.jagged_like(out);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
