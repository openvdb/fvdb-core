// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/jagged/JaggedReductions.h>

#include <ATen/Dispatch.h>

#include <limits>

namespace {

/// @brief Expand a 1-D index tensor so it can be used with scatter_reduce_ on
///        a multi-dimensional data tensor.
///
/// Reshapes @p idx1d from [N] to [N, 1, 1, ...] and broadcasts it to match
/// the full shape of @p data. The result is cast to kLong.
///
/// @param idx1d 1-D index tensor of length N (e.g. batch indices)
/// @param data  The data tensor whose shape the index should match
/// @return A kLong tensor with the same shape as @p data, suitable for
///         scatter_reduce_ along dimension 0
torch::Tensor
broadcastIdxToMatchData(const torch::Tensor &idx1d, const torch::Tensor &data) {
    torch::Tensor idx = idx1d.to(torch::kLong);
    if (data.dim() > 1) {
        std::vector<int64_t> viewShape(data.dim(), 1);
        viewShape[0] = -1;
        idx          = idx.view(viewShape).expand_as(data);
    }
    return idx;
}

/// @brief Return the identity element for a minimum reduction over the given dtype.
///
/// For floating-point types this is +infinity. For integral types it is the
/// largest representable value (e.g. INT32_MAX for kInt). Bool is not supported
/// and must be rejected by the caller.
///
/// @param dtype Scalar type of the tensor being reduced
/// @return A Scalar that, when used as the initial value for scatter_reduce_
///         with "amin", acts as a neutral element
torch::Scalar
minIdentity(at::ScalarType dtype) {
    if (at::isFloatingType(dtype)) {
        return std::numeric_limits<double>::infinity();
    }
    torch::Scalar val;
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "jmin_identity", [&] {
        val = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
    });
    return val;
}

/// @brief Return the identity element for a maximum reduction over the given dtype.
///
/// For floating-point types this is -infinity. For integral types it is the
/// smallest representable value (e.g. INT32_MIN for kInt). Bool is not supported
/// and must be rejected by the caller.
///
/// @param dtype Scalar type of the tensor being reduced
/// @return A Scalar that, when used as the initial value for scatter_reduce_
///         with "amax", acts as a neutral element
torch::Scalar
maxIdentity(at::ScalarType dtype) {
    if (at::isFloatingType(dtype)) {
        return -std::numeric_limits<double>::infinity();
    }
    torch::Scalar val;
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "jmax_identity", [&] {
        val = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
    });
    return val;
}

/// @brief Compute a boolean mask indicating which groups contain zero elements.
///
/// Derives per-group sizes from consecutive differences in @p offsets and
/// returns a mask that is @c true for empty groups. The mask is shaped
/// [numGroups, 1, 1, ...] (with @p outDim total dimensions) so it can be
/// broadcast directly with the reduction output tensor via torch::where.
///
/// @param offsets  The JaggedTensor offsets tensor of length numGroups + 1
/// @param numGroups Number of groups (i.e. num_tensors())
/// @param outDim   Dimensionality of the reduction output tensor
/// @return A boolean tensor broadcastable with shape [numGroups, ...]
torch::Tensor
emptyGroupMask(const torch::Tensor &offsets, int64_t numGroups, int64_t outDim) {
    torch::Tensor sizes = offsets.slice(0, 1, numGroups + 1) - offsets.slice(0, 0, numGroups);
    torch::Tensor mask  = (sizes == 0);
    if (outDim > 1) {
        std::vector<int64_t> viewShape(outDim, 1);
        viewShape[0] = numGroups;
        mask         = mask.view(viewShape);
    }
    return mask;
}

} // anonymous namespace

namespace fvdb {
namespace detail {
namespace ops {

JaggedTensor
jaggedSum(const JaggedTensor &jt, int64_t dim, bool keepdim) {
    const torch::Tensor &data     = jt.jdata();
    const torch::Tensor &batchIdx = jt.jidx();
    const torch::Tensor &offsets  = jt.joffsets();
    const torch::Tensor &listIdx  = jt.jlidx();
    const int64_t jdim            = data.dim();

    TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim < jdim,
                      "dim must be between ",
                      -(jdim - 1),
                      " and ",
                      jdim - 1,
                      " inclusive");
    if (dim < 0) {
        dim += jdim;
    }

    if (dim == 0) {
        torch::Tensor retData;
        if (batchIdx.size(0) == 0) {
            retData = data.sum(0).unsqueeze(0);
        } else {
            torch::Tensor idx = broadcastIdxToMatchData(batchIdx, data);
            auto outShape     = data.sizes().vec();
            outShape[0]       = jt.num_tensors();
            retData           = torch::zeros(outShape, data.options());
            retData.scatter_reduce_(0, idx, data, "sum", /*include_self=*/true);
        }
        const torch::Tensor retOffsets = torch::arange(
            0,
            retData.size(0) + 1,
            torch::TensorOptions().dtype(JOffsetsScalarType).device(retData.device()));
        const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0));

        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            retData, retOffsets, retJidx, listIdx, jt.num_outer_lists());
    } else {
        return jt.jagged_like(data.sum(dim, keepdim));
    }
}

std::vector<JaggedTensor>
jaggedMin(const JaggedTensor &jt, int64_t dim, bool keepdim) {
    const torch::Tensor &data     = jt.jdata();
    const torch::Tensor &batchIdx = jt.jidx();
    const torch::Tensor &offsets  = jt.joffsets();
    const torch::Tensor &listIdx  = jt.jlidx();
    const int64_t jdim            = data.dim();

    TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim <= jdim,
                      "dim must be between ",
                      -(jdim - 1),
                      " and ",
                      jdim - 1,
                      " inclusive");
    if (dim < 0) {
        dim += jdim;
    }

    TORCH_CHECK(data.scalar_type() != at::kBool, "jmin does not support bool dtype");

    if (dim == 0) {
        torch::Tensor minVals, minIndices;
        if (batchIdx.size(0) == 0) {
            auto minTuple = data.min(0);
            minVals       = std::get<0>(minTuple).unsqueeze(0);
            minIndices    = std::get<1>(minTuple).unsqueeze(0);
        } else {
            torch::Tensor idx = broadcastIdxToMatchData(batchIdx, data);
            auto outShape     = data.sizes().vec();
            outShape[0]       = jt.num_tensors();

            minVals = torch::full(outShape, minIdentity(data.scalar_type()), data.options());
            minVals.scatter_reduce_(0, idx, data, "amin", /*include_self=*/false);
            torch::Tensor emptyMask = emptyGroupMask(offsets, jt.num_tensors(), minVals.dim());
            minVals                 = torch::where(emptyMask, torch::zeros_like(minVals), minVals);

            torch::Tensor gatheredMin = minVals.detach().index({batchIdx.to(torch::kLong)});
            torch::Tensor matches     = (data.detach() == gatheredMin);
            torch::Tensor globalPos   = torch::arange(
                data.size(0), torch::TensorOptions().dtype(torch::kLong).device(data.device()));
            torch::Tensor baseOff  = offsets.index({batchIdx.to(torch::kLong)}).to(torch::kLong);
            torch::Tensor localPos = broadcastIdxToMatchData(globalPos - baseOff, data);
            torch::Tensor maskedPos =
                torch::where(matches, localPos, torch::full_like(localPos, -1));

            minIndices =
                torch::full(outShape,
                            (int64_t)-1,
                            torch::TensorOptions().dtype(torch::kLong).device(data.device()));
            minIndices.scatter_reduce_(0, idx, maskedPos, "amax", /*include_self=*/false);
        }

        const torch::Tensor retOffsets = torch::arange(
            0,
            minVals.size(0) + 1,
            torch::TensorOptions().dtype(JOffsetsScalarType).device(minVals.device()));
        const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, minVals.size(0));

        JaggedTensor retVals = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            minVals, retOffsets, retJidx, listIdx, jt.num_outer_lists());
        JaggedTensor retIdxs = retVals.jagged_like(minIndices);
        return {retVals, retIdxs};
    } else {
        auto minTuple            = data.min(dim, keepdim);
        torch::Tensor minVals    = std::get<0>(minTuple);
        torch::Tensor minIndices = std::get<1>(minTuple);
        return {jt.jagged_like(minVals), jt.jagged_like(minIndices)};
    }
}

std::vector<JaggedTensor>
jaggedMax(const JaggedTensor &jt, int64_t dim, bool keepdim) {
    const torch::Tensor &data     = jt.jdata();
    const torch::Tensor &batchIdx = jt.jidx();
    const torch::Tensor &offsets  = jt.joffsets();
    const torch::Tensor &listIdx  = jt.jlidx();
    const int64_t jdim            = data.dim();

    TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim <= jdim,
                      "dim must be between ",
                      -(jdim - 1),
                      " and ",
                      jdim - 1,
                      " inclusive");
    if (dim < 0) {
        dim += jdim;
    }

    TORCH_CHECK(data.scalar_type() != at::kBool, "jmax does not support bool dtype");

    if (dim == 0) {
        torch::Tensor maxVals, maxIndices;
        if (batchIdx.size(0) == 0) {
            auto maxTuple = data.max(0);
            maxVals       = std::get<0>(maxTuple).unsqueeze(0);
            maxIndices    = std::get<1>(maxTuple).unsqueeze(0);
        } else {
            torch::Tensor idx = broadcastIdxToMatchData(batchIdx, data);
            auto outShape     = data.sizes().vec();
            outShape[0]       = jt.num_tensors();

            maxVals = torch::full(outShape, maxIdentity(data.scalar_type()), data.options());
            maxVals.scatter_reduce_(0, idx, data, "amax", /*include_self=*/false);
            torch::Tensor emptyMask = emptyGroupMask(offsets, jt.num_tensors(), maxVals.dim());
            maxVals                 = torch::where(emptyMask, torch::zeros_like(maxVals), maxVals);

            torch::Tensor gatheredMax = maxVals.detach().index({batchIdx.to(torch::kLong)});
            torch::Tensor matches     = (data.detach() == gatheredMax);
            torch::Tensor globalPos   = torch::arange(
                data.size(0), torch::TensorOptions().dtype(torch::kLong).device(data.device()));
            torch::Tensor baseOff  = offsets.index({batchIdx.to(torch::kLong)}).to(torch::kLong);
            torch::Tensor localPos = broadcastIdxToMatchData(globalPos - baseOff, data);
            torch::Tensor maskedPos =
                torch::where(matches, localPos, torch::full_like(localPos, -1));

            maxIndices =
                torch::full(outShape,
                            (int64_t)-1,
                            torch::TensorOptions().dtype(torch::kLong).device(data.device()));
            maxIndices.scatter_reduce_(0, idx, maskedPos, "amax", /*include_self=*/false);
        }

        const torch::Tensor retOffsets = torch::arange(
            0,
            maxVals.size(0) + 1,
            torch::TensorOptions().dtype(JOffsetsScalarType).device(maxVals.device()));
        const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, maxVals.size(0));
        JaggedTensor retVals        = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            maxVals, retOffsets, retJidx, listIdx, jt.num_outer_lists());
        JaggedTensor retIdxs = retVals.jagged_like(maxIndices);
        return {retVals, retIdxs};
    } else {
        auto maxTuple            = data.max(dim, keepdim);
        torch::Tensor maxVals    = std::get<0>(maxTuple);
        torch::Tensor maxIndices = std::get<1>(maxTuple);
        return {jt.jagged_like(maxVals), jt.jagged_like(maxIndices)};
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
