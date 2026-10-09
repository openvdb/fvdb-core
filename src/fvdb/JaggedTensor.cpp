// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/Config.h>
#include <fvdb/JaggedTensor.h>

// Ops headers
#include <fvdb/detail/ops/JCat0.h>
#include <fvdb/detail/ops/JIdxForJOffsets.h>
#include <fvdb/detail/ops/JOffsetsFromJIdx.h>
#include <fvdb/detail/ops/JaggedStructureCheck.h>
#include <fvdb/detail/ops/JaggedTensorIndex.h>
#include <fvdb/detail/ops/jagged/JaggedReductions.h>
#include <fvdb/detail/ops/jagged/JaggedSort.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fvdb {

namespace {

// Checks that a structure tensor is integral, then casts it to the canonical dtype on its own
// device
torch::Tensor
canonicalStructureTensor(const torch::Tensor &t, c10::ScalarType dtype, const char *name) {
    TORCH_CHECK_VALUE(t.defined(), name, " must be a defined tensor");
    TORCH_CHECK_VALUE(c10::isIntegralType(t.scalar_type(), /*includeBool=*/false),
                      name,
                      " must have an integer dtype, but got ",
                      t.scalar_type());
    return t.to(dtype);
}

// Structure tensors that all start on the host are validated there, which needs no kernel and no
// sync. Otherwise they are validated on the data device.
torch::Device
structureCheckDevice(const torch::Tensor &data, std::initializer_list<torch::Tensor> structure) {
    for (const auto &t: structure) {
        if (!t.device().is_cpu() && t.numel() > 0) {
            return data.device();
        }
    }
    return torch::kCPU;
}

// A JaggedTensor with at most one tensor stores an empty jidx. Element-wise ops compare jidx
// shapes, so every constructor must follow this convention.
torch::Tensor
singleTensorJIdx(const torch::Tensor &indices, int64_t numTensors) {
    return numTensors <= 1 ? torch::empty({0}, indices.options()) : indices;
}

// Checks list id shapes against the tensor count. Returns the outer list count that outer ids must
// stay below, or -1 when the count is inferred from the ids.
int64_t
checkListIdsShape(const torch::Tensor &listIds,
                  int64_t numTensors,
                  std::optional<int64_t> numOuterLists) {
    TORCH_CHECK_VALUE(listIds.dim() == 2 && (listIds.size(1) == 1 || listIds.size(1) == 2),
                      "list_ids must have shape [num_tensors, 1] or [num_tensors, 2], but got ",
                      listIds.sizes());
    const int64_t rows = listIds.size(0);

    if (listIds.size(1) == 1) {
        TORCH_CHECK_VALUE(rows == 0 || rows == numTensors,
                          "list_ids for ldim 1 must have 0 or num_tensors (",
                          numTensors,
                          ") rows, but got ",
                          rows);
        TORCH_CHECK_VALUE(!numOuterLists.has_value() || *numOuterLists == numTensors,
                          "num_outer_lists (",
                          numOuterLists.value_or(0),
                          ") must equal num_tensors (",
                          numTensors,
                          ") for ldim 1");
        return numTensors;
    }

    TORCH_CHECK_VALUE(rows == numTensors,
                      "list_ids for ldim 2 must have num_tensors (",
                      numTensors,
                      ") rows, but got ",
                      rows);
    TORCH_CHECK_VALUE(!numOuterLists.has_value() || *numOuterLists >= 0,
                      "num_outer_lists must be non-negative, but got ",
                      numOuterLists.value_or(0));
    return numOuterLists.value_or(-1);
}

// Validates the structure tensors' values in one pass, which costs one device-to-host copy on CUDA,
// and returns the outer list count. Offsets or indices may be undefined.
int64_t
validateStructure(const torch::Tensor &offsets,
                  int64_t numElements,
                  const torch::Tensor &indices,
                  int64_t numTensors,
                  const torch::Tensor &listIds,
                  std::optional<int64_t> numOuterLists) {
    using namespace detail::ops;

    const int64_t knownOuterLists    = checkListIdsShape(listIds, numTensors, numOuterLists);
    const bool nested                = listIds.size(1) == 2;
    const JaggedStructureCheck check = checkJaggedStructure(offsets,
                                                            numElements,
                                                            indices,
                                                            numTensors,
                                                            nested ? listIds : torch::Tensor(),
                                                            nested ? knownOuterLists : -1);

    const auto failed = [&](JaggedStructureFailure bit) { return (check.failures & bit) != 0; };
    TORCH_CHECK_VALUE(!failed(kOffsetsStart), "offsets must start at 0");
    TORCH_CHECK_VALUE(!failed(kOffsetsEnd),
                      "offsets must end at the number of elements in data (",
                      numElements,
                      ")");
    TORCH_CHECK_VALUE(!failed(kOffsetsDecreasing), "offsets must be non-decreasing");
    TORCH_CHECK_VALUE(!failed(kIndicesNegative), "indices must be non-negative");
    TORCH_CHECK_VALUE(
        !failed(kIndicesTooLarge), "indices must be less than num_tensors (", numTensors, ")");
    TORCH_CHECK_VALUE(!failed(kIndicesDecreasing), "indices must be non-decreasing");
    TORCH_CHECK_VALUE(!failed(kOuterIdsNegative), "outer list ids must be non-negative");
    TORCH_CHECK_VALUE(!failed(kOuterIdsDecreasing), "outer list ids must be non-decreasing");
    TORCH_CHECK_VALUE(!failed(kInnerIdsNotCounting),
                      "inner list ids must count 0, 1, ... within each outer list");
    TORCH_CHECK_VALUE(!failed(kOuterIdsTooLarge),
                      "outer list ids must be less than num_outer_lists (",
                      knownOuterLists,
                      ")");

    if (!nested) {
        return numTensors;
    }
    return knownOuterLists >= 0 ? knownOuterLists : check.lastOuterId + 1;
}

} // namespace

void
JaggedTensor::binary_op_check(const JaggedTensor &other) const {
    TORCH_CHECK(this->device() == other.device(),
                "device should match between this tensor and other tensor");
    TORCH_CHECK(mData.sizes().equals(other.jdata().sizes()),
                "data shape should match between this tensor and other tensor");
    TORCH_CHECK(mBatchIdx.sizes().equals(other.jidx().sizes()),
                "batch indices' shape should match between this tensor and other tensor");
    TORCH_CHECK(mOffsets.sizes().equals(other.joffsets().sizes()),
                "offsets shape should match between this tensor and other tensor");
    if (Config::global().pedanticErrorCheckingEnabled()) {
        // This is a slow check that we cap optionally do for correctness.
        TORCH_CHECK_VALUE(torch::equal(mOffsets, other.joffsets()),
                          "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK_VALUE(
            torch::equal(other.mListIdx, mListIdx),
            "JaggedTensors must have the same lshape. ",
            "This error was raised because config.pedantic_error_checking was enabled");
    }
}

torch::Tensor
JaggedTensor::joffsets_from_jidx_and_jdata(torch::Tensor jidx,
                                           torch::Tensor jdata,
                                           int64_t num_tensors) {
    return detail::ops::jOffsetsFromJIdx(jidx, jdata, num_tensors);
}

torch::Tensor
JaggedTensor::jidx_from_joffsets(torch::Tensor joffsets, int64_t num_elements) {
    // A JaggedTensor with a single list stores an empty jidx by convention (every element maps to
    // batch index 0, and consumers treat an empty jidx as all-zeros). joffsets.size(0) <= 2 means
    // zero or one list, so skip materializing a full array of zeros -- this mirrors the
    // single-tensor constructors and JIdxForGrid.cu, and avoids the redundant 4 B/element array
    // (plus a binary-search kernel) that every batchSize==1 op output was paying.
    if (joffsets.size(0) <= 2) {
        return torch::empty({0},
                            torch::TensorOptions().dtype(JIdxScalarType).device(joffsets.device()));
    }
    return detail::ops::jIdxForJOffsets(joffsets, num_elements);
}

JaggedTensor::JaggedTensor(torch::Tensor data)
    : mData(data), mBatchIdx(torch::empty(
                       {0}, torch::TensorOptions().dtype(JIdxScalarType).device(data.device()))) {
    mListIdx =
        torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(data.device()));
    mOffsets       = joffsets_from_jidx_and_jdata(mBatchIdx, mData, 1);
    mNumOuterLists = 1;
}

JaggedTensor::JaggedTensor(const std::vector<torch::Tensor> &tensors) {
    // TODO: (Francis): rewrite as a cuda kernel

    // An empty list has no tensor to take a device, dtype, or element shape from, so it holds an
    // empty float32 CPU data tensor with edim 0.
    if (tensors.empty()) {
        mData          = torch::empty({0});
        mBatchIdx      = torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType));
        mOffsets       = torch::zeros({1}, torch::TensorOptions().dtype(JOffsetsScalarType));
        mListIdx       = torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType));
        mNumOuterLists = 0;
        return;
    }

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (tensors.size() == 1) {
        // If you have a single element tensor with 0 dimensions, we unsqueeze it to make it 1D
        mData = tensors[0];
        if (tensors[0].dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        mBatchIdx =
            torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType).device(mData.device()));
        mOffsets = torch::tensor(
            {JOffsetsType(0), mData.size(0)},
            torch::TensorOptions()
                .dtype(JOffsetsScalarType)
                .device(mData.device())
                .pinned_memory(mData.device().is_cuda() || mData.device().is_privateuseone()));
        mListIdx = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device()));
        mNumOuterLists = 1;
        return;
    }

    torch::Device device = tensors[0].device();

    std::vector<torch::Tensor> jIdxs;
    mOffsets              = torch::empty({(JOffsetsType)tensors.size() + 1},
                            torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    auto elementCountsAcc = mOffsets.accessor<JOffsetsType, 1>();
    elementCountsAcc[0]   = 0;

    jIdxs.reserve(tensors.size());
    std::vector<torch::Tensor> tensorsReshaped; // Reshape 0D tensors to 1D
    tensorsReshaped.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK_VALUE(tensors[i].device() == device, "All tensors must be on the same device");
        if (tensors[i].dim() == 0 && tensors[i].numel() == 1) {
            tensorsReshaped.push_back(tensors[i].view({1}));
        } else {
            tensorsReshaped.push_back(tensors[i]);
        }
        jIdxs.push_back(torch::full(
            {tensorsReshaped[i].size(0)},
            (int)i,
            torch::TensorOptions().dtype(JIdxScalarType).device(tensorsReshaped[i].device())));
        elementCountsAcc[i + 1] = tensorsReshaped[i].size(0);
    }
    mOffsets = mOffsets.to(tensors[0].device());
    torch::cumsum_out(mOffsets, mOffsets, 0);
    mBatchIdx = torch::cat(jIdxs, 0);
    mData     = torch::cat(tensorsReshaped, 0);
    mListIdx  = torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(device));
    mNumOuterLists = tensors.size();
}

JaggedTensor::JaggedTensor(const std::vector<std::vector<torch::Tensor>> &tensors) {
    // TODO: (Francis): rewrite as a cuda kernel
    torch::Device device      = torch::kCPU;
    bool deviceIsNotSet       = true;
    JOffsetsType totalTensors = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = 0; j < tensors[i].size(); j += 1) {
            if (deviceIsNotSet) {
                device         = tensors[i][j].device();
                deviceIsNotSet = false;
            }
            TORCH_CHECK_VALUE(tensors[i][j].device() == device,
                              "All tensors must be on the same device");
            totalTensors += 1;
        }
    }

    // Outer lists that are all empty have no tensor to take a device, dtype, or element shape
    // from, so they hold an empty float32 data tensor with edim 0.
    if (totalTensors == 0) {
        mData     = torch::empty({0}, torch::TensorOptions().device(device));
        mBatchIdx = torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType).device(device));
        mOffsets =
            torch::zeros({1}, torch::TensorOptions().dtype(JOffsetsScalarType).device(device));
        mListIdx =
            torch::empty({0, 2}, torch::TensorOptions().dtype(JLIdxScalarType).device(device));
        mNumOuterLists = tensors.size();
        return;
    }

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (tensors.size() == 1 && tensors[0].size() == 1) {
        mData = tensors[0][0];
        if (mData.dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        mBatchIdx =
            torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType).device(mData.device()));
        mOffsets =
            torch::tensor({JOffsetsType(0), mData.size(0)},
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(mData.device()));
        mListIdx = torch::zeros(
            {1, 2}, torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device()));
        mNumOuterLists = 1;
        return;
    }

    mOffsets              = torch::empty({totalTensors + 1},
                            torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    auto elementCountsAcc = mOffsets.accessor<JOffsetsType, 1>();
    elementCountsAcc[0]   = 0;

    torch::Tensor listIndexes =
        torch::empty({totalTensors, (JLIdxType)2},
                     torch::TensorOptions().dtype(JLIdxScalarType).device(torch::kCPU));
    auto listIndexesAcc = listIndexes.accessor<JLIdxType, 2>();

    std::vector<torch::Tensor> tensorsReshaped; // Reshape 0D tensors to 1D
    tensorsReshaped.reserve(totalTensors);

    int64_t tensorCount = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = 0; j < tensors[i].size(); j += 1) {
            listIndexesAcc[tensorCount][0] = i;
            listIndexesAcc[tensorCount][1] = j;

            torch::Tensor tij = tensors[i][j];
            if (tij.dim() == 0 && tij.numel() == 1) {
                tensorsReshaped.push_back(tij.view({1}));
            } else {
                tensorsReshaped.push_back(tij);
            }
            elementCountsAcc[tensorCount + 1] = tensorsReshaped[tensorCount].size(0);
            tensorCount += 1;
        }
    }

    mOffsets = mOffsets.to(device);
    torch::cumsum_out(mOffsets, mOffsets, 0);
    mData          = torch::cat(tensorsReshaped, 0);
    mBatchIdx      = jidx_from_joffsets(mOffsets, mData.size(0));
    mListIdx       = listIndexes.to(device);
    mNumOuterLists = tensors.size();
}

JaggedTensor::JaggedTensor(const std::vector<int64_t> &lsizes, const torch::Tensor data) {
    // TODO: (Francis): rewrite as a cuda kernel

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (lsizes.size() == 1) {
        TORCH_CHECK_VALUE(lsizes[0] == data.size(0),
                          "Sum of list sizes must equal the number of elements in data");
        mOffsets =
            torch::tensor({JOffsetsType(0), data.size(0)},
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(data.device()));
        mListIdx = torch::empty(
            {0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(data.device()));
        mNumOuterLists = 1;
        mBatchIdx =
            torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType).device(data.device()));
        mData = data;
        if (mData.dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        return;
    }

    torch::Tensor offsetsCPU =
        torch::empty({(JOffsetsType)lsizes.size() + 1},
                     torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    auto offsetsCPUAcc = offsetsCPU.accessor<JOffsetsType, 1>();

    mListIdx =
        torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(data.device()));
    mNumOuterLists = lsizes.size();

    JOffsetsType cumulativeElements = 0;
    for (size_t i = 0; i < lsizes.size(); ++i) {
        offsetsCPUAcc[i] = cumulativeElements;
        cumulativeElements += lsizes[i];
    }
    offsetsCPUAcc[lsizes.size()] = cumulativeElements;
    TORCH_CHECK_VALUE(cumulativeElements == data.size(0),
                      "Sum of list sizes must equal the number of elements in data");

    mOffsets  = offsetsCPU.to(data.device());
    mData     = data;
    mBatchIdx = jidx_from_joffsets(mOffsets, data.size(0));
}

JaggedTensor::JaggedTensor(const std::vector<std::vector<int64_t>> &lsizes,
                           const torch::Tensor data) {
    // TODO (Francis) : Rewrite as a cuda kernel
    int64_t totalTensors = 0;
    for (const auto &inner: lsizes) {
        totalTensors += inner.size();
    }

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (lsizes.size() == 1 && lsizes[0].size() == 1) {
        TORCH_CHECK_VALUE(lsizes[0][0] == data.size(0), "Invalid size for data tensor.");
        mData = data;
        if (mData.dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        mBatchIdx =
            torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType).device(mData.device()));
        mOffsets =
            torch::tensor({JOffsetsType(0), mData.size(0)},
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(mData.device()));
        mListIdx = torch::zeros(
            {1, 2}, torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device()));
        mNumOuterLists = 1;
        return;
    }

    torch::Tensor offsetsCPU =
        torch::empty({(JOffsetsType)totalTensors + 1},
                     torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    torch::Tensor listIdsCPU =
        torch::empty({(JLIdxType)totalTensors, 2},
                     torch::TensorOptions().dtype(JLIdxScalarType).device(torch::kCPU));
    auto offsetsCPUAcc = offsetsCPU.accessor<JOffsetsType, 1>();
    auto listIdsCPUAcc = listIdsCPU.accessor<JLIdxType, 2>();

    JOffsetsType cumulativeElements = 0;
    int64_t tensorCount             = 0;
    for (size_t i = 0; i < lsizes.size(); ++i) {
        for (size_t j = 0; j < lsizes[i].size(); j += 1) {
            offsetsCPUAcc[tensorCount]    = cumulativeElements;
            listIdsCPUAcc[tensorCount][0] = i;
            listIdsCPUAcc[tensorCount][1] = j;
            cumulativeElements += lsizes[i][j];
            tensorCount += 1;
        }
    }
    offsetsCPUAcc[totalTensors] = cumulativeElements;
    TORCH_CHECK_VALUE(cumulativeElements == data.size(0),
                      "Sum of list sizes must equal the number of elements in data");

    mOffsets       = offsetsCPU.to(data.device());
    mListIdx       = listIdsCPU.to(data.device());
    mBatchIdx      = jidx_from_joffsets(mOffsets, data.size(0));
    mData          = data;
    mNumOuterLists = lsizes.size();
}

void
JaggedTensor::recompute_lsizes_if_dirty() {
    if (!mLShapeCache.mDirty) {
        return;
    }
    mLShapeCache.clear();
    if (ldim() == 1) {
        const torch::Tensor offsetsCpu = mOffsets.cpu();
        const auto acc                 = offsetsCpu.accessor<JOffsetsType, 1>();
        for (int i = 0; i < num_tensors(); ++i) {
            const JOffsetsType startIdx = acc[i];
            const JOffsetsType endIdx   = acc[i + 1];
            mLShapeCache.mLShape1.push_back(endIdx - startIdx);
        }
        mLShapeCache.mDirty = false;
        return;
    } else if (ldim() == 2) {
        const torch::Tensor offsetsCpu = mOffsets.cpu();
        const torch::Tensor listIdxCpu = mListIdx.cpu();
        const auto offAcc              = offsetsCpu.accessor<JOffsetsType, 1>();
        const auto lixAcc              = listIdxCpu.accessor<JLIdxType, 2>();

        // Empty outer lists have no rows in the list indices, so allocate every outer list first.
        mLShapeCache.mLShape2.resize(mNumOuterLists);
        for (int64_t i = 0; i < num_tensors(); ++i) {
            const JLIdxType outerIdx = lixAcc[i][0];
            TORCH_CHECK(outerIdx >= 0 && outerIdx < mNumOuterLists,
                        "Corrupt list indices. Outer list id ",
                        outerIdx,
                        " is out of range for ",
                        mNumOuterLists,
                        " outer lists");
            mLShapeCache.mLShape2[outerIdx].push_back(offAcc[i + 1] - offAcc[i]);
        }
        mLShapeCache.mDirty = false;
        return;
    } else {
        TORCH_CHECK(false,
                    "Unsupported list dimension. Currently JaggedTensor only supports up to 2.");
    }
}

std::vector<torch::Tensor>
JaggedTensor::unbind1() const {
    int64_t ldim = mListIdx.size(1);
    if (ldim != 1) {
        TORCH_WARN(
            "Calling unbind on a multidimensional list of jagged tensors will return a flattened list");
    }
    // Get sizes from cache (uses cached values if available, otherwise syncs once)
    auto sizes = lsizes1();

    return mData.split_with_sizes(sizes, 0);
}

std::vector<std::vector<torch::Tensor>>
JaggedTensor::unbind2() const {
    int64_t ldim = mListIdx.size(1);

    if (ldim != 2) {
        TORCH_CHECK_VALUE(false, "Called unbind2() on a list with list dimension != 2");
    }

    // Get nested sizes from cache
    auto nested_sizes = lsizes2();

    std::vector<std::vector<torch::Tensor>> ret;
    ret.reserve(nested_sizes.size());

    JOffsetsType offset = 0;
    for (const auto &inner_sizes: nested_sizes) {
        std::vector<torch::Tensor> inner_tensors;
        inner_tensors.reserve(inner_sizes.size());

        for (JOffsetsType size: inner_sizes) {
            inner_tensors.push_back(mData.narrow(0, offset, size));
            offset += size;
        }
        ret.push_back(std::move(inner_tensors));
    }
    return ret;
}

std::vector<int64_t>
JaggedTensor::lsizes1() const {
    TORCH_CHECK(ldim() == 1, "Nesting dimension must be 1");
    const_cast<JaggedTensor *>(this)->recompute_lsizes_if_dirty();
    return mLShapeCache.mLShape1;
}

std::vector<std::vector<int64_t>>
JaggedTensor::lsizes2() const {
    TORCH_CHECK(ldim() == 2, "Nesting dimension must be 2");
    const_cast<JaggedTensor *>(this)->recompute_lsizes_if_dirty();
    return mLShapeCache.mLShape2;
}

int64_t
JaggedTensor::ldim() const {
    TORCH_CHECK_VALUE(mListIdx.dim() == 2, "Corrupt list indices. This should never happen");
    const int64_t rows = mListIdx.size(0);
    TORCH_CHECK_VALUE(rows == num_tensors() || (rows == 0 && mListIdx.size(1) == 1),
                      "Corrupt list indices. This should never happen");
    return mListIdx.size(1);
}

std::vector<int64_t>
JaggedTensor::esizes() const {
    std::vector<int64_t> sizes;
    for (size_t i = 1; i < mData.sizes().size(); i++) {
        sizes.push_back(mData.size(i));
    }
    return sizes;
}

int64_t
JaggedTensor::edim() const {
    return mData.dim() > 0 ? mData.dim() - 1 : 0;
}

JaggedTensor
JaggedTensor::jagged_like(torch::Tensor data) const {
    TORCH_CHECK_VALUE(data.dim() > 0,
                      "assigned data must have shape [N, ...], but got data.dim() = 0");
    ldim(); // Checks the list index shape
    TORCH_CHECK_VALUE(data.size(0) == mData.size(0),
                      "Assigned data must have the same number of elements as the JaggedTensor");

    JaggedTensor ret;
    ret.mBatchIdx      = jidx();
    ret.mOffsets       = joffsets();
    ret.mListIdx       = jlidx();
    ret.mNumOuterLists = mNumOuterLists;
    ret.mData          = data.to(device());
    ret.mLShapeCache   = mLShapeCache;
    return ret;
}

JaggedTensor
JaggedTensor::from_data_indices_and_list_ids(torch::Tensor data,
                                             torch::Tensor indices,
                                             torch::Tensor list_ids,
                                             int64_t num_tensors,
                                             std::optional<int64_t> num_outer_lists) {
    TORCH_CHECK_VALUE(data.defined() && data.dim() > 0,
                      "data must have shape [N, ...], but got data.dim() = ",
                      data.defined() ? data.dim() : 0);
    TORCH_CHECK_VALUE(num_tensors >= 0, "num_tensors must be non-negative, but got ", num_tensors);
    indices  = canonicalStructureTensor(indices, JIdxScalarType, "indices");
    list_ids = canonicalStructureTensor(list_ids, JLIdxScalarType, "list_ids");
    const torch::Device checkDevice = structureCheckDevice(data, {indices, list_ids});
    indices                         = indices.to(checkDevice);
    list_ids                        = list_ids.to(checkDevice);
    TORCH_CHECK_VALUE(indices.dim() == 1, "indices must be one-dimensional");
    const int64_t numElements = data.size(0);
    TORCH_CHECK_VALUE(indices.size(0) == numElements ||
                          (indices.size(0) == 0 && (num_tensors == 1 || numElements == 0)),
                      "indices must have one entry per element of data (",
                      numElements,
                      "), or be empty when num_tensors == 1, but got ",
                      indices.size(0),
                      " entries");
    TORCH_CHECK_VALUE(numElements == 0 || num_tensors > 0,
                      "data has ",
                      numElements,
                      " elements but num_tensors is 0");

    const int64_t outerLists = validateStructure(
        torch::Tensor(), numElements, indices, num_tensors, list_ids, num_outer_lists);
    indices  = indices.to(data.device());
    list_ids = list_ids.to(data.device());

    // The indices are now known to be sorted and in range, so each tensor starts at a binary search
    // position. This avoids the second sync that unique_dim in joffsets_from_jidx_and_jdata costs.
    const auto offsetOpts = torch::TensorOptions().dtype(JOffsetsScalarType).device(data.device());
    torch::Tensor offsets;
    if (indices.size(0) > 0) {
        offsets = torch::searchsorted(indices, torch::arange(num_tensors + 1, indices.options()));
    } else if (num_tensors == 1) {
        offsets = torch::tensor({JOffsetsType(0), numElements}, offsetOpts);
    } else {
        offsets = torch::zeros({num_tensors + 1}, offsetOpts);
    }

    return from_jdata_joffsets_jidx_and_lidx_unsafe(
        data, offsets, singleTensorJIdx(indices, num_tensors), list_ids, outerLists);
}

JaggedTensor
JaggedTensor::from_data_offsets_and_list_ids(torch::Tensor data,
                                             torch::Tensor offsets,
                                             torch::Tensor list_ids,
                                             std::optional<int64_t> num_outer_lists) {
    TORCH_CHECK_VALUE(data.defined() && data.dim() > 0,
                      "data must have shape [N, ...], but got data.dim() = ",
                      data.defined() ? data.dim() : 0);
    offsets  = canonicalStructureTensor(offsets, JOffsetsScalarType, "offsets");
    list_ids = canonicalStructureTensor(list_ids, JLIdxScalarType, "list_ids");
    const torch::Device checkDevice = structureCheckDevice(data, {offsets, list_ids});
    offsets                         = offsets.to(checkDevice);
    list_ids                        = list_ids.to(checkDevice);
    TORCH_CHECK_VALUE(
        offsets.dim() == 1 && offsets.size(0) > 0,
        "offsets must be one-dimensional with num_tensors + 1 entries, but got shape ",
        offsets.sizes());
    const int64_t numTensors = offsets.size(0) - 1;

    const int64_t outerLists = validateStructure(
        offsets, data.size(0), torch::Tensor(), numTensors, list_ids, num_outer_lists);
    offsets  = offsets.to(data.device());
    list_ids = list_ids.to(data.device());

    return from_data_offsets_and_list_ids_unsafe(data, offsets, list_ids, outerLists);
}

JaggedTensor
JaggedTensor::from_data_indices_and_list_ids_unsafe(torch::Tensor data,
                                                    torch::Tensor indices,
                                                    torch::Tensor list_ids,
                                                    int64_t num_tensors,
                                                    int64_t num_outer_lists) {
    const torch::Tensor offsets = joffsets_from_jidx_and_jdata(indices, data, num_tensors);
    return from_jdata_joffsets_jidx_and_lidx_unsafe(
        data, offsets, singleTensorJIdx(indices, num_tensors), list_ids, num_outer_lists);
}

JaggedTensor
JaggedTensor::from_data_offsets_and_list_ids_unsafe(torch::Tensor data,
                                                    torch::Tensor offsets,
                                                    torch::Tensor list_ids,
                                                    int64_t num_outer_lists) {
    return from_jdata_joffsets_jidx_and_lidx_unsafe(
        data, offsets, jidx_from_joffsets(offsets, data.size(0)), list_ids, num_outer_lists);
}

JaggedTensor
JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(torch::Tensor jdata,
                                                       torch::Tensor joffsets,
                                                       torch::Tensor jidx,
                                                       torch::Tensor lidx,
                                                       int64_t numOuterLists) {
    TORCH_CHECK_VALUE(
        lidx.dim() == 2,
        "Invalid list indices when constructing JaggedTensor from data, offsets, indices, list indices");
    TORCH_CHECK_VALUE(
        lidx.size(0) == (joffsets.size(0) - 1) || (lidx.size(0) == 0 && lidx.size(1) == 1),
        "Invalid list indices when constructing JaggedTensor from data, offsets, indices, list indices");
    TORCH_CHECK_VALUE(
        joffsets.dim() == 1,
        "Invalid offsets when constructing JaggedTensor from data, offsets, indices, list indices");
    JaggedTensor ret;
    ret.mData          = jdata;
    ret.mOffsets       = joffsets;
    ret.mListIdx       = lidx;
    ret.mNumOuterLists = numOuterLists;
    ret.mBatchIdx      = jidx;
    ret.mLShapeCache.markDirty();
    return ret;
}

void
JaggedTensor::set_jdata(const torch::Tensor &jdata) {
    TORCH_CHECK_VALUE(jdata.dim() > 0,
                      "assigned data must have shape [N, ...], but got data.dim() = 0");
    TORCH_CHECK_VALUE(jdata.device() == mOffsets.device(), "Incorrect device for data");
    ldim(); // Checks the list index shape
    TORCH_CHECK_VALUE(jdata.size(0) == mData.size(0), "assigned data must have shape [N, ...]");
    mData = jdata;
}

JaggedTensor
JaggedTensor::rmask(const torch::Tensor &mask) const {
    TORCH_CHECK(mask.device() == mBatchIdx.device(),
                "mask must be on the same device as the JaggedTensor");
    TORCH_CHECK(mask.dim() == 1, "mask must be 1-dimensional");
    TORCH_CHECK(mask.size(0) == mData.size(0),
                "mask must have the same size as the first dimension of the JaggedTensor");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be of type bool");

    TORCH_CHECK((mask.size(0) == mBatchIdx.size(0)) ||
                    (mBatchIdx.size(0) == 0 && mOffsets.size(0) == 2),
                "Bad jidx. This should never happen. mask.size(0) = ",
                mask.size(0),
                " mBatchIdx.size(0) = ",
                mBatchIdx.size(0));
    const torch::Tensor retData     = mData.index({mask, "..."});
    const torch::Tensor retBatchIds = mBatchIdx.size(0) > 0 ? mBatchIdx.index({mask}) : mBatchIdx;
    const torch::Tensor retOffsets =
        joffsets_from_jidx_and_jdata(retBatchIds, retData, num_tensors());
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        retData, retOffsets, retBatchIds, mListIdx, mNumOuterLists);
}

JaggedTensor
JaggedTensor::index(int64_t index) const {
    return detail::ops::jaggedTensorIndexInt(*this, index);
}

JaggedTensor
JaggedTensor::index(int64_t start, int64_t stop, int64_t step) const {
    return detail::ops::jaggedTensorIndexSlice(*this, start, stop, step);
}

JaggedTensor
JaggedTensor::index(const JaggedTensor &indices) const {
    return detail::ops::jaggedTensorIndexJaggedTensor(*this, indices);
}

JaggedTensor
JaggedTensor::jreshape(const std::vector<int64_t> &lsizes) const {
    return JaggedTensor(lsizes, mData);
}

JaggedTensor
JaggedTensor::jreshape(const std::vector<std::vector<int64_t>> &lsizes) const {
    return JaggedTensor(lsizes, mData);
}

JaggedTensor
JaggedTensor::jreshape_as(const JaggedTensor &other) const {
    return other.jagged_like(mData);
}

JaggedTensor
JaggedTensor::jflatten(const int64_t dim) const {
    int64_t jdim = dim;
    if (dim < 0) {
        jdim += ldim();
    }
    TORCH_CHECK_INDEX(jdim >= 0 && jdim < ldim(), "Invalid dimension to flatten");

    if (ldim() == 2) {
        if (jdim == 1) {
            // Outer list k starts at the first tensor whose outer id is >= k. This keeps empty
            // outer lists, which have no rows in the list indices.
            const torch::Tensor outerIds =
                mListIdx.index({torch::indexing::Slice(), 0}).to(torch::kLong).contiguous();
            const torch::Tensor outerBounds =
                torch::arange(num_outer_lists() + 1,
                              torch::TensorOptions().dtype(torch::kLong).device(mData.device()));
            const torch::Tensor firstTensor = torch::searchsorted(outerIds, outerBounds);
            const torch::Tensor newOffsets  = mOffsets.index({firstTensor});
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                mData,
                newOffsets,
                jidx_from_joffsets(newOffsets, mData.size(0)),
                torch::empty({0, 1},
                             torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device())),
                num_outer_lists());
        } else {
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                mData,
                mOffsets,
                mBatchIdx,
                torch::empty({0, 1},
                             torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device())),
                mOffsets.size(0) - 1);
        }
    } else if (ldim() == 1) {
        return JaggedTensor(mData);
    } else {
        TORCH_CHECK(false,
                    "Unsupported list dimension. Currently JaggedTensor only supports up to 2.");
    }
}
// JaggedTensor JaggedTensor::jagged_argsort() {
//     jidx_from_joffsets(); // why??
//     torch::Tensor argsortIdx = detail::ops::jaggedArgsort(*this);
//
//     return jagged_like(argsortIdx);
// }

JaggedTensor
JaggedTensor::jsum(int64_t dim, bool keepdim) const {
    return detail::ops::jaggedSum(*this, dim, keepdim);
}

std::vector<JaggedTensor>
JaggedTensor::jmin(int64_t dim, bool keepdim) const {
    return detail::ops::jaggedMin(*this, dim, keepdim);
}

JaggedTensor
JaggedTensor::jsqueeze(std::optional<int64_t> dim) const {
    torch::Tensor jdataSqueezed = dim.has_value() ? mData.squeeze(dim.value()) : mData.squeeze();
    if (jdataSqueezed.dim() == 0) {
        jdataSqueezed = jdataSqueezed.unsqueeze(0);
    }
    return jagged_like(jdataSqueezed);
}

std::vector<JaggedTensor>
JaggedTensor::jmax(int64_t dim, bool keepdim) const {
    return detail::ops::jaggedMax(*this, dim, keepdim);
}

JaggedTensor
JaggedTensor::jcat(const std::vector<JaggedTensor> &vec, std::optional<int64_t> dimension) {
    // Null dimension is just list concatenation
    if (!dimension.has_value()) {
        TORCH_CHECK_VALUE(vec.size() > 0, "Empty jagged tensor list");

        // Concat along the batch dimension
        std::vector<torch::Tensor> data;
        std::vector<torch::Tensor> offsets;
        std::vector<torch::Tensor> lidx;
        JOffsetsType curOffset      = 0;
        int64_t totalLists          = 0;
        torch::Tensor curListOffset = torch::zeros(
            {1, vec[0].mListIdx.size(1)},
            torch::TensorOptions().dtype(JLIdxScalarType).device(vec[0].mData.device()));
        for (size_t i = 0; i < vec.size(); ++i) {
            const auto &jvec = vec[i];
            TORCH_CHECK_VALUE(jvec.mData.device() == vec[0].mData.device(),
                              "All JaggedTensors must be on the same device");
            TORCH_CHECK_VALUE(jvec.mListIdx.size(1) == vec[0].mListIdx.size(1),
                              "All JaggedTensors must have the same list dimension");
            TORCH_CHECK_VALUE(jvec.scalar_type() == vec[0].scalar_type(),
                              "All JaggedTensors must have the same scalar type");

            data.push_back(jvec.mData);
            if (i < vec.size() - 1) {
                offsets.push_back(jvec.mOffsets.index({torch::indexing::Slice(0, -1)}) + curOffset);
            } else {
                offsets.push_back(jvec.mOffsets + curOffset);
            }
            if (jvec.mListIdx.size(1) == 2) {
                lidx.push_back(jvec.mListIdx + curListOffset);
            }
            curOffset += jvec.mData.size(0);
            curListOffset[0][0] += jvec.mNumOuterLists;
            totalLists += jvec.mNumOuterLists;
        }
        const torch::Tensor retJData    = torch::cat(data, 0);
        const torch::Tensor retJOffsets = torch::cat(offsets, 0);
        const torch::Tensor retJidx     = jidx_from_joffsets(retJOffsets, retJData.size(0));
        // ldim 1 inputs may mix empty and explicit list indices, so use the empty form
        const torch::Tensor retLidx =
            vec[0].mListIdx.size(1) == 1
                ? torch::empty(
                      {0, 1},
                      torch::TensorOptions().dtype(JLIdxScalarType).device(retJData.device()))
                : torch::cat(lidx, 0);
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            retJData, retJOffsets, retJidx, retLidx, totalLists);
    } else {
        int64_t dim = dimension.value();
        TORCH_CHECK_VALUE(vec.size() > 0, "empty tensor list");
        const int64_t jdim = vec[0].mData.dim();
        TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim <= jdim,
                          "dim must be between ",
                          -(jdim - 1),
                          " and ",
                          jdim - 1,
                          " inclusive");
        if (dim < 0) {
            dim += jdim;
        }

        if (dim == 0) {
            return detail::ops::jCat0(vec);
        } else {
            std::vector<torch::Tensor> data;
            for (const auto &jvec: vec) {
                data.push_back(jvec.mData);
            }
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(torch::cat(data, dim),
                                                                          vec[0].mOffsets,
                                                                          vec[0].mBatchIdx,
                                                                          vec[0].mListIdx,
                                                                          vec[0].mNumOuterLists);
        }
    }
}

JaggedTensor
JaggedTensor::to(at::TensorOptions options,
                 bool non_blocking,
                 bool copy,
                 std::optional<at::MemoryFormat> memory_format) const {
    JaggedTensor ret = *this;
    ret.mData        = ret.mData.to(options, non_blocking, copy, memory_format);
    ret.mBatchIdx    = ret.mBatchIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
    ret.mOffsets     = ret.mOffsets.to(ret.mData.device(), non_blocking, copy, memory_format);
    ret.mListIdx     = ret.mListIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
    return ret;
}

JaggedTensor
JaggedTensor::to(std::optional<torch::ScalarType> dtype,
                 std::optional<at::Layout> layout,
                 std::optional<at::Device> device,
                 std::optional<bool> pin_memory,
                 bool non_blocking,
                 bool copy,
                 std::optional<at::MemoryFormat> memory_format) {
    JaggedTensor ret = *this;
    ret.mData = ret.mData.to(dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
    ret.mBatchIdx = ret.mBatchIdx.to(
        JIdxScalarType, layout, device, pin_memory, non_blocking, copy, memory_format);
    ret.mOffsets = ret.mOffsets.to(
        JOffsetsScalarType, layout, device, pin_memory, non_blocking, copy, memory_format);
    ret.mListIdx = ret.mListIdx.to(
        JLIdxScalarType, layout, device, pin_memory, non_blocking, copy, memory_format);
    return ret;
}
JaggedTensor
JaggedTensor::to(torch::Device device,
                 torch::ScalarType dtype,
                 bool non_blocking,
                 bool copy,
                 std::optional<at::MemoryFormat> memory_format) {
    JaggedTensor ret = *this;
    ret.mData        = ret.mData.to(device, dtype, non_blocking, copy, memory_format);
    ret.mBatchIdx    = ret.mBatchIdx.to(device, non_blocking, copy, memory_format);
    ret.mOffsets     = ret.mOffsets.to(device, non_blocking, copy, memory_format);
    ret.mListIdx     = ret.mListIdx.to(device, non_blocking, copy, memory_format);
    return ret;
}
JaggedTensor
JaggedTensor::to(torch::ScalarType dtype,
                 bool non_blocking,
                 bool copy,
                 std::optional<at::MemoryFormat> memory_format) {
    JaggedTensor ret = *this;
    ret.mData        = ret.mData.to(dtype, non_blocking, copy, memory_format);
    ret.mBatchIdx    = ret.mBatchIdx.to(JIdxScalarType, non_blocking, copy, memory_format);
    ret.mOffsets     = ret.mOffsets.to(JOffsetsScalarType, non_blocking, copy, memory_format);
    ret.mListIdx     = ret.mListIdx.to(JLIdxScalarType, non_blocking, copy, memory_format);
    return ret;
}

JaggedTensor
JaggedTensor::sqrt() const {
    return jagged_like(torch::sqrt(mData));
}
JaggedTensor
JaggedTensor::abs() const {
    return jagged_like(torch::abs(mData));
}

JaggedTensor
JaggedTensor::floor() const {
    return jagged_like(torch::floor(mData));
}

JaggedTensor
JaggedTensor::ceil() const {
    return jagged_like(torch::ceil(mData));
}

JaggedTensor
JaggedTensor::round(int decimals) const {
    return jagged_like(torch::round(mData, decimals));
}

JaggedTensor &
JaggedTensor::sqrt_() {
    mData.sqrt_();
    return *this;
}
JaggedTensor &
JaggedTensor::abs_() {
    mData.abs_();
    return *this;
}

JaggedTensor &
JaggedTensor::floor_() {
    mData.floor_();
    return *this;
}

JaggedTensor &
JaggedTensor::ceil_() {
    mData.ceil_();
    return *this;
}

JaggedTensor &
JaggedTensor::round_(int decimals) {
    mData.round_(decimals);
    return *this;
}

const JaggedTensor &
JaggedTensor::set_requires_grad(bool requires_grad) const {
    mData.set_requires_grad(requires_grad);
    return *this;
}

bool
JaggedTensor::requires_grad() const {
    return mData.requires_grad();
}

JaggedTensor
JaggedTensor::detach() const {
    return jagged_like(mData.detach());
}

JaggedTensor
JaggedTensor::clone() const {
    return jagged_like(mData.clone());
}

JaggedTensor
JaggedTensor::operator+(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData + other.mData);
}
JaggedTensor
JaggedTensor::operator+(const int other) const {
    return jagged_like(mData + other);
}
JaggedTensor
JaggedTensor::operator+(const float other) const {
    return jagged_like(mData + other);
}
JaggedTensor
JaggedTensor::operator+(const torch::Tensor &other) const {
    return jagged_like(mData + other);
}

JaggedTensor &
JaggedTensor::operator+=(const JaggedTensor &other) {
    binary_op_check(other);
    mData += other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator+=(const int other) {
    mData += other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator+=(const float other) {
    mData += other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator+=(const torch::Tensor &other) {
    mData += other;
    return *this;
}

JaggedTensor
JaggedTensor::operator-(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData - other.mData);
}
JaggedTensor
JaggedTensor::operator-(const int other) const {
    return jagged_like(mData - other);
}
JaggedTensor
JaggedTensor::operator-(const float other) const {
    return jagged_like(mData - other);
}
JaggedTensor
JaggedTensor::operator-(const torch::Tensor &other) const {
    return jagged_like(mData - other);
}

JaggedTensor
JaggedTensor::operator-() const {
    return jagged_like(-mData);
}

JaggedTensor &
JaggedTensor::operator-=(const JaggedTensor &other) {
    binary_op_check(other);
    mData -= other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator-=(const int other) {
    mData -= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator-=(const float other) {
    mData -= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator-=(const torch::Tensor &other) {
    mData -= other;
    return *this;
}

JaggedTensor
JaggedTensor::operator*(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData * other.mData);
}
JaggedTensor
JaggedTensor::operator*(const int other) const {
    return jagged_like(mData * other);
}
JaggedTensor
JaggedTensor::operator*(const float other) const {
    return jagged_like(mData * other);
}
JaggedTensor
JaggedTensor::operator*(const torch::Tensor &other) const {
    return jagged_like(mData * other);
}

JaggedTensor &
JaggedTensor::operator*=(const JaggedTensor &other) {
    binary_op_check(other);
    mData *= other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator*=(const int other) {
    mData *= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator*=(const float other) {
    mData *= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator*=(const torch::Tensor &other) {
    mData *= other;
    return *this;
}

JaggedTensor
JaggedTensor::operator/(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData / other.mData);
}
JaggedTensor
JaggedTensor::operator/(const int other) const {
    return jagged_like(mData / other);
}
JaggedTensor
JaggedTensor::operator/(const float other) const {
    return jagged_like(mData / other);
}
JaggedTensor
JaggedTensor::operator/(const torch::Tensor &other) const {
    return jagged_like(mData / other);
}

JaggedTensor &
JaggedTensor::operator/=(const JaggedTensor &other) {
    binary_op_check(other);
    mData /= other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator/=(const int other) {
    mData /= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator/=(const float other) {
    mData /= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator/=(const torch::Tensor &other) {
    mData /= other;
    return *this;
}

JaggedTensor
JaggedTensor::floordiv(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(torch::floor_divide(mData, other.mData));
}
JaggedTensor
JaggedTensor::floordiv(const int other) const {
    return jagged_like(torch::floor_divide(mData, other));
}
JaggedTensor
JaggedTensor::floordiv(const float other) const {
    return jagged_like(torch::floor_divide(mData, other));
}
JaggedTensor
JaggedTensor::floordiv(const torch::Tensor &other) const {
    return jagged_like(torch::floor_divide(mData, other));
}

JaggedTensor &
JaggedTensor::floordiveq(const JaggedTensor &other) {
    binary_op_check(other);
    mData.floor_divide_(other.mData);
    return *this;
}
JaggedTensor &
JaggedTensor::floordiveq(const int other) {
    mData = torch::floor_divide(mData, other);
    return *this;
}
JaggedTensor &
JaggedTensor::floordiveq(const float other) {
    mData = torch::floor_divide(mData, other);
    return *this;
}
JaggedTensor &
JaggedTensor::floordiveq(const torch::Tensor &other) {
    mData.floor_divide_(other);
    return *this;
}

JaggedTensor
JaggedTensor::operator%(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData % other.mData);
}
JaggedTensor
JaggedTensor::operator%(const int other) const {
    return jagged_like(mData % other);
}
JaggedTensor
JaggedTensor::operator%(const float other) const {
    return jagged_like(mData % other);
}
JaggedTensor
JaggedTensor::operator%(const torch::Tensor &other) const {
    return jagged_like(mData % other);
}

JaggedTensor &
JaggedTensor::operator%=(const JaggedTensor &other) {
    binary_op_check(other);
    mData = mData % other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator%=(const int other) {
    mData = mData % other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator%=(const float other) {
    mData = mData % other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator%=(const torch::Tensor &other) {
    mData = mData % other;
    return *this;
}

JaggedTensor
JaggedTensor::pow(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData.pow(other.mData));
}
JaggedTensor
JaggedTensor::pow(const int other) const {
    return jagged_like(mData.pow(other));
}
JaggedTensor
JaggedTensor::pow(const float other) const {
    return jagged_like(mData.pow(other));
}
JaggedTensor
JaggedTensor::pow(const torch::Tensor &other) const {
    return jagged_like(mData.pow(other));
}

JaggedTensor &
JaggedTensor::poweq(const JaggedTensor &other) {
    binary_op_check(other);
    mData.pow_(other.mData);
    return *this;
}
JaggedTensor &
JaggedTensor::poweq(const int other) {
    mData = mData.pow(other);
    return *this;
}
JaggedTensor &
JaggedTensor::poweq(const float other) {
    mData = mData.pow(other);
    return *this;
}
JaggedTensor &
JaggedTensor::poweq(const torch::Tensor &other) {
    mData.pow_(other);
    return *this;
}

JaggedTensor
JaggedTensor::operator>(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData > other.mData);
}
JaggedTensor
JaggedTensor::operator>(const int other) const {
    return jagged_like(mData > other);
}
JaggedTensor
JaggedTensor::operator>(const float other) const {
    return jagged_like(mData > other);
}
JaggedTensor
JaggedTensor::operator>(const torch::Tensor &other) const {
    return jagged_like(mData > other);
}

JaggedTensor
JaggedTensor::operator>=(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData >= other.mData);
}
JaggedTensor
JaggedTensor::operator>=(const int other) const {
    return jagged_like(mData >= other);
}
JaggedTensor
JaggedTensor::operator>=(const float other) const {
    return jagged_like(mData >= other);
}
JaggedTensor
JaggedTensor::operator>=(const torch::Tensor &other) const {
    return jagged_like(mData >= other);
}

JaggedTensor
JaggedTensor::operator<(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData < other.mData);
}
JaggedTensor
JaggedTensor::operator<(const int other) const {
    return jagged_like(mData < other);
}
JaggedTensor
JaggedTensor::operator<(const float other) const {
    return jagged_like(mData < other);
}
JaggedTensor
JaggedTensor::operator<(const torch::Tensor &other) const {
    return jagged_like(mData < other);
}

JaggedTensor
JaggedTensor::operator<=(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData <= other.mData);
}
JaggedTensor
JaggedTensor::operator<=(const int other) const {
    return jagged_like(mData <= other);
}
JaggedTensor
JaggedTensor::operator<=(const float other) const {
    return jagged_like(mData <= other);
}
JaggedTensor
JaggedTensor::operator<=(const torch::Tensor &other) const {
    return jagged_like(mData <= other);
}

JaggedTensor
JaggedTensor::operator==(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData == other.mData);
}
JaggedTensor
JaggedTensor::operator==(const int other) const {
    return jagged_like(mData == other);
}
JaggedTensor
JaggedTensor::operator==(const float other) const {
    return jagged_like(mData == other);
}
JaggedTensor
JaggedTensor::operator==(const torch::Tensor &other) const {
    return jagged_like(mData == other);
}

JaggedTensor
JaggedTensor::operator!=(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData != other.mData);
}
JaggedTensor
JaggedTensor::operator!=(const int other) const {
    return jagged_like(mData != other);
}
JaggedTensor
JaggedTensor::operator!=(const float other) const {
    return jagged_like(mData != other);
}
JaggedTensor
JaggedTensor::operator!=(const torch::Tensor &other) const {
    return jagged_like(mData != other);
}

} // namespace fvdb
