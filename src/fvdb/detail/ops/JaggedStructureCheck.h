// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JAGGEDSTRUCTURECHECK_H
#define FVDB_DETAIL_OPS_JAGGEDSTRUCTURECHECK_H

#include <torch/types.h>

#include <cstdint>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Bits set in JaggedStructureCheck::failures, one per violated invariant
enum JaggedStructureFailure : uint64_t {
    kOffsetsStart        = 1u << 0, ///< offsets[0] != 0
    kOffsetsEnd          = 1u << 1, ///< offsets[-1] != number of elements
    kOffsetsDecreasing   = 1u << 2, ///< offsets decrease somewhere
    kIndicesNegative     = 1u << 3, ///< an index is negative
    kIndicesTooLarge     = 1u << 4, ///< an index is >= the tensor count
    kIndicesDecreasing   = 1u << 5, ///< indices decrease somewhere
    kOuterIdsNegative    = 1u << 6, ///< an outer list id is negative
    kOuterIdsDecreasing  = 1u << 7, ///< outer list ids decrease somewhere
    kInnerIdsNotCounting = 1u << 8, ///< inner ids do not count 0, 1, ... within an outer list
    kOuterIdsTooLarge    = 1u << 9, ///< an outer list id is >= the outer list count
};

/// @brief Outcome of checkJaggedStructure
struct JaggedStructureCheck {
    uint64_t failures   = 0;  ///< OR of JaggedStructureFailure bits
    int64_t lastOuterId = -1; ///< Outer id of the last list id row, or -1 if there are no rows
};

/// @brief Check the values of a JaggedTensor's structure tensors in one pass.
///
/// Each argument tensor is optional; an undefined tensor is not checked. On CUDA the result is read
/// back with a single device-to-host copy.
///
/// @param offsets Offsets with numTensors + 1 entries (JOffsetsType)
/// @param numElements Number of elements in the data tensor
/// @param indices Per-element tensor indices (JIdxType)
/// @param numTensors Number of tensors
/// @param listIds List ids of shape [numTensors, 2] (JLIdxType)
/// @param numOuterLists Outer list count to check outer ids against, or -1 to skip that check
/// @return The violated invariants and the last outer id
JaggedStructureCheck checkJaggedStructure(const torch::Tensor &offsets,
                                          int64_t numElements,
                                          const torch::Tensor &indices,
                                          int64_t numTensors,
                                          const torch::Tensor &listIds,
                                          int64_t numOuterLists);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGEDSTRUCTURECHECK_H
