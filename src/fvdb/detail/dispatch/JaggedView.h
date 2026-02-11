// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// JaggedView.h — Trivially-copyable jagged tensor views for dispatch.
//
// jagged_in provides read-only access to a JaggedTensor's data and batch
// structure.  It mirrors dispatch::tensor_in (same template parameters,
// same contiguity specializations, same __hostdev__ access methods) but
// additionally carries the per-element batch index array that maps each
// flat element index to its batch.
//
// Constructors are host-only (they pull raw pointers from JaggedTensor).
// Access methods are __hostdev__.  The struct is a trivially-copyable POD
// after construction, safe for CUDA kernel capture.
//
// NOTE: Generalized from the first jagged-iterating ops ported to the new
// dispatch pattern (IjkToIndex, PointsInGrid).  May need extension as more
// ops with jagged inputs are migrated.
//
#ifndef FVDB_DETAIL_DISPATCH_JAGGED_VIEW_H
#define FVDB_DETAIL_DISPATCH_JAGGED_VIEW_H

#include "dispatch/macros.h"
#include "dispatch/torch/types.h"

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cassert>
#include <cstdint>

namespace fvdb {
namespace detail {
namespace dispatch {

// =============================================================================
// jagged_in — read-only jagged tensor access
// =============================================================================

template <torch::DeviceType Dev,
          torch::ScalarType Stype,
          int64_t Rank,
          ::dispatch::contiguity Contig = ::dispatch::contiguity::strided>
struct jagged_in;

// -----------------------------------------------------------------------------
// jagged_in — strided specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype, int64_t Rank>
struct jagged_in<Dev, Stype, Rank, ::dispatch::contiguity::strided> {
    static_assert(Rank > 0, "Rank must be positive");

    using value_type = ::dispatch::torch_scalar_cpp_type_t<Stype>;

    // Data access (same layout as dispatch::tensor_in<strided>)
    value_type const *data;
    int64_t sizes[Rank];
    int64_t strides[Rank];

    // Jagged structure
    JIdxType const *batch_idx_ptr; // per-element batch index (may be nullptr for single-batch)
    int64_t n_elements;            // total element count == sizes[0]

    explicit jagged_in(JaggedTensor const &jt)
        : data(jt.jdata().data_ptr<value_type>()), n_elements(jt.element_count()) {
        auto const &dt = jt.jdata();
        assert(dt.dim() >= Rank && "JaggedTensor data rank must be >= view Rank");
        for (int64_t d = 0; d < Rank; ++d) {
            sizes[d]   = dt.size(d);
            strides[d] = dt.stride(d);
        }
        // batch index: may be empty for single-batch jagged tensors.
        // Replicate the old JaggedAccessor/PackedJaggedAccessor convention:
        //   batchIdx(idx) returns mBatchIdx[idx] if non-empty, else 0.
        auto const &bidx = jt.jidx();
        batch_idx_ptr    = bidx.numel() > 0 ? bidx.data_ptr<JIdxType>() : nullptr;
    }

    /// Number of elements (first dimension of the data tensor).
    __hostdev__ int64_t
    numel() const {
        return n_elements;
    }

    /// Batch index for element `eidx`.  Returns 0 for single-batch tensors.
    __hostdev__ JIdxType
    batchIdx(int64_t eidx) const {
        return batch_idx_ptr ? batch_idx_ptr[eidx] : JIdxType(0);
    }

    /// Multi-index data access (strided).
    template <typename... Idx>
    __hostdev__ value_type
    operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Number of indices must equal Rank");
        int64_t const indices[] = {static_cast<int64_t>(idx)...};
        int64_t offset          = 0;
        for (int64_t d = 0; d < Rank; ++d)
            offset += indices[d] * strides[d];
        return data[offset];
    }

    __hostdev__ int64_t
    size(int64_t d) const {
        return sizes[d];
    }

    __hostdev__ int64_t
    stride(int64_t d) const {
        return strides[d];
    }
};

// -----------------------------------------------------------------------------
// jagged_in — contiguous specialization
// -----------------------------------------------------------------------------

template <torch::DeviceType Dev, torch::ScalarType Stype, int64_t Rank>
struct jagged_in<Dev, Stype, Rank, ::dispatch::contiguity::contiguous> {
    static_assert(Rank > 0, "Rank must be positive");

    using value_type = ::dispatch::torch_scalar_cpp_type_t<Stype>;

    // Data access (same layout as dispatch::tensor_in<contiguous>)
    value_type const *data;
    int64_t sizes[Rank];

    // Jagged structure
    JIdxType const *batch_idx_ptr;
    int64_t n_elements;

    explicit jagged_in(JaggedTensor const &jt)
        : data(jt.jdata().data_ptr<value_type>()), n_elements(jt.element_count()) {
        auto const &dt = jt.jdata();
        assert(dt.dim() >= Rank && "JaggedTensor data rank must be >= view Rank");
        assert(dt.is_contiguous() && "Contiguous jagged_in requires contiguous data tensor");
        for (int64_t d = 0; d < Rank; ++d) {
            sizes[d] = dt.size(d);
        }
        auto const &bidx = jt.jidx();
        batch_idx_ptr    = bidx.numel() > 0 ? bidx.data_ptr<JIdxType>() : nullptr;
    }

    __hostdev__ int64_t
    numel() const {
        return n_elements;
    }

    __hostdev__ JIdxType
    batchIdx(int64_t eidx) const {
        return batch_idx_ptr ? batch_idx_ptr[eidx] : JIdxType(0);
    }

    /// Multi-index data access (contiguous, row-major).
    template <typename... Idx>
    __hostdev__ value_type
    operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) == Rank, "Number of indices must equal Rank");
        int64_t const indices[] = {static_cast<int64_t>(idx)...};
        int64_t offset          = indices[0];
        for (int64_t d = 1; d < Rank; ++d)
            offset = offset * sizes[d] + indices[d];
        return data[offset];
    }

    __hostdev__ int64_t
    size(int64_t d) const {
        return sizes[d];
    }

    __hostdev__ int64_t
    stride(int64_t d) const {
        int64_t s = 1;
        for (int64_t i = Rank - 1; i > d; --i)
            s *= sizes[i];
        return s;
    }
};

} // namespace dispatch
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_JAGGED_VIEW_H
