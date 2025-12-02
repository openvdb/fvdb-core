// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_JAGGEDTENSOR_H
#define FVDB_JAGGEDTENSOR_H

#include <fvdb/detail/utils/Utils.h>

#include <torch/custom_class.h>
#include <torch/types.h>

#include <optional>

namespace fvdb {

using JIdxType     = int32_t;
using JOffsetsType = int64_t;
using JLIdxType    = int32_t;

constexpr c10::ScalarType JIdxScalarType     = c10::CppTypeToScalarType<JIdxType>::value;
constexpr c10::ScalarType JOffsetsScalarType = c10::CppTypeToScalarType<JOffsetsType>::value;
constexpr c10::ScalarType JLIdxScalarType    = c10::CppTypeToScalarType<JLIdxType>::value;

/// @brief JaggedAccessor provides efficient access to jagged tensor data
///
/// @tparam ScalarT The scalar type of the data elements (e.g., float, double)
/// @tparam NDims The number of dimensions in the data tensor
template <typename ScalarT, size_t NDims> class JaggedAccessor {
    torch::TensorAccessor<JIdxType, 1> mBatchIdx;
    torch::TensorAccessor<JOffsetsType, 1> mOffsets;
    torch::TensorAccessor<JLIdxType, 2> mListIndexes;
    torch::TensorAccessor<ScalarT, NDims> mData;

    friend class JaggedTensor;

    JaggedAccessor(torch::TensorAccessor<JIdxType, 1> batchIdx,
                   torch::TensorAccessor<JOffsetsType, 1> offsets,
                   torch::TensorAccessor<JLIdxType, 2> listIndexes,
                   torch::TensorAccessor<ScalarT, NDims> data)
        : mBatchIdx(batchIdx), mOffsets(offsets), mListIndexes(listIndexes), mData(data) {}

  public:
    template <typename T, size_t N> using TensorAccessorType = torch::TensorAccessor<T, N>;

    inline __hostdev__ int64_t
    elementCount() const {
        return mData.size(0);
    }

    inline __hostdev__ JIdxType
    batchIdx(int64_t idx) const {
        return mBatchIdx.size(0) > 0 ? mBatchIdx[idx] : 0;
    }

    inline __hostdev__ JOffsetsType
    offsetStart(int64_t idx) const {
        return mOffsets[idx];
    }

    inline __hostdev__ JOffsetsType
    offsetEnd(int64_t idx) const {
        return mOffsets[idx + 1];
    }

    inline __hostdev__ const torch::TensorAccessor<ScalarT, NDims> &
    data() const {
        return mData;
    }
};

/// @brief PackedJaggedAccessor provides efficient access to jagged tensor data
///
/// @tparam ScalarT The scalar type of the data elements (e.g., float, double)
/// @tparam NDims The number of dimensions in the data tensor
/// @tparam PtrTraits The pointer traits to use for the accessor
/// @tparam index_t The type of the index (e.g., int32_t, int64_t)
template <typename ScalarT,
          size_t NDims,
          template <typename U> typename PtrTraits = torch::DefaultPtrTraits,
          typename index_t                         = int64_t>
class PackedJaggedAccessor {
    torch::GenericPackedTensorAccessor<JIdxType, 1, PtrTraits, index_t> mBatchIdx;
    torch::GenericPackedTensorAccessor<JOffsetsType, 1, PtrTraits, index_t> mOffsets;
    torch::GenericPackedTensorAccessor<JLIdxType, 2, PtrTraits, index_t> mListIndexes;
    torch::GenericPackedTensorAccessor<ScalarT, NDims, PtrTraits, index_t> mData;

    friend class JaggedTensor;

    PackedJaggedAccessor(
        torch::GenericPackedTensorAccessor<JIdxType, 1, PtrTraits, index_t> batchIdx,
        torch::GenericPackedTensorAccessor<JOffsetsType, 1, PtrTraits, index_t> offsets,
        torch::GenericPackedTensorAccessor<JLIdxType, 2, PtrTraits, index_t> listIndexes,
        torch::GenericPackedTensorAccessor<ScalarT, NDims, PtrTraits, index_t> data)
        : mBatchIdx(batchIdx), mOffsets(offsets), mListIndexes(listIndexes), mData(data) {}

  public:
    template <typename T, size_t N>
    using TensorAccessorType = torch::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>;

    inline __hostdev__ JOffsetsType
    numTensors() const {
        return mOffsets.size(0) - 1;
    }

    /// @brief Get the number of elements in the jagged tensor
    /// @return The number of elements in the jagged tensor
    inline __hostdev__ index_t
    elementCount() const {
        return mData.size(0);
    }

    /// @brief Get the batch index of an element
    /// @param idx The index of the element
    /// @return The batch index of the element
    inline __hostdev__ JIdxType
    batchIdx(index_t idx) const {
        return mBatchIdx.size(0) > 0 ? mBatchIdx[idx] : 0;
    }

    /// @brief Get the start offset of a tensor
    /// @param idx The index of the tensor
    /// @return The start offset of the tensor
    inline __hostdev__ JOffsetsType
    offsetStart(index_t idx) const {
        return mOffsets[idx];
    }

    /// @brief Get the end offset of a tensor
    /// @param idx The index of the tensor
    /// @return The end offset of the tensor
    inline __hostdev__ JOffsetsType
    offsetEnd(index_t idx) const {
        return mOffsets[idx + 1];
    }

    /// @brief Get the data tensor
    /// @return The data tensor
    inline __hostdev__ const TensorAccessorType<ScalarT, NDims> &
    data() const {
        return mData;
    }

    /// @brief Get the data tensor
    /// @return The data tensor
    inline __hostdev__ TensorAccessorType<ScalarT, NDims> &
    data() {
        return mData;
    }
};

/// @brief Alias for PackedJaggedAccessor with int32_t indices
template <typename ScalarT,
          size_t NDims,
          template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
using PackedJaggedAccessor32 = PackedJaggedAccessor<ScalarT, NDims, PtrTraits, int32_t>;

/// @brief Alias for PackedJaggedAccessor with int64_t indices
template <typename ScalarT,
          size_t NDims,
          template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
using PackedJaggedAccessor64 = PackedJaggedAccessor<ScalarT, NDims, PtrTraits, int64_t>;

/// @brief JaggedTensor is a class that represents a jagged tensor
///
/// A jagged tensor is a tensor that stores variable-length sequences in a compact representation.
/// It is represented by a data tensor and a set of indices/offsets/list indexes.
class JaggedTensor : public torch::CustomClassHolder {
    torch::Tensor mData;     // Actual data indexed by a jagged tensor
    torch::Tensor mBatchIdx; // Which (linear) batch is each datum in
    torch::Tensor mOffsets;  // Offset of each tensor in the list of lists
    torch::Tensor mListIdx;  // LoL indexing of tensor with shape [num_tensors, ldim]
    int64_t mNumOuterLists;  // Number of outer lists in this JaggedTensor

    // Store the number of elements in each tensor in the jagged tensor
    // Computing this requires a GPU -> CPU copy so we cache it
    struct {
        std::vector<int64_t> mLShape1;
        std::vector<std::vector<int64_t>> mLShape2;
        std::vector<std::vector<std::vector<int64_t>>> mLShape3;
        bool mDirty = true;
        void
        markDirty() {
            mDirty = true;
        }
        void
        clear() {
            mLShape1.clear();
            mLShape2.clear();
            mLShape3.clear();
            mDirty = true;
        }
    } mLShapeCache;

    void recompute_lsizes_if_dirty();

    void binary_op_check(const JaggedTensor &other) const;

  public:
    /// @brief Compute the offsets from the indices and data
    /// @param jidx The indices of the jagged tensor
    /// @param jdata The data of the jagged tensor
    /// @param num_tensors The number of tensors in the jagged tensor
    /// @return The offsets of the jagged tensor
    static torch::Tensor
    joffsets_from_jidx_and_jdata(torch::Tensor jidx, torch::Tensor jdata, int64_t num_tensors);

    /// @brief Compute the indices from the offsets
    /// @param joffsets The offsets of the jagged tensor
    /// @param num_elements The number of elements in the jagged tensor
    /// @return The indices of the jagged tensor
    static torch::Tensor jidx_from_joffsets(torch::Tensor joffsets, int64_t num_elements);

    /// @brief Create a JaggedTensor from the given data, offsets, indices, and list indices
    static JaggedTensor from_jdata_joffsets_jidx_and_lidx_unsafe(torch::Tensor jdata,
                                                                 torch::Tensor joffsets,
                                                                 torch::Tensor jidx,
                                                                 torch::Tensor jlidx,
                                                                 int64_t numOuterLists);

    /// @brief Create a JaggedTensor from data, per-element indices, and list IDs.
    ///
    /// This function validates that data, indices, list_ids, and num_tensors are compatible.
    /// The offsets are computed internally from the indices.
    ///
    /// Example (ldim == 1, list of tensors):
    ///
    ///     torch::Tensor data = torch::tensor({1, 2, 3, 4, 5});
    ///     torch::Tensor indices = torch::tensor({0, 0, 1, 1, 1});  // Elements 0-1 in tensor 0,
    ///                                                              // elements 2-4 in tensor 1
    ///     torch::Tensor list_ids = torch::empty({0, 1}, torch::kInt);  // Empty for ldim == 1
    ///     JaggedTensor jt = JaggedTensor::from_data_indices_and_list_ids(data, indices, list_ids,
    ///     2);
    ///     // Result: [[1, 2], [3, 4, 5]]
    ///
    /// Example (ldim == 2, list of lists of tensors):
    ///
    ///     torch::Tensor data = torch::tensor({1, 2, 3, 4, 5, 6});
    ///     torch::Tensor indices = torch::tensor({0, 0, 1, 1, 1, 2});  // 3 tensors
    ///     torch::Tensor list_ids = torch::tensor({{0, 0}, {0, 1}, {1, 0}});  // Outer/inner
    ///     indices JaggedTensor jt = JaggedTensor::from_data_indices_and_list_ids(data, indices,
    ///     list_ids, 3);
    ///     // Result: [[[1, 2], [3, 4, 5]], [[6]]]
    ///
    /// @param data Flattened data tensor containing all elements.
    ///             Shape: (total_elements, ...).
    /// @param indices Index tensor mapping each element to its tensor index (0 to num_tensors-1).
    ///                Shape: (total_elements,), or empty if num_tensors == 1.
    /// @param list_ids Tensor defining the hierarchical position of each tensor.
    ///                 For ldim == 1: shape (num_tensors, 1) or empty tensor with shape (0, 1).
    ///                 Empty tensor assumes a single, naturally ordered list of tensors.
    ///                 For ldim == 2: shape (num_tensors, 2) where each row is (outer_idx,
    ///                 inner_idx).
    /// @param num_tensors Total number of tensors.
    /// @return A JaggedTensor representing defined by the data, indices, and list ids.
    static JaggedTensor from_data_indices_and_list_ids(torch::Tensor data,
                                                       torch::Tensor indices,
                                                       torch::Tensor list_ids,
                                                       int64_t num_tensors);

    /// @brief Create a JaggedTensor from data, offsets, and list IDs.
    ///
    /// This function validates that data, offsets, and list_ids are compatible.
    /// The per-element indices are computed internally from the offsets.
    ///
    /// Example (ldim == 1, list of tensors):
    ///
    ///     torch::Tensor data = torch::tensor({1, 2, 3, 4, 5});
    ///     torch::Tensor offsets = torch::tensor({0, 2, 5});  // 2 tensors: data[0:2], data[2:5]
    ///     torch::Tensor list_ids = torch::empty({0, 1}, torch::kInt);  // Empty for ldim == 1
    ///     JaggedTensor jt = JaggedTensor::from_data_offsets_and_list_ids(data, offsets, list_ids);
    ///     // Result: [[1, 2], [3, 4, 5]]
    ///
    /// Example (ldim == 2, list of lists of tensors):
    ///
    ///     torch::Tensor data = torch::tensor({1, 2, 3, 4, 5, 6});
    ///     torch::Tensor offsets = torch::tensor({0, 2, 5, 6});  // 3 tensors: [0:2], [2:5], [5:6]
    ///     torch::Tensor list_ids = torch::tensor({{0, 0}, {0, 1}, {1, 0}});
    ///     // list_ids maps: tensor 0 -> list[0][0], tensor 1 -> list[0][1], tensor 2 -> list[1][0]
    ///     JaggedTensor jt = JaggedTensor::from_data_offsets_and_list_ids(data, offsets, list_ids);
    ///     // Result: [[[1, 2], [3, 4, 5]], [[6]]]
    ///
    /// @param data Flattened data tensor containing all elements.
    ///             Shape: (total_elements, ...).
    /// @param offsets Offset tensor marking tensor boundaries. offsets[i] is the start index
    ///                of tensor i, and offsets[i+1] - offsets[i] is its length.
    ///                Shape: (num_tensors + 1,).
    /// @param list_ids Tensor defining the hierarchical position of each tensor.
    ///                 For ldim == 1: shape (num_tensors, 1) or empty tensor with shape (0, 1).
    ///                 Empty tensor assumes a single, naturally ordered list of tensors.
    ///                 For ldim == 2: shape (num_tensors, 2) where each row is (outer_idx,
    ///                 inner_idx).
    /// @return A JaggedTensor representing defined by the data, offsets, and list ids.
    static JaggedTensor from_data_offsets_and_list_ids(torch::Tensor data,
                                                       torch::Tensor offsets,
                                                       torch::Tensor list_ids);

    /// @brief Concatenate the list of JaggedTensors along a given dimension.
    ///        There are two modes for this function.
    ///        1. If dim is an integer:
    ///            e.g. if [jt_a, jt_b] are two JaggedTensors of the form
    ///            jt_a = [[a_11, a_12], [a_21], [a_31, a_32]] and jt_b = [[b_11, b_12], [b_21],
    ///            [b_31, b_32]], then JaggedTensor::jcat({jt_a, jt_b}) will return a JaggedTensor
    ///            of the form
    ///            [[torch.cat([a_11, b_11], dim=dim), torch.cat([a_12, b_12], dim=dim)],
    ///             [torch.cat([a_21, b_21], dim=dim)],
    ///             [torch.cat([a_31, b_31], dim=dim), torch.cat([a_32, b_32], dim=dim)]]
    ///        2. If dim is std::nullopt:
    ///            e.g. if [jt_a, jt_b] are two JaggedTensors of the form
    ///            jt_a = [[a_11, a_12], [a_21], [a_31, a_32]] and jt_b = [[b_11], [b_21, b_22]],
    ///            then JaggedTensor::jcat({jt_a, jt_b}) will return a JaggedTensor of the form
    ///            [[a_11, a_12], [a_21], [a_31, a_32], [b_11], [b_21, b_22]]
    /// @param vec A vector of JaggedTensors to concatenate
    /// @param dim The dimension along which to concatenate each JaggedTensor or std::nullopt to
    /// concatenate
    ///            the JaggedTensors as lists
    /// @return A JaggedTensor containing the concatenated data
    static JaggedTensor jcat(const std::vector<JaggedTensor> &vec, std::optional<int64_t> dim);

    /// @brief Create an empty JaggedTensor
    JaggedTensor() {
        mData          = torch::Tensor();
        mBatchIdx      = torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType));
        mOffsets       = torch::zeros({1}, torch::TensorOptions().dtype(JOffsetsScalarType));
        mListIdx       = torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType));
        mNumOuterLists = 0;
    }

    /// @brief Create a JaggedTensor representing a list with a single tensor. Note this function
    /// does not copy the
    ///        data tensor, it only creates a view of it.
    /// @param data The data tensor
    JaggedTensor(torch::Tensor data);

    /// @brief Create a JaggedTensor representing a list of tensors.
    /// @param tensors A list of tensors
    JaggedTensor(const std::vector<torch::Tensor> &tensors);

    /// @brief Create a JaggedTensor representing a list of lists of tensors.
    /// @param tensors A list of lists of tensors
    JaggedTensor(const std::vector<std::vector<torch::Tensor>> &tensors);

    /// @brief Create a JaggedTensor representing a list of tensors where the number of elements in
    /// each tensor is given
    ///        by the lsizes vector. i.e. if lsizes = [2, 1, 2], then the first tensor will have 2
    ///        elements, the second tensor will have 1 element, and the third tensor will have 2
    ///        elements. The raw data tensor must then have a number of elements equal to the sum of
    ///        the elements in lsizes (i.e. shape [sum(lsizes), ...])
    /// @param lsizes A vector of integers indicating the number of elements in each tensor
    /// @param data The raw data tensor
    JaggedTensor(const std::vector<int64_t> &lsizes, const torch::Tensor data);

    /// @brief Create a JaggedTensor representing a list of lists of tensors where the number of
    /// elements in each tensor
    ///       is given by the lsizes vector. i.e. if lsizes = [[2, 1], [5, 6, 7]], then the first
    ///       list will have 2 tensors with 1 and 2 elements respectively and the second list will
    ///       have 3 tensors with 5, 6, and 7 elements respectively. The raw data tensor must then
    ///       have a number of elements equal to the sum of the elements in lsizes (i.e. shape
    ///       [sum(lsizes), ...])
    /// @param lsizes A vector of vectors of integers indicating the number of elements in each
    /// tensor
    /// @param total_tensors The total number of tensors in the list of lists
    /// @param data The raw data tensor
    JaggedTensor(const std::vector<std::vector<int64_t>> &lsizes,
                 const int64_t total_tensors,
                 const torch::Tensor data);

    /// @brief Create a JaggedTensor with the same list structure as this one but with the given raw
    /// data.
    ///        The returned JaggedTensor will share the same memory for indices/list ids/offsets as
    ///        this one those are modified.
    /// @param data A tensor with the same number of elements as the original data
    /// @return A JaggedTensor with the same list structure as this one but with the given data
    JaggedTensor jagged_like(torch::Tensor data) const;

    /// @brief Set the raw data of this JaggedTensor to the given tensor
    /// @param data A data tensor with the same number of elements as the original data
    void set_jdata(const torch::Tensor &data);

    /// @brief  Get the raw data indexed by this JaggedTensor
    /// @return The raw data tensor
    const torch::Tensor &
    jdata() const {
        return mData;
    }

    /// @brief Get the indices of this jagged tensor. i.e. a tensor of size (num_elements,)
    /// indicating which
    ///        tensor each element belongs to
    /// @return The indices of this JaggedTensor
    const torch::Tensor &
    jidx() const {
        return mBatchIdx;
    }

    /// @brief Get the offsets of each tensor indexed by this JaggedTensor. i.e. a tensor of size
    /// (num_tensors + 1)
    ///        where joffsets[i] is the start offset in jdata and joffsets[i+1] is the end offset in
    ///        jdata
    /// @return The offsets of each tensor indexed by this JaggedTensor
    const torch::Tensor &
    joffsets() const {
        return mOffsets;
    }

    /// @brief Get the list indices of each tensor indexed by this JaggedTensor. i.e. a tensor of
    /// size (num_tensors, ldim)
    ///        where e.g. jlidx[i][j] is the index of the j-th list in the i-th tensor (for a list
    ///        of lists JaggedTensor)
    /// @return The list indices of each tensor indexed by this JaggedTensor
    const torch::Tensor &
    jlidx() const {
        return mListIdx;
    }

    /// @brief Get the number of outer lists in this JaggedTensor
    int64_t
    num_outer_lists() const {
        return mNumOuterLists;
    }

    /// @brief Get the number of tensors in this JaggedTensor
    int64_t
    num_tensors() const {
        return mOffsets.size(0) - 1;
    }

    /// @brief Get the number of elements in each tensor indexed by this JaggedTensor. Assumes the
    /// JaggedTensor has ldim() == 1 i.e. it represents a list of tensors
    /// @return The number of elements in each tensor indexed by this JaggedTensor
    std::vector<int64_t> lsizes1() const;

    /// @brief Get the number of elements in each tensor indexed by this JaggedTensor. Assumes
    /// JaggedTensor has ldim() == 2 i.e. it represents a list of lists of tensors
    /// @return The number of elements in each tensor indexed by this JaggedTensor such that
    /// lsizes2()[i][j] is the number of elements in the j-th tensor in i-th list
    std::vector<std::vector<int64_t>> lsizes2() const;

    /// @brief Get the number of nested lists encoded by this JaggedTensor. An ldim of one means
    /// this JaggedTensor encodes a list of tensors, an ldim of 2 means this JaggedTensor
    //  encodes a list of lists of tensors, etc.
    /// @return The number of nested lists encoded by this JaggedTensor
    int64_t ldim() const;

    /// @brief Get the size of each element indexed by this JaggedTensor. i.e. if the JaggedTensor
    /// represents a list of tensors where each tensor has shape [N_i, A, B, C], then esizes() will
    //  return [A, B, C]
    /// @return The size of each element indexed by this JaggedTensor
    std::vector<int64_t> esizes() const;

    /// @brief Get the number of dimensions of each element indexed by this JaggedTensor. i.e. if
    /// the JaggedTensor represents a list of tensors where each tensor has shape [N_i, A, B, C],
    // then edim() will return 3
    /// @return The number of dimensions of each element indexed by this JaggedTensor
    int64_t edim() const;

    /// @brief Convert the JaggedTensor to a list of tensors assuming this JaggedTensor represents a
    /// list of tensors.
    ///        Note this function doesn't work for nested lists of tensors (instead use unbind2())
    /// @return A list of tensors where each tensor is indexed by this JaggedTensor.
    std::vector<torch::Tensor> unbind1() const;

    /// @brief Convert the JaggedTensor to a list of lists of tensors assuming this JaggedTensor
    /// represents a list of lists of tensors.
    ///        Note this function doesn't work for a flat list of tensors (instead use unbind1())
    /// @return A list of lists of tensors where each tensor is indexed by this JaggedTensor.
    std::vector<std::vector<torch::Tensor>> unbind2() const;

    /// @brief Index JaggedTensor with an integer index along the outer list dimension.
    ///
    /// index(i) returns the i^th list in this tensor if ldim() == 1 or a list containing the
    /// i^th tensor if ldim() == 2.
    /// @param index The integer index to use to index this JaggedTensor
    /// @return A JaggedTensor that is a view of the indexed data from the original JaggedTensor
    JaggedTensor index(int64_t index) const;

    /// @brief Index JaggedTensor with a slice along the outer list dimension.
    ///
    /// index(start, stop, step) returns a JaggedTensor containing the specified range of
    /// lists if ldim() == 1 or a list containing the specified range of tensors if ldim() ==
    /// 2.
    /// @note Currently only supports contiguous slices (i.e. step = 1)
    /// @param start The starting index of the slice
    /// @param stop The ending index of the slice (exclusive)
    /// @param step The step size of the slice
    /// @return A JaggedTensor that is a view of the indexed data from the original JaggedTensor
    JaggedTensor index(int64_t start, int64_t stop, int64_t step) const;

    /// @brief Index JaggedTensor with another JaggedTensor of indices.
    ///
    /// index(indices) returns a JaggedTensor containing tensors indexed by the provided indices.
    /// For boolean mask values: index(mask)[i][j].jdata = index(i)[j].jdata[mask[i][j].jdata]
    /// For integer indices: index(indices)[i][j].jdata = index(i)[j].jdata[indices[i][j]]
    /// @param indices The JaggedTensor containing the indices to use for indexing
    /// @return A JaggedTensor that is a view of the indexed data from the original JaggedTensor
    JaggedTensor index(const JaggedTensor &indices) const;

    /// @brief Reshape JaggedTensor to have a new list structure.
    ///
    /// The provided lsizes should be compatible with this tensor; the sum of the elements in
    /// lsizes should be equal to the number of elements in this JaggedTensor.
    ///
    /// @note This function creates a view over the original JaggedTensor so modifying the returned
    /// JaggedTensor will modify the original tensor.
    /// @param lsizes The new list structure
    /// @return A JaggedTensor that is a view of the reshaped data from the original JaggedTensor
    JaggedTensor jreshape(const std::vector<int64_t> &lsizes) const;

    /// @brief Reshape JaggedTensor to have a new nested list structure.
    ///
    /// The provided lsizes should be compatible with this tensor; the sum of the nested list
    /// elements in lsizes should be equal to the number of elements in this JaggedTensor.
    ///
    /// @note This function creates a view over the original JaggedTensor so modifying the returned
    /// JaggedTensor will modify the original tensor.
    /// @param lsizes The new nested list structure
    /// @return A JaggedTensor that is a view of the reshaped data from the original JaggedTensor
    JaggedTensor jreshape(const std::vector<std::vector<int64_t>> &lsizes) const;

    /// @brief Reshape JaggedTensor to have the same list structure as another JaggedTensor.
    ///
    /// @note This function creates a view over the original JaggedTensor so modifying the returned
    /// JaggedTensor will modify the original tensor.
    /// @param other The JaggedTensor to reshape this JaggedTensor to have the same list structure
    /// as
    /// @return A JaggedTensor that is a view of the reshaped data from the original JaggedTensor
    JaggedTensor jreshape_as(const JaggedTensor &other) const;

    /// Flatten one of the list dimensions of this JaggedTensor. i.e. if this JaggedTensor
    /// represents a list of lists of tensors then jflatten(0) will flatten the outer list dimension
    /// and jflatten(1) will flatten the inner list dimension. e.g. if this JaggedTensor represents
    /// a list of lists of tensors [[A, B], [C], [D, E]] then
    ///     - jflatten(0) will return a JaggedTensor [A, B, C, D, E]
    ///     - jflatten(1) will return a JaggedTensor [[torch.cat(A, B, dim=0)], [C], [torch.cat(D,
    ///     E, dim=0)]]
    /// e.g. if this JaggedTensor represents a list of tensors with shapes [A, B, C] then
    ///    - jflatten(0) will return a JaggedTensor with shape [torch.cat(A, B, C, dim=0)]
    ///    - jflatten(1) will raise an exception as there is no inner list dimension
    /// Note this function creates a view over the original JaggedTensor so modifying the returned
    /// JaggedTensor will modify the original tensor.
    /// @param dim The dimension to flatten
    /// @return A JaggedTensor with the flattened list dimension
    JaggedTensor jflatten(const int64_t dim = 0) const;

    /// @brief Sorts each batch element in ascending order, note that jdata has to be 1-dimensional
    /// @return An indexing tensor with the same size as jdata, that permutes the elements of data
    /// to be in sorted order
    // JaggedTensor jagged_argsort();

    /// @brief Compute the summation of each batch element
    /// @param dim The dimension to sum over
    /// @param keepdim Whether to keep the summed dimension
    /// @return A tensor of size (batch_size, *) containing the sum of each batch element, feature
    /// dimensions are preserved
    JaggedTensor jsum(int64_t dim = 0, bool keepdim = false) const;

    /// @brief Compute the minimum of each batch element
    /// @param dim The dimension to sum over
    /// @param keepdim Whether to keep the min dimension
    /// @return Minimum value of size (batch_size, *) and argmin of size (batch_size, *)
    std::vector<JaggedTensor> jmin(int64_t dim = 0, bool keepdim = false) const;

    /// @brief Compute the maximum of each batch element
    /// @param dim The dimension to sum over
    /// @param keepdim Whether to keep the max dimension
    /// @return Maximum value of size (batch_size, *) and argmax of size (batch_size, *)
    std::vector<JaggedTensor> jmax(int64_t dim = 0, bool keepdim = false) const;

    /// @brief Squeeze each tensor in the JaggedTensor. i.e. if this JaggedTensor represents a list
    ///        of tensors with shape [N_i, 1, B, C] then jsqueeze() will return a JaggedTensor with
    ///        where each tensor has shape [N_i, B, C].
    ///        i.e. if this JaggedTensor is a list of tensors [t1, t2, t3], then:
    ///        jsqueeze(dim) will return a JaggedTensor
    ///        [t1.squeeze(dim), t2.squeeze(dim), t3.squeeze(dim)]
    /// @param dim The dimension to squeeze. None to squeeze all dimensions of size 1
    /// @return A JaggedTensor with the squeezed tensors
    /// @note This function creates a view over the original JaggedTensor so modifying the
    ///       returned JaggedTensor will modify the original tensor.
    JaggedTensor jsqueeze(std::optional<int64_t> dim = std::nullopt) const;

    // Operators on raw data

    /// @brief Get the size of a dimension of the data tensor
    /// @param dim The dimension to get the size of
    /// @return The size of the dimension
    inline int64_t
    rsize(int64_t dim) const {
        return mData.size(dim);
    }

    /// @brief Get the number of dimensions of the data tensor
    /// @return The number of dimensions of the data tensor
    inline int64_t
    rdim() const {
        return mData.dim();
    }

    /// @brief Get the sizes of all dimensions of the data tensor
    /// @return The sizes of all dimensions of the data tensor
    inline std::vector<int64_t>
    rsizes() const {
        return mData.sizes().vec();
    }

    /// @brief Mask the JaggedTensor with a mask tensor
    /// @param mask The mask tensor
    /// @return A JaggedTensor with the masked data
    JaggedTensor rmask(const torch::Tensor &mask) const;

    /// @brief Get an accessor for the JaggedTensor. Useful for reading/writing values in the
    /// JaggedTensor
    /// @tparam Scalar The type of the data in the JaggedTensor
    /// @tparam NDims The number of dimensions of the data in the JaggedTensor (i.e. edim() + 1)
    /// @return An accessor for the JaggedTensor
    template <typename Scalar, size_t NDims>
    JaggedAccessor<Scalar, NDims>
    accessor() const {
        return JaggedAccessor<Scalar, NDims>(mBatchIdx.accessor<JIdxType, 1>(),
                                             mOffsets.accessor<JOffsetsType, 1>(),
                                             mListIdx.accessor<JLIdxType, 2>(),
                                             mData.accessor<Scalar, NDims>());
    }

    /// @brief Get a packed accessor for the JaggedTensor with 32-bit indices. Useful for
    /// reading/writing values in the JaggedTensor in Cuda
    /// @tparam Scalar The type of the data in the JaggedTensor
    /// @tparam NDims The number of dimensions of the data in the JaggedTensor (i.e. edim() + 1)
    /// @tparam PtrTraits The type of the pointer traits for the packed accessor
    /// @return A packed accessor for the JaggedTensor
    template <typename Scalar,
              size_t NDims,
              template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
    PackedJaggedAccessor32<Scalar, NDims, PtrTraits>
    packed_accessor32() const {
        return PackedJaggedAccessor32<Scalar, NDims, PtrTraits>(
            mBatchIdx.packed_accessor32<JIdxType, 1, PtrTraits>(),
            mOffsets.packed_accessor32<JOffsetsType, 1, PtrTraits>(),
            mListIdx.packed_accessor32<JLIdxType, 2, PtrTraits>(),
            mData.packed_accessor32<Scalar, NDims, PtrTraits>());
    }

    /// @brief Get a packed accessor for the JaggedTensor with 64-bit indices. Useful for
    /// reading/writing values in the JaggedTensor in Cuda
    /// @tparam Scalar The type of the data in the JaggedTensor
    /// @tparam NDims The number of dimensions of the data in the JaggedTensor (i.e. edim() + 1)
    /// @tparam PtrTraits The type of the pointer traits for the packed accessor
    /// @return A packed accessor for the JaggedTensor
    template <typename Scalar,
              size_t NDims,
              template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
    PackedJaggedAccessor64<Scalar, NDims, PtrTraits>
    packed_accessor64() const {
        return PackedJaggedAccessor64<Scalar, NDims, PtrTraits>(
            mBatchIdx.packed_accessor64<JIdxType, 1, PtrTraits>(),
            mOffsets.packed_accessor64<JOffsetsType, 1, PtrTraits>(),
            mListIdx.packed_accessor64<JLIdxType, 2, PtrTraits>(),
            mData.packed_accessor64<Scalar, NDims, PtrTraits>());
    }

    /// @brief Raise an exception if the JaggedTensor is in an invalid state
    inline void
    check_valid() const {
        TORCH_CHECK((jidx().size(0) == 0 && joffsets().size(0) == 2) ||
                        (jidx().size(0) == jdata().size(0)),
                    "tensor must be a valid JaggedTensor");
        TORCH_CHECK(jidx().device() == jdata().device(),
                    "batch index and data must be on the same device");
        TORCH_CHECK(jidx().dtype() == JIdxScalarType, "batch index must be int");
        TORCH_CHECK(joffsets().device() == jdata().device(),
                    "offsets and data must be on the same device");
        TORCH_CHECK(jlidx().device() == jdata().device(),
                    "list indices and data must be on the same device");
        TORCH_CHECK_VALUE(jlidx().numel() == 0 || jlidx().size(0) == (joffsets().size(0) - 1),
                          "Corrupt list indices. This should never happen");
    }

    /// @brief Get the total number of elements in the JaggedTensor
    /// @return The total number of elements in the JaggedTensor
    inline int64_t
    element_count() const {
        return jdata().size(0);
    }

    /// @brief Get the device of the JaggedTensor
    inline torch::Device
    device() const {
        return mData.device();
    }

    /// @brief Get the data type of the JaggedTensor
    /// @return The data type of the JaggedTensor
    caffe2::TypeMeta
    dtype() const {
        return mData.dtype();
    }

    /// @brief Get the layout of the JaggedTensor
    /// @return The layout of the JaggedTensor
    torch::Layout
    layout() const {
        return mData.layout();
    }

    /// @brief Get the scalar type of the JaggedTensor
    /// @return The scalar type of the JaggedTensor
    inline torch::ScalarType
    scalar_type() const {
        return mData.scalar_type();
    }

    /// @brief Check if the JaggedTensor is on a CUDA device
    /// @return True if the JaggedTensor is on a CUDA device, false otherwise
    inline bool
    is_cuda() const {
        return mData.is_cuda();
    }

    /// @brief Check if the JaggedTensor is on a PrivateUse1 device
    /// @return True if the JaggedTensor is on a PrivateUse1 device, false otherwise
    inline bool
    is_privateuseone() const {
        return mData.is_privateuseone();
    }

    /// @brief Check if the JaggedTensor is on a CPU device
    /// @return True if the JaggedTensor is on a CPU device, false otherwise
    inline bool
    is_cpu() const {
        return mData.is_cpu();
    }

    /// @brief Get the device ID of the JaggedTensor
    /// @return The device ID of the JaggedTensor
    int64_t
    get_device() const {
        return mData.get_device();
    }

    /// @brief Check if the JaggedTensor is a complex type
    /// @return True if the JaggedTensor is a complex type, false otherwise
    bool
    is_complex() const {
        return at::isComplexType(this->scalar_type());
    }

    /// @brief Check if the JaggedTensor is a floating point type
    /// @return True if the JaggedTensor is a floating point type, false otherwise
    bool
    is_floating_point() const {
        return at::isFloatingType(this->scalar_type());
    }

    /// @brief Check if the JaggedTensor is a signed type
    /// @return True if the JaggedTensor is a signed type, false otherwise
    bool
    is_signed() const {
        return at::isSignedType(this->scalar_type());
    }

    /// @brief Get the total number of elements in the JaggedTensor
    /// @return The total number of elements in the JaggedTensor
    int64_t
    numel() const {
        return mData.numel();
    }

    /// @brief Check if the JaggedTensor is contiguous
    /// @return True if the JaggedTensor is contiguous, false otherwise
    inline bool
    is_contiguous() const {
        return mData.is_contiguous();
    }

    /// @brief Return a contiguous copy of the JaggedTensor
    /// @return A contiguous copy of the JaggedTensor
    inline JaggedTensor
    contiguous() const {
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(mData.contiguous(),
                                                                      mOffsets.contiguous(),
                                                                      mBatchIdx.contiguous(),
                                                                      mListIdx.contiguous(),
                                                                      mNumOuterLists);
    }

    /// @brief Convert the JaggedTensor to a new device and/or data type
    /// @param options The options for the new JaggedTensor
    /// @param non_blocking Use an asynchronous copy if true, synchronous otherwise
    /// @param copy Whether to copy the data
    /// @param memory_format The memory format of the new JaggedTensor
    /// @return A JaggedTensor on the new device and/or data type
    JaggedTensor to(at::TensorOptions options                     = {},
                    bool non_blocking                             = false,
                    bool copy                                     = false,
                    std::optional<at::MemoryFormat> memory_format = std::nullopt) const;

    /// @brief Convert the JaggedTensor to a new data type
    /// @param dtype The data type of the new JaggedTensor
    /// @param layout The layout of the new JaggedTensor
    /// @param device The device of the new JaggedTensor
    /// @param pin_memory Whether to pin the memory of the new JaggedTensor
    /// @param non_blocking Use an asynchronous copy if true, synchronous otherwise
    /// @param copy Whether to copy the data
    /// @param memory_format The memory format of the new JaggedTensor
    /// @return A JaggedTensor on the new device and/or data type
    JaggedTensor to(std::optional<torch::ScalarType> dtype,
                    std::optional<at::Layout> layout,
                    std::optional<at::Device> device,
                    std::optional<bool> pin_memory,
                    bool non_blocking,
                    bool copy,
                    std::optional<at::MemoryFormat> memory_format);

    /// @brief Convert the JaggedTensor to a new device and data type
    /// @param device The device of the new JaggedTensor
    /// @param dtype The data type of the new JaggedTensor
    /// @param non_blocking Use an asynchronous copy if true, synchronous otherwise
    /// @param copy Whether to copy the data
    /// @param memory_format The memory format of the new JaggedTensor
    /// @return A JaggedTensor on the new device and data type
    JaggedTensor to(torch::Device device,
                    torch::ScalarType dtype,
                    bool non_blocking                             = false,
                    bool copy                                     = false,
                    std::optional<at::MemoryFormat> memory_format = std::nullopt);

    /// @brief Convert the JaggedTensor to a new data type
    /// @param dtype The data type of the new JaggedTensor
    /// @param non_blocking Use an asynchronous copy if true, synchronous otherwise
    /// @param copy Whether to copy the data
    /// @param memory_format The memory format of the new JaggedTensor
    /// @return A JaggedTensor on the new data type
    JaggedTensor to(torch::ScalarType dtype,
                    bool non_blocking                             = false,
                    bool copy                                     = false,
                    std::optional<at::MemoryFormat> memory_format = std::nullopt);

    /// @brief Get the options for the JaggedTensor
    /// @return The options for the JaggedTensor
    torch::TensorOptions
    options() const {
        return torch::TensorOptions().dtype(dtype()).device(device()).layout(layout());
    }

    /// @brief Convert the JaggedTensor to a CUDA JaggedTensor
    /// @return A CUDA JaggedTensor
    JaggedTensor
    cuda() const {
        return to(this->options().device(torch::kCUDA), /*non_blocking*/ false, /*copy*/ false);
    }

    /// @brief Convert the JaggedTensor to a CPU JaggedTensor
    /// @return A CPU JaggedTensor
    JaggedTensor
    cpu() const {
        return to(this->options().device(torch::kCPU), /*non_blocking*/ false, /*copy*/ false);
    }

    /// @brief Add two JaggedTensors element-wise
    /// @param other The JaggedTensor to add
    /// @return A new JaggedTensor with the result
    JaggedTensor operator+(const JaggedTensor &other) const;

    /// @brief Add a scalar integer to each element
    /// @param other The integer to add
    /// @return A new JaggedTensor with the result
    JaggedTensor operator+(const int other) const;

    /// @brief Add a scalar float to each element
    /// @param other The float to add
    /// @return A new JaggedTensor with the result
    JaggedTensor operator+(const float other) const;

    /// @brief Add a tensor to each element
    /// @param other The tensor to add
    /// @return A new JaggedTensor with the result
    JaggedTensor operator+(const torch::Tensor &other) const;

    /// @brief Add a JaggedTensor to this one in-place
    /// @param other The JaggedTensor to add
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator+=(const JaggedTensor &other);

    /// @brief Add a scalar integer to each element in-place
    /// @param other The integer to add
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator+=(const int other);

    /// @brief Add a scalar float to each element in-place
    /// @param other The float to add
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator+=(const float other);

    /// @brief Add a tensor to each element in-place
    /// @param other The tensor to add
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator+=(const torch::Tensor &other);

    /// @brief Subtract a JaggedTensor from this one element-wise
    /// @param other The JaggedTensor to subtract
    /// @return A new JaggedTensor with the result
    JaggedTensor operator-(const JaggedTensor &other) const;

    /// @brief Subtract a scalar integer from each element
    /// @param other The integer to subtract
    /// @return A new JaggedTensor with the result
    JaggedTensor operator-(const int other) const;

    /// @brief Subtract a scalar float from each element
    /// @param other The float to subtract
    /// @return A new JaggedTensor with the result
    JaggedTensor operator-(const float other) const;

    /// @brief Subtract a tensor from each element
    /// @param other The tensor to subtract
    /// @return A new JaggedTensor with the result
    JaggedTensor operator-(const torch::Tensor &other) const;

    /// @brief Negate each element
    /// @return A new JaggedTensor with negated values
    JaggedTensor operator-() const;

    /// @brief Subtract a JaggedTensor from this one in-place
    /// @param other The JaggedTensor to subtract
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator-=(const JaggedTensor &other);

    /// @brief Subtract a scalar integer from each element in-place
    /// @param other The integer to subtract
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator-=(const int other);

    /// @brief Subtract a scalar float from each element in-place
    /// @param other The float to subtract
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator-=(const float other);

    /// @brief Subtract a tensor from each element in-place
    /// @param other The tensor to subtract
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator-=(const torch::Tensor &other);

    /// @brief Multiply two JaggedTensors element-wise
    /// @param other The JaggedTensor to multiply by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator*(const JaggedTensor &other) const;

    /// @brief Multiply each element by a scalar integer
    /// @param other The integer to multiply by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator*(const int other) const;

    /// @brief Multiply each element by a scalar float
    /// @param other The float to multiply by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator*(const float other) const;

    /// @brief Multiply each element by a tensor
    /// @param other The tensor to multiply by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator*(const torch::Tensor &other) const;

    /// @brief Multiply this JaggedTensor by another in-place
    /// @param other The JaggedTensor to multiply by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator*=(const JaggedTensor &other);

    /// @brief Multiply each element by a scalar integer in-place
    /// @param other The integer to multiply by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator*=(const int other);

    /// @brief Multiply each element by a scalar float in-place
    /// @param other The float to multiply by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator*=(const float other);

    /// @brief Multiply each element by a tensor in-place
    /// @param other The tensor to multiply by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator*=(const torch::Tensor &other);

    /// @brief Divide this JaggedTensor by another element-wise
    /// @param other The JaggedTensor to divide by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator/(const JaggedTensor &other) const;

    /// @brief Divide each element by a scalar integer
    /// @param other The integer to divide by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator/(const int other) const;

    /// @brief Divide each element by a scalar float
    /// @param other The float to divide by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator/(const float other) const;

    /// @brief Divide each element by a tensor
    /// @param other The tensor to divide by
    /// @return A new JaggedTensor with the result
    JaggedTensor operator/(const torch::Tensor &other) const;

    /// @brief Divide this JaggedTensor by another in-place
    /// @param other The JaggedTensor to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator/=(const JaggedTensor &other);

    /// @brief Divide each element by a scalar integer in-place
    /// @param other The integer to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator/=(const int other);

    /// @brief Divide each element by a scalar float in-place
    /// @param other The float to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator/=(const float other);

    /// @brief Divide each element by a tensor in-place
    /// @param other The tensor to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator/=(const torch::Tensor &other);

    /// @brief Perform floor division with another JaggedTensor
    /// @param other The JaggedTensor to divide by
    /// @return A new JaggedTensor with the floor division result
    JaggedTensor floordiv(const JaggedTensor &other) const;

    /// @brief Perform floor division with a scalar integer
    /// @param other The integer to divide by
    /// @return A new JaggedTensor with the floor division result
    JaggedTensor floordiv(const int other) const;

    /// @brief Perform floor division with a scalar float
    /// @param other The float to divide by
    /// @return A new JaggedTensor with the floor division result
    JaggedTensor floordiv(const float other) const;

    /// @brief Perform floor division with a tensor
    /// @param other The tensor to divide by
    /// @return A new JaggedTensor with the floor division result
    JaggedTensor floordiv(const torch::Tensor &other) const;

    /// @brief Perform floor division with another JaggedTensor in-place
    /// @param other The JaggedTensor to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &floordiveq(const JaggedTensor &other);

    /// @brief Perform floor division with a scalar integer in-place
    /// @param other The integer to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &floordiveq(const int other);

    /// @brief Perform floor division with a scalar float in-place
    /// @param other The float to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &floordiveq(const float other);

    /// @brief Perform floor division with a tensor in-place
    /// @param other The tensor to divide by
    /// @return Reference to this JaggedTensor
    JaggedTensor &floordiveq(const torch::Tensor &other);

    /// @brief Compute remainder with another JaggedTensor
    /// @param other The JaggedTensor to compute remainder with
    /// @return A new JaggedTensor with the remainder
    JaggedTensor operator%(const JaggedTensor &other) const;

    /// @brief Compute remainder with a scalar integer
    /// @param other The integer to compute remainder with
    /// @return A new JaggedTensor with the remainder
    JaggedTensor operator%(const int other) const;

    /// @brief Compute remainder with a scalar float
    /// @param other The float to compute remainder with
    /// @return A new JaggedTensor with the remainder
    JaggedTensor operator%(const float other) const;

    /// @brief Compute remainder with a tensor
    /// @param other The tensor to compute remainder with
    /// @return A new JaggedTensor with the remainder
    JaggedTensor operator%(const torch::Tensor &other) const;

    /// @brief Compute remainder with another JaggedTensor in-place
    /// @param other The JaggedTensor to compute remainder with
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator%=(const JaggedTensor &other);

    /// @brief Compute remainder with a scalar integer in-place
    /// @param other The integer to compute remainder with
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator%=(const int other);

    /// @brief Compute remainder with a scalar float in-place
    /// @param other The float to compute remainder with
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator%=(const float other);

    /// @brief Compute remainder with a tensor in-place
    /// @param other The tensor to compute remainder with
    /// @return Reference to this JaggedTensor
    JaggedTensor &operator%=(const torch::Tensor &other);

    /// @brief Raise this JaggedTensor to the power of another
    /// @param other The JaggedTensor exponent
    /// @return A new JaggedTensor with the result
    JaggedTensor pow(const JaggedTensor &other) const;

    /// @brief Raise each element to a scalar integer power
    /// @param other The integer exponent
    /// @return A new JaggedTensor with the result
    JaggedTensor pow(const int other) const;

    /// @brief Raise each element to a scalar float power
    /// @param other The float exponent
    /// @return A new JaggedTensor with the result
    JaggedTensor pow(const float other) const;

    /// @brief Raise each element to a tensor power
    /// @param other The tensor exponent
    /// @return A new JaggedTensor with the result
    JaggedTensor pow(const torch::Tensor &other) const;

    /// @brief Raise this JaggedTensor to the power of another in-place
    /// @param other The JaggedTensor exponent
    /// @return Reference to this JaggedTensor
    JaggedTensor &poweq(const JaggedTensor &other);

    /// @brief Raise each element to a scalar integer power in-place
    /// @param other The integer exponent
    /// @return Reference to this JaggedTensor
    JaggedTensor &poweq(const int other);

    /// @brief Raise each element to a scalar float power in-place
    /// @param other The float exponent
    /// @return Reference to this JaggedTensor
    JaggedTensor &poweq(const float other);

    /// @brief Raise each element to a tensor power in-place
    /// @param other The tensor exponent
    /// @return Reference to this JaggedTensor
    JaggedTensor &poweq(const torch::Tensor &other);

    /// @brief Compare if elements are greater than another JaggedTensor
    /// @param other The JaggedTensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>(const JaggedTensor &other) const;

    /// @brief Compare if elements are greater than a scalar integer
    /// @param other The integer to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>(const int other) const;

    /// @brief Compare if elements are greater than a scalar float
    /// @param other The float to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>(const float other) const;

    /// @brief Compare if elements are greater than a tensor
    /// @param other The tensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>(const torch::Tensor &other) const;

    /// @brief Compare if elements are greater than or equal to another JaggedTensor
    /// @param other The JaggedTensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>=(const JaggedTensor &other) const;

    /// @brief Compare if elements are greater than or equal to a scalar integer
    /// @param other The integer to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>=(const int other) const;

    /// @brief Compare if elements are greater than or equal to a scalar float
    /// @param other The float to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>=(const float other) const;

    /// @brief Compare if elements are greater than or equal to a tensor
    /// @param other The tensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator>=(const torch::Tensor &other) const;

    /// @brief Compare if elements are less than another JaggedTensor
    /// @param other The JaggedTensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<(const JaggedTensor &other) const;

    /// @brief Compare if elements are less than a scalar integer
    /// @param other The integer to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<(const int other) const;

    /// @brief Compare if elements are less than a scalar float
    /// @param other The float to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<(const float other) const;

    /// @brief Compare if elements are less than a tensor
    /// @param other The tensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<(const torch::Tensor &other) const;

    /// @brief Compare if elements are less than or equal to another JaggedTensor
    /// @param other The JaggedTensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<=(const JaggedTensor &other) const;

    /// @brief Compare if elements are less than or equal to a scalar integer
    /// @param other The integer to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<=(const int other) const;

    /// @brief Compare if elements are less than or equal to a scalar float
    /// @param other The float to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<=(const float other) const;

    /// @brief Compare if elements are less than or equal to a tensor
    /// @param other The tensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator<=(const torch::Tensor &other) const;

    /// @brief Compare if elements are equal to another JaggedTensor
    /// @param other The JaggedTensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator==(const JaggedTensor &other) const;

    /// @brief Compare if elements are equal to a scalar integer
    /// @param other The integer to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator==(const int other) const;

    /// @brief Compare if elements are equal to a scalar float
    /// @param other The float to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator==(const float other) const;

    /// @brief Compare if elements are equal to a tensor
    /// @param other The tensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator==(const torch::Tensor &other) const;

    /// @brief Compare if elements are not equal to another JaggedTensor
    /// @param other The JaggedTensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator!=(const JaggedTensor &other) const;

    /// @brief Compare if elements are not equal to a scalar integer
    /// @param other The integer to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator!=(const int other) const;

    /// @brief Compare if elements are not equal to a scalar float
    /// @param other The float to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator!=(const float other) const;

    /// @brief Compare if elements are not equal to a tensor
    /// @param other The tensor to compare against
    /// @return A JaggedTensor with boolean results
    JaggedTensor operator!=(const torch::Tensor &other) const;

    /// @brief Compute the square root of each element
    /// @return A new JaggedTensor with square root values
    JaggedTensor sqrt() const;

    /// @brief Compute the absolute value of each element
    /// @return A new JaggedTensor with absolute values
    JaggedTensor abs() const;

    /// @brief Round each element to the specified number of decimal places
    /// @param decimals Number of decimal places to round to
    /// @return A new JaggedTensor with rounded values
    JaggedTensor round(int decimals = 0) const;

    /// @brief Compute the floor of each element
    /// @return A new JaggedTensor with floor values
    JaggedTensor floor() const;

    /// @brief Compute the ceiling of each element
    /// @return A new JaggedTensor with ceiling values
    JaggedTensor ceil() const;

    /// @brief Compute the square root of each element in-place
    /// @return Reference to this JaggedTensor
    JaggedTensor &sqrt_();

    /// @brief Compute the absolute value of each element in-place
    /// @return Reference to this JaggedTensor
    JaggedTensor &abs_();

    /// @brief Round each element to the specified number of decimal places in-place
    /// @param decimals Number of decimal places to round to
    /// @return Reference to this JaggedTensor
    JaggedTensor &round_(int decimals = 0);

    /// @brief Compute the floor of each element in-place
    /// @return Reference to this JaggedTensor
    JaggedTensor &floor_();

    /// @brief Compute the ceiling of each element in-place
    /// @return Reference to this JaggedTensor
    JaggedTensor &ceil_();

    /// @brief Set whether this JaggedTensor requires gradients
    /// @param requires_grad Whether gradients should be computed
    /// @return Reference to this JaggedTensor
    const JaggedTensor &set_requires_grad(bool requires_grad) const;

    /// @brief Check if this JaggedTensor requires gradients
    /// @return True if gradients are required, false otherwise
    bool requires_grad() const;

    /// @brief Detach this JaggedTensor from the computation graph
    /// @return A new JaggedTensor detached from gradients
    JaggedTensor detach() const;

    /// @brief Create a deep copy of this JaggedTensor
    /// @return A new JaggedTensor with copied data
    JaggedTensor clone() const;
};

} // namespace fvdb

#endif // FVDB_JAGGEDTENSOR_H
