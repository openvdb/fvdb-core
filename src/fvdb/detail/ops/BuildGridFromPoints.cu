// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/BuilderResource.h>
#include <fvdb/GridBatchData.h>
#include <fvdb/detail/GridBatchDataFactory.h>
#include <fvdb/detail/ops/BuildGridFromIjk.h>
#include <fvdb/detail/ops/BuildGridFromPoints.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/nanovdb/CreateEmptyGridHandle.h>

#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <thrust/universal_vector.h>

#include <limits>

namespace fvdb {
namespace detail {
namespace ops {

template <torch::DeviceType>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints(const JaggedTensor &points,
                            const std::vector<VoxelCoordTransform> &txs);

namespace {

/// @brief Pointer-like adaptor that yields index-space voxel coordinates from world-space points
///        on demand, so the coordinates never have to be materialized in memory.
///
/// `nanovdb::tools::cuda::voxelsToGrid` reads its input through a pointer-like type rather than a
/// raw pointer (see `nanovdb::tools::cuda::fancy_ptr`, which documents the contract): `operator[]`
/// is what the kernels actually call, and `operator*` exists only so `pointer_traits` can deduce
/// `element_type` from its return type. Returning `nanovdb::Coord` by value satisfies both and
/// selects the `is_same<Vec3T, Coord>` branch of `TileKeyFunctor` — the same branch a raw
/// `nanovdb::Coord *` takes — so the resulting topology is bit-for-bit what the previous
/// materialize-then-sort path produced.
///
/// This replaces a `torch::empty({N, 3}, kInt32)` coordinate tensor (12 B per input point) that
/// was written once and read once.
///
/// The class supports both memory layouts of an (N, 3) tensor so that non-contiguous views --
/// e.g. the xyz columns of an (N, 6) point cloud, `cloud[:, :3]` -- are read in place rather
/// than copied. `IsContiguous` selects between compile-time strides (3, 1) and runtime strides
/// taken from the tensor, so the packed fast path pays nothing for the generality.
///
/// @tparam ScalarT Scalar type of the input points (floating point or half)
/// @tparam IsContiguous Whether the points are a packed (N, 3) array; when false, element
///         strides are supplied at construction
template <typename ScalarT, bool IsContiguous> class TransformedPointPtr {
  public:
    using MathT = typename at::opmath_type<ScalarT>;

    /// @param points Pointer to the first component of the first point
    /// @param transform World-space to index-space transform for this batch item
    /// @param rowStride Distance in elements between consecutive points (3 if packed)
    /// @param colStride Distance in elements between a point's components (1 if packed)
    __hostdev__
    TransformedPointPtr(const ScalarT *points,
                        const VoxelCoordTransform &transform,
                        int64_t rowStride = 3,
                        int64_t colStride = 1)
        : mPoints(points), mTransform(transform), mRowStride(rowStride), mColStride(colStride) {}

    /// @brief Return the index-space voxel coordinate of the i'th point. Required by PointsToGrid.
    __hostdev__ inline nanovdb::Coord
    operator[](size_t i) const {
        MathT x, y, z;
        if constexpr (IsContiguous) {
            const ScalarT *point = mPoints + 3 * i;
            x                    = static_cast<MathT>(point[0]);
            y                    = static_cast<MathT>(point[1]);
            z                    = static_cast<MathT>(point[2]);
        } else {
            const ScalarT *point = mPoints + static_cast<int64_t>(i) * mRowStride;
            x                    = static_cast<MathT>(point[0]);
            y                    = static_cast<MathT>(point[mColStride]);
            z                    = static_cast<MathT>(point[2 * mColStride]);
        }
        return mTransform.apply(x, y, z).round();
    }

    /// @brief Required by `pointer_traits` to deduce `element_type` -- only the return *type* is
    ///        used. Deliberately does not read `mPoints`: `pointer_traits` never evaluates this,
    ///        and returning a default coordinate keeps it well defined for an empty point set if
    ///        some future code path (or debug instrumentation) ever does evaluate it.
    __hostdev__ inline nanovdb::Coord
    operator*() const {
        return nanovdb::Coord();
    }

  private:
    const ScalarT *mPoints;
    VoxelCoordTransform mTransform;
    int64_t mRowStride;
    int64_t mColStride;
};

} // namespace

template <typename ScalarT>
__device__ void
ijkForPointsCallback(int32_t bidx,
                     int32_t eidx,
                     const JaggedRAcc64<ScalarT, 2> points,
                     const VoxelCoordTransform *transforms,
                     TorchRAcc64<int32_t, 2> outIJKData) {
    using MathT                          = typename at::opmath_type<ScalarT>;
    const auto &point                    = points.data()[eidx];
    const VoxelCoordTransform &transform = transforms[bidx];
    const nanovdb::Coord ijk0            = transform
                                    .apply(static_cast<MathT>(point[0]),
                                           static_cast<MathT>(point[1]),
                                           static_cast<MathT>(point[2]))
                                    .round();
    outIJKData[eidx][0] = ijk0[0];
    outIJKData[eidx][1] = ijk0[1];
    outIJKData[eidx][2] = ijk0[2];
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints<torch::kCUDA>(const JaggedTensor &points,
                                          const std::vector<VoxelCoordTransform> &txs) {
    using GridT = nanovdb::ValueOnIndex;

    TORCH_CHECK(points.device().is_cuda(), "points must be on a CUDA device");
    TORCH_CHECK(points.device().has_index(), "points device must have a valid index");

    c10::cuda::CUDAGuard deviceGuard(points.device());

    // This guide buffer is a hack to pass in a device with an index to the grid building
    // functions. We can't pass in a device directly but we can pass in a buffer which gets
    // passed to TorchDeviceBuffer::create. The guide buffer holds the device and effectively
    // passes it to the created buffer.
    TorchDeviceBuffer guide(0, points.device());

    // FIXME: Same host sync as _createNanoGridFromIJK -- the per-batch-item loop below needs the
    // offsets on the host in order to slice the point array. Ideally this would be a single
    // invocation over the whole batch.
    const torch::Tensor pointsBOffsetTensor = points.joffsets().cpu();
    const auto pointsBOffset                = pointsBOffsetTensor.accessor<fvdb::JOffsetsType, 1>();

    // The loop below indexes txs by batch item, so make the invariant explicit rather than
    // relying on the caller. joffsets holds one offset per batch item plus a terminating entry.
    TORCH_CHECK(pointsBOffset.size(0) >= 1,
                "Expected joffsets to have at least one entry, got ",
                pointsBOffset.size(0));
    TORCH_CHECK(static_cast<int64_t>(txs.size()) == pointsBOffset.size(0) - 1,
                "Expected one transform per batch item, but got ",
                txs.size(),
                " transforms for ",
                pointsBOffset.size(0) - 1,
                " batch items");

    // TransformedPointPtr reads the points in place for both layouts: compile-time strides for
    // a packed (N, 3) array, runtime strides otherwise (e.g. the xyz columns of an (N, 6)
    // tensor). Neither path copies the point data.
    const torch::Tensor pointsData = points.jdata();
    const bool pointsAreContiguous = pointsData.is_contiguous();
    const int64_t rowStride        = pointsData.stride(0);
    const int64_t colStride        = pointsData.stride(1);

    return AT_DISPATCH_V2(
        points.scalar_type(),
        "buildGridFromPoints",
        AT_WRAP([&]() -> nanovdb::GridHandle<TorchDeviceBuffer> {
            const scalar_t *pointsPtr = pointsData.data_ptr<scalar_t>();

            // Create a grid for each batch item and store the handles
            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> handles;
            handles.reserve(pointsBOffset.size(0) - 1);
            for (int64_t i = 0; i < (pointsBOffset.size(0) - 1); i += 1) {
                const int64_t startIdx = pointsBOffset[i];
                const int64_t nPoints  = pointsBOffset[i + 1] - startIdx;

                // PointsToGrid casts its element count to int for the segmented radix sort, so
                // anything at or above 2^31 would silently produce a corrupt grid.
                TORCH_CHECK(nPoints <= std::numeric_limits<int32_t>::max(),
                            "Cannot build a grid from ",
                            nPoints,
                            " points in a single batch item (limit is ",
                            std::numeric_limits<int32_t>::max(),
                            ")");

                if (nPoints == 0) {
                    handles.push_back(createEmptyGridHandle(guide.device()));
                } else if (pointsAreContiguous) {
                    using PointPtrT = TransformedPointPtr<scalar_t, true>;
                    handles.push_back(
                        nanovdb::tools::cuda::
                            voxelsToGrid<GridT, PointPtrT, TorchDeviceBuffer, BuilderResource>(
                                PointPtrT(pointsPtr + 3 * startIdx, txs[i]), nPoints, 1.0, guide));
                } else {
                    using PointPtrT = TransformedPointPtr<scalar_t, false>;
                    handles.push_back(
                        nanovdb::tools::cuda::
                            voxelsToGrid<GridT, PointPtrT, TorchDeviceBuffer, BuilderResource>(
                                PointPtrT(
                                    pointsPtr + startIdx * rowStride, txs[i], rowStride, colStride),
                                nPoints,
                                1.0,
                                guide));
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }

            if (handles.size() == 1) {
                // If there's only one handle, just return it
                return std::move(handles[0]);
            } else {
                // This copies all the handles into a single handle -- only do it if there are
                // multiple grids
                return nanovdb::cuda::mergeGridHandles(handles, &guide);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints<torch::kPrivateUse1>(const JaggedTensor &points,
                                                 const std::vector<VoxelCoordTransform> &txs) {
    TORCH_CHECK(points.device().is_privateuseone(), "GridBatchData must be on PrivateUse1 device");

    const torch::TensorOptions deviceOptions = torch::TensorOptions().device(points.device());
    const torch::TensorOptions ijkOptions    = deviceOptions.dtype(torch::kInt32);

    torch::Tensor ijk = torch::empty({points.jdata().size(0), 3}, ijkOptions);
    auto ijkAcc       = ijk.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();

    thrust::universal_vector<VoxelCoordTransform> transforms(txs.size());
    auto transformsPtr = transforms.data().get();
    cudaMemcpy(
        transformsPtr, txs.data(), sizeof(VoxelCoordTransform) * txs.size(), cudaMemcpyDefault);

    AT_DISPATCH_V2(points.scalar_type(),
                   "ijkForPoints",
                   AT_WRAP([&] {
                       auto cb = [=] __device__(int32_t bidx,
                                                int32_t eidx,
                                                int32_t cidx,
                                                JaggedRAcc64<scalar_t, 2> pacc) {
                           ijkForPointsCallback(bidx, eidx, pacc, transformsPtr, ijkAcc);
                       };
                       forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, points, cb);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf);

    JaggedTensor coords = points.jagged_like(ijk);
    return ops::_createNanoGridFromIJK(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints<torch::kCPU>(const JaggedTensor &pointsJagged,
                                         const std::vector<VoxelCoordTransform> &txs) {
    using GridT = nanovdb::ValueOnIndex;
    return AT_DISPATCH_V2(
        pointsJagged.scalar_type(),
        "buildPaddedGridFromPoints",
        AT_WRAP([&]() {
            using ScalarT = scalar_t;
            static_assert(is_floating_point_or_half<ScalarT>::value,
                          "Invalid type for points, must be floating point");
            using MathT      = typename at::opmath_type<ScalarT>;
            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            pointsJagged.check_valid();

            const torch::TensorAccessor<ScalarT, 2> &pointsAcc =
                pointsJagged.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &pointsBOffsetsAcc =
                pointsJagged.joffsets().accessor<fvdb::JOffsetsType, 1>();

            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
            batchHandles.reserve(pointsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (pointsBOffsetsAcc.size(0) - 1); bi += 1) {
                VoxelCoordTransform tx = txs[bi];

                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = pointsBOffsetsAcc[bi];
                const int64_t end   = pointsBOffsetsAcc[bi + 1];

                for (int64_t pi = start; pi < end; pi += 1) {
                    nanovdb::Coord ijk = tx.apply(static_cast<MathT>(pointsAcc[pi][0]),
                                                  static_cast<MathT>(pointsAcc[pi][1]),
                                                  static_cast<MathT>(pointsAcc[pi][2]))
                                             .round();
                    proxyGridAccessor.setValue(ijk, 1.0f);
                }

                proxyGridAccessor.merge();
                auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridT, TorchDeviceBuffer>(
                    *proxyGrid, 0u, false, false);
                ret.buffer().to(torch::kCPU);
                batchHandles.push_back(std::move(ret));
            }

            if (batchHandles.size() == 1) {
                return std::move(batchHandles[0]);
            } else {
                return nanovdb::mergeGrids(batchHandles);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

c10::intrusive_ptr<GridBatchData>
buildGridFromPoints(const JaggedTensor &points,
                    const std::vector<nanovdb::Vec3d> &voxelSizes,
                    const std::vector<nanovdb::Vec3d> &origins) {
    TORCH_CHECK_VALUE(
        points.ldim() == 1,
        "Expected points to have 1 list dimension, i.e. be a single list of coordinate values, but got",
        points.ldim(),
        "list dimensions");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_VALUE(points.rdim() == 2,
                      std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                          std::to_string(points.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(points.rsize(1) == 3,
                      "Expected 3 dimensional points but got points.rshape[1] = " +
                          std::to_string(points.rsize(1)));
    TORCH_CHECK(
        points.num_tensors() == points.num_outer_lists(),
        "If this happens, Francis' paranoia about tensors and points was justified. File a bug");
    TORCH_CHECK_VALUE(points.num_outer_lists() <= GridBatchData::MAX_GRIDS_PER_BATCH,
                      "Cannot create a grid with more than ",
                      GridBatchData::MAX_GRIDS_PER_BATCH,
                      " grids in a batch. ",
                      "You passed in ",
                      points.num_outer_lists(),
                      " points sets.");
    const int64_t numGrids = points.joffsets().size(0) - 1;
    TORCH_CHECK(
        numGrids == points.num_outer_lists(),
        "If this happens, Francis' paranoia about grids and points was justified. File a bug");
    std::vector<VoxelCoordTransform> transforms;
    transforms.reserve(numGrids);
    for (int64_t i = 0; i < numGrids; i += 1) {
        transforms.push_back(primalVoxelTransformForSizeAndOrigin(voxelSizes[i], origins[i]));
    }
    auto handle = FVDB_DISPATCH_KERNEL(points.device(), [&]() {
        return dispatchBuildGridFromPoints<DeviceTag>(points, transforms);
    });
    return makeGridBatchData(std::move(handle), voxelSizes, origins);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
