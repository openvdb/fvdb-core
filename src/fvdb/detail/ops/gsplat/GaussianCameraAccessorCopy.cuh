// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAACCESSORCOPY_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAACCESSORCOPY_CUH

#include <nanovdb/math/Math.h>

#include <cstdint>

namespace fvdb::detail::ops {

template <typename ScalarType, typename Acc33>
inline __device__ void
copyMat3Accessor(const int64_t C,
                 nanovdb::math::Mat3<ScalarType> *__restrict__ out,
                 const Acc33 &acc /* [C,3,3] */) {
    constexpr int64_t kElementsPerMat3 = 9;
    for (int64_t i = threadIdx.x; i < C * kElementsPerMat3; i += blockDim.x) {
        const int64_t camId      = i / kElementsPerMat3;
        const int64_t entryId    = i % kElementsPerMat3;
        const int64_t rowId      = entryId / 3;
        const int64_t colId      = entryId % 3;
        out[camId][rowId][colId] = acc[camId][rowId][colId];
    }
}

template <typename ScalarType, typename Acc44>
inline __device__ void
copyWorldToCamRotation(const int64_t C,
                       nanovdb::math::Mat3<ScalarType> *__restrict__ out,
                       const Acc44 &acc /* [C,4,4] */) {
    constexpr int64_t kElementsPerMat3 = 9;
    for (int64_t i = threadIdx.x; i < C * kElementsPerMat3; i += blockDim.x) {
        const int64_t camId      = i / kElementsPerMat3;
        const int64_t entryId    = i % kElementsPerMat3;
        const int64_t rowId      = entryId / 3;
        const int64_t colId      = entryId % 3;
        out[camId][rowId][colId] = acc[camId][rowId][colId];
    }
}

template <typename ScalarType, typename Acc44>
inline __device__ void
copyWorldToCamTranslation(const int64_t C,
                          nanovdb::math::Vec3<ScalarType> *__restrict__ out,
                          const Acc44 &acc /* [C,4,4] */) {
    constexpr int64_t kElementsPerVec3 = 3;
    for (int64_t i = threadIdx.x; i < C * kElementsPerVec3; i += blockDim.x) {
        const int64_t camId   = i / kElementsPerVec3;
        const int64_t entryId = i % kElementsPerVec3;
        out[camId][entryId]   = acc[camId][entryId][3];
    }
}

template <typename ScalarType, typename AccCk>
inline __device__ void
copyDistortionCoeffs(const int64_t C,
                     const int64_t K,
                     ScalarType *__restrict__ out /* [C*K] */,
                     const AccCk &acc /* [C,K] */) {
    const int64_t total = C * K;
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) {
        const int64_t camId      = i / K;
        const int64_t entryId    = i % K;
        out[camId * K + entryId] = acc[camId][entryId];
    }
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANCAMERAACCESSORCOPY_CUH
