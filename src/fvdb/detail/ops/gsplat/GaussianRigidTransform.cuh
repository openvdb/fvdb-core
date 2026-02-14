// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRIGIDTRANSFORM_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRIGIDTRANSFORM_CUH

#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>

#include <nanovdb/math/Math.h>

namespace fvdb::detail::ops {

/// @brief Rigid transform (cached rotation + translation).
///
/// Quaternion is stored as \([w,x,y,z]\) and is assumed to represent a rotation.
/// The corresponding rotation matrix \(R(q)\) is cached to avoid recomputing it for every point
/// transform (UT sigma points, rolling-shutter iterations, ray generation, etc.).
template <typename T> struct RigidTransform {
    nanovdb::math::Mat3<T> R;
    nanovdb::math::Vec4<T> q;
    nanovdb::math::Vec3<T> t;

    /// @brief Default constructor (identity transform).
    ///
    /// Initializes to unit quaternion \([1,0,0,0]\) and zero translation.
    inline __host__ __device__
    RigidTransform()
        : R(nanovdb::math::Mat3<T>(nanovdb::math::Vec3<T>(T(1), T(0), T(0)),
                                   nanovdb::math::Vec3<T>(T(0), T(1), T(0)),
                                   nanovdb::math::Vec3<T>(T(0), T(0), T(1)))),
          q(T(1), T(0), T(0), T(0)), t(T(0), T(0), T(0)) {}

    /// @brief Construct from quaternion and translation.
    /// @param[in] q_in Rotation quaternion \([w,x,y,z]\).
    /// @param[in] t_in Translation vector.
    inline __host__ __device__
    RigidTransform(const nanovdb::math::Vec4<T> &q_in, const nanovdb::math::Vec3<T> &t_in)
        : R(quaternionToRotationMatrix<T>(q_in)), q(q_in), t(t_in) {}

    /// @brief Construct from rotation matrix and translation.
    /// @param[in] R_in Rotation matrix.
    /// @param[in] t_in Translation vector.
    inline __host__ __device__
    RigidTransform(const nanovdb::math::Mat3<T> &R_in, const nanovdb::math::Vec3<T> &t_in)
        : R(R_in), q(rotationMatrixToQuaternion<T>(R_in)), t(t_in) {}

    /// @brief Apply the transform to a 3D point: \(R(q)\,p + t\).
    inline __host__ __device__ nanovdb::math::Vec3<T>
    apply(const nanovdb::math::Vec3<T> &p_world) const {
        // p_cam = R * p_world + t
        return R * p_world + t;
    }

    /// @brief Interpolate between two rigid transforms.
    ///
    /// Translation is linearly interpolated; rotation uses NLERP along the shortest arc.
    inline static __host__ __device__ RigidTransform<T>
    interpolate(const T u, const RigidTransform<T> &start, const RigidTransform<T> &end) {
        const nanovdb::math::Vec3<T> t_interp = start.t + u * (end.t - start.t);
        const nanovdb::math::Vec4<T> q_interp = nlerpQuaternionShortestPath<T>(start.q, end.q, u);
        return RigidTransform<T>(q_interp, t_interp);
    }
};

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANRIGIDTRANSFORM_CUH
