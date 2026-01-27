// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>

#include <gtest/gtest.h>

#include <cmath>

namespace fvdb::detail::ops {
namespace {

using Mat3f = nanovdb::math::Mat3<float>;
using Vec4f = nanovdb::math::Vec4<float>;
using Vec3f = nanovdb::math::Vec3<float>;

// Minimal math helpers (avoid pulling in <cmath> from a .cu test; CUDA provides these).
__host__ __device__ inline float
mySqrt(float x) {
#if defined(__CUDA_ARCH__)
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

__host__ __device__ inline float
mySin(float x) {
#if defined(__CUDA_ARCH__)
    return sinf(x);
#else
    return std::sin(x);
#endif
}

__host__ __device__ inline float
myCos(float x) {
#if defined(__CUDA_ARCH__)
    return cosf(x);
#else
    return std::cos(x);
#endif
}

inline Mat3f
quatToRotationMatrixHost(const Vec4f &q_wxyz) {
    // Normalize the quaternion
    float w = q_wxyz[0], x = q_wxyz[1], y = q_wxyz[2], z = q_wxyz[3];
    const float n2 = w * w + x * x + y * y + z * z;
    if (n2 > 0.0f) {
        const float invN = 1.0f / mySqrt(n2);
        w *= invN;
        x *= invN;
        y *= invN;
        z *= invN;
    } else {
        w = 1.0f;
        x = y = z = 0.0f;
    }

    const float x2 = x * x, y2 = y * y, z2 = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;

    return Mat3f((1.0f - 2.0f * (y2 + z2)),
                 (2.0f * (xy - wz)),
                 (2.0f * (xz + wy)),       // 1st row
                 (2.0f * (xy + wz)),
                 (1.0f - 2.0f * (x2 + z2)),
                 (2.0f * (yz - wx)),       // 2nd row
                 (2.0f * (xz - wy)),
                 (2.0f * (yz + wx)),
                 (1.0f - 2.0f * (x2 + y2)) // 3rd row
    );
}

inline void
expectMatNear(const Mat3f &A, const Mat3f &B, float tol) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(A[r][c], B[r][c], tol) << "Mismatch at (" << r << "," << c << ")";
        }
    }
}

inline Vec4f
axisAngleToQuatWxyz(float ax, float ay, float az, float angleRad) {
    const float n = mySqrt(ax * ax + ay * ay + az * az);
    if (n <= 0.0f)
        return Vec4f(1.0f, 0.0f, 0.0f, 0.0f);
    ax /= n;
    ay /= n;
    az /= n;

    const float half = 0.5f * angleRad;
    const float s    = mySin(half);
    const float c    = myCos(half);
    return Vec4f(c, s * ax, s * ay, s * az);
}

inline void
expectVecNear(const Vec3f &a, const Vec3f &b, float tol) {
    EXPECT_NEAR(a[0], b[0], tol);
    EXPECT_NEAR(a[1], b[1], tol);
    EXPECT_NEAR(a[2], b[2], tol);
}

inline void
expectQuatNear(const Vec4f &a, const Vec4f &b, float tol) {
    EXPECT_NEAR(a[0], b[0], tol);
    EXPECT_NEAR(a[1], b[1], tol);
    EXPECT_NEAR(a[2], b[2], tol);
    EXPECT_NEAR(a[3], b[3], tol);
}

inline Vec4f
normalizeQuat(Vec4f q) {
    const float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (n2 > 0.0f) {
        const float invN = 1.0f / mySqrt(n2);
        q[0] *= invN;
        q[1] *= invN;
        q[2] *= invN;
        q[3] *= invN;
    } else {
        q[0] = 1.0f;
        q[1] = q[2] = q[3] = 0.0f;
    }
    return q;
}

inline float
quatDot(const Vec4f &a, const Vec4f &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

inline Vec4f
nlerpRefShortestPath(const Vec4f &q0, Vec4f q1, float u) {
    float dot = quatDot(q0, q1);
    if (dot < 0.0f) {
        q1[0] = -q1[0];
        q1[1] = -q1[1];
        q1[2] = -q1[2];
        q1[3] = -q1[3];
        dot   = -dot;
    }
    (void)dot; // suppress unused variable warning
    const float s = 1.0f - u;
    return normalizeQuat(Vec4f(s * q0[0] + u * q1[0],
                               s * q0[1] + u * q1[1],
                               s * q0[2] + u * q1[2],
                               s * q0[3] + u * q1[3]));
}

} // namespace

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_Identity) {
    const Mat3f R(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    const Vec4f q = rotationMatrixToQuaternion<float>(R);

    EXPECT_NEAR(q[0], 1.0f, 1e-6f);
    EXPECT_NEAR(q[1], 0.0f, 1e-6f);
    EXPECT_NEAR(q[2], 0.0f, 1e-6f);
    EXPECT_NEAR(q[3], 0.0f, 1e-6f);

    const float n = mySqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    EXPECT_NEAR(n, 1.0f, 1e-6f);
    EXPECT_GE(q[0], 0.0f); // sign convention
}

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_RoundTrip_KnownAxes) {
    const float pi   = 3.14159265358979323846f;
    const Vec4f qs[] = {
        axisAngleToQuatWxyz(1.0f, 0.0f, 0.0f, 0.5f * pi), // +90° about X
        axisAngleToQuatWxyz(0.0f, 1.0f, 0.0f, 0.5f * pi), // +90° about Y
        axisAngleToQuatWxyz(0.0f, 0.0f, 1.0f, 0.5f * pi), // +90° about Z
        axisAngleToQuatWxyz(0.0f, 1.0f, 0.0f, 1.0f * pi), // 180° about Y (w=0 edge case)
    };

    for (const auto &q_in: qs) {
        const Mat3f R_in  = quatToRotationMatrixHost(q_in);
        const Vec4f q_out = rotationMatrixToQuaternion<float>(R_in);
        const Mat3f R_out = quatToRotationMatrixHost(q_out);

        expectMatNear(R_in, R_out, 2e-5f);
        EXPECT_GE(q_out[0], 0.0f);
    }
}

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_ProducesPositiveWForEquivalentRotation) {
    const float pi    = 3.14159265358979323846f;
    const Vec4f q     = axisAngleToQuatWxyz(0.0f, 0.0f, 1.0f, pi / 3.0f);
    const Vec4f q_neg = Vec4f(-q[0], -q[1], -q[2], -q[3]);

    const Mat3f R     = quatToRotationMatrixHost(q_neg);
    const Vec4f q_out = rotationMatrixToQuaternion<float>(R);
    EXPECT_GE(q_out[0], 0.0f);

    const Mat3f R_out = quatToRotationMatrixHost(q_out);
    expectMatNear(R, R_out, 2e-5f);
}

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_RoundTrip_DeterministicSamples) {
    const float pi = 3.14159265358979323846f;
    const struct Sample {
        float ax, ay, az, ang;
    } samples[] = {
        {0.3f, 0.7f, -0.2f, 0.1f * pi},
        {-0.8f, 0.1f, 0.5f, 0.25f * pi},
        {0.1f, -0.2f, 0.9f, 0.8f * pi},
        {0.9f, 0.4f, 0.1f, 1.1f * pi},
        {-0.4f, -0.6f, 0.2f, 0.6f * pi},
        {0.2f, -0.9f, -0.3f, 1.7f * pi},
        {-0.1f, 0.5f, 0.8f, 0.33f * pi},
        {0.6f, -0.3f, 0.7f, 1.9f * pi},
    };

    for (const auto &s: samples) {
        const Vec4f q_in  = axisAngleToQuatWxyz(s.ax, s.ay, s.az, s.ang);
        const Mat3f R_in  = quatToRotationMatrixHost(q_in);
        const Vec4f q_out = rotationMatrixToQuaternion<float>(R_in);
        const Mat3f R_out = quatToRotationMatrixHost(q_out);

        expectMatNear(R_in, R_out, 2e-5f);
        EXPECT_GE(q_out[0], 0.0f);
    }
}

TEST(GaussianUtilsTest, InterpolatePose_NlerpMatchesReference) {
    const float pi      = 3.14159265358979323846f;
    const Vec4f q_start = axisAngleToQuatWxyz(1.0f, 0.0f, 0.0f, pi / 3.0f);        // 60deg about X
    const Vec4f q_end   = axisAngleToQuatWxyz(0.0f, 1.0f, 0.0f, 2.0f * pi / 3.0f); // 120deg about Y

    const Mat3f R_start = quatToRotationMatrixHost(q_start);
    const Mat3f R_end   = quatToRotationMatrixHost(q_end);

    const Vec3f t_start(1.0f, 2.0f, 3.0f);
    const Vec3f t_end(-4.0f, 5.0f, 0.5f);
    const float u = 0.25f;

    const Pose<float> pose = interpolatePose<float>(u, R_start, t_start, R_end, t_end);

    const Vec4f q0    = rotationMatrixToQuaternion<float>(R_start);
    const Vec4f q1    = rotationMatrixToQuaternion<float>(R_end);
    const Vec4f q_ref = nlerpRefShortestPath(q0, q1, u);

    expectQuatNear(pose.q, q_ref, 2e-6f);
    expectVecNear(pose.t, t_start + u * (t_end - t_start), 1e-6f);
}

TEST(GaussianUtilsTest, Pose_WorldToCamAndBack_RoundTrip) {
    const float pi = 3.14159265358979323846f;

    // Non-identity rotation + non-zero translation to catch ordering bugs.
    const Vec4f q = axisAngleToQuatWxyz(0.2f, 0.9f, -0.4f, 0.37f * pi);
    const Vec3f t(1.25f, -2.5f, 0.75f);
    const Pose<float> pose(q, t);

    const Vec3f p_world(0.3f, -1.1f, 2.7f);
    const Vec3f p_cam = pose.transformPointWorldToCam(p_world);
    const Vec3f p_world_rt = pose.transformPointCamToWorld(p_cam);
    expectVecNear(p_world_rt, p_world, 2e-5f);

    // Also verify the opposite direction for completeness.
    const Vec3f p_cam_in(-0.2f, 0.4f, 1.8f);
    const Vec3f p_world_from_cam = pose.transformPointCamToWorld(p_cam_in);
    const Vec3f p_cam_rt = pose.transformPointWorldToCam(p_world_from_cam);
    expectVecNear(p_cam_rt, p_cam_in, 2e-5f);
}

} // namespace fvdb::detail::ops
