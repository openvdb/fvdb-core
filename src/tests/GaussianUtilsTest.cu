// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>

#include <gtest/gtest.h>

#include <cmath>

namespace fvdb::detail::ops {
namespace {

using Mat3f = nanovdb::math::Mat3<float>;
using Vec4f = nanovdb::math::Vec4<float>;

// Minimal math helpers (avoid pulling in <cmath> from a .cu test; CUDA provides these).
__host__ __device__ inline float
mySqrt(float x)
{
#if defined(__CUDA_ARCH__)
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

__host__ __device__ inline float
mySin(float x)
{
#if defined(__CUDA_ARCH__)
    return sinf(x);
#else
    return std::sin(x);
#endif
}

__host__ __device__ inline float
myCos(float x)
{
#if defined(__CUDA_ARCH__)
    return cosf(x);
#else
    return std::cos(x);
#endif
}

inline Mat3f
quatToRotationMatrixHost(const Vec4f& q_wxyz)
{
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
                 (2.0f * (xz + wy)), // 1st row
                 (2.0f * (xy + wz)),
                 (1.0f - 2.0f * (x2 + z2)),
                 (2.0f * (yz - wx)), // 2nd row
                 (2.0f * (xz - wy)),
                 (2.0f * (yz + wx)),
                 (1.0f - 2.0f * (x2 + y2)) // 3rd row
    );
}

inline void
expectMatNear(const Mat3f& A, const Mat3f& B, float tol)
{
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(A[r][c], B[r][c], tol) << "Mismatch at (" << r << "," << c << ")";
        }
    }
}

inline Vec4f
axisAngleToQuatWxyz(float ax, float ay, float az, float angleRad)
{
    const float n = mySqrt(ax * ax + ay * ay + az * az);
    if (n <= 0.0f) return Vec4f(1.0f, 0.0f, 0.0f, 0.0f);
    ax /= n;
    ay /= n;
    az /= n;

    const float half = 0.5f * angleRad;
    const float s    = mySin(half);
    const float c    = myCos(half);
    return Vec4f(c, s * ax, s * ay, s * az);
}

} // namespace

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_Identity)
{
    const Mat3f R(1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 1.0f);
    const Vec4f q = rotationMatrixToQuaternion<float>(R);

    EXPECT_NEAR(q[0], 1.0f, 1e-6f);
    EXPECT_NEAR(q[1], 0.0f, 1e-6f);
    EXPECT_NEAR(q[2], 0.0f, 1e-6f);
    EXPECT_NEAR(q[3], 0.0f, 1e-6f);

    const float n = mySqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    EXPECT_NEAR(n, 1.0f, 1e-6f);
    EXPECT_GE(q[0], 0.0f); // sign convention
}

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_RoundTrip_KnownAxes)
{
    const float pi = 3.14159265358979323846f;
    const Vec4f qs[] = {
        axisAngleToQuatWxyz(1.0f, 0.0f, 0.0f, 0.5f * pi), // +90° about X
        axisAngleToQuatWxyz(0.0f, 1.0f, 0.0f, 0.5f * pi), // +90° about Y
        axisAngleToQuatWxyz(0.0f, 0.0f, 1.0f, 0.5f * pi), // +90° about Z
        axisAngleToQuatWxyz(0.0f, 1.0f, 0.0f, 1.0f * pi), // 180° about Y (w=0 edge case)
    };

    for (const auto& q_in : qs) {
        const Mat3f R_in  = quatToRotationMatrixHost(q_in);
        const Vec4f q_out = rotationMatrixToQuaternion<float>(R_in);
        const Mat3f R_out = quatToRotationMatrixHost(q_out);

        expectMatNear(R_in, R_out, 2e-5f);
        EXPECT_GE(q_out[0], 0.0f);
    }
}

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_ProducesPositiveWForEquivalentRotation)
{
    const float pi = 3.14159265358979323846f;
    const Vec4f q     = axisAngleToQuatWxyz(0.0f, 0.0f, 1.0f, pi / 3.0f);
    const Vec4f q_neg = Vec4f(-q[0], -q[1], -q[2], -q[3]);

    const Mat3f R     = quatToRotationMatrixHost(q_neg);
    const Vec4f q_out = rotationMatrixToQuaternion<float>(R);
    EXPECT_GE(q_out[0], 0.0f);

    const Mat3f R_out = quatToRotationMatrixHost(q_out);
    expectMatNear(R, R_out, 2e-5f);
}

TEST(GaussianUtilsTest, RotationMatrixToQuaternion_RoundTrip_DeterministicSamples)
{
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

    for (const auto& s : samples) {
        const Vec4f q_in  = axisAngleToQuatWxyz(s.ax, s.ay, s.az, s.ang);
        const Mat3f R_in  = quatToRotationMatrixHost(q_in);
        const Vec4f q_out = rotationMatrixToQuaternion<float>(R_in);
        const Mat3f R_out = quatToRotationMatrixHost(q_out);

        expectMatNear(R_in, R_out, 2e-5f);
        EXPECT_GE(q_out[0], 0.0f);
    }
}

} // namespace fvdb::detail::ops

