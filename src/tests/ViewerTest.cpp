// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// A test for the Viewer class.

#include "fvdb/detail/viewer/GaussianSplat3dView.h"

#include <fvdb/detail/io/GaussianPlyIO.h>
#include <fvdb/detail/viewer/Viewer.h>

#include <c10/core/DeviceType.h>

#include <gtest/gtest.h>

// #define LOCAL_TESTING

TEST(Viewer, ViewerTest) {
    fvdb::detail::viewer::Viewer viewer = fvdb::detail::viewer::Viewer("127.0.0.1", 8080, false);

#ifdef LOCAL_TESTING
    std::string ply_path = "";
    torch::Device device(torch::kCUDA);
    auto [splats, metadata] = fvdb::detail::io::loadGaussianPly(ply_path, device);

    fvdb::detail::viewer::GaussianSplat3dView &view =
        viewer.addGaussianSplat3d("test_view", splats);
    view.setFar(1000000.0f);

    torch::Tensor cameraToWorld = std::get<torch::Tensor>(metadata.at("camera_to_world_matrices"));
    torch::Tensor projectionMat = std::get<torch::Tensor>(metadata.at("projection_matrices"));

    float frustumNear = 0.3f;
    float frustumFar  = 1.0f;

    fvdb::detail::viewer::CameraView &cameraView = viewer.addCameraView(
        "test_camera_view", cameraToWorld, projectionMat, frustumNear, frustumFar);

    const float axisLength = 0.5f;
    cameraView.setAxisLength(axisLength);
    ASSERT_FLOAT_EQ(cameraView.getAxisLength(), axisLength);

    const float axisThickness = 0.0125f;
    cameraView.setAxisThickness(axisThickness);
    ASSERT_FLOAT_EQ(cameraView.getAxisThickness(), axisThickness);

    const float frustumLineWidth = 2.0f;
    cameraView.setFrustumLineWidth(frustumLineWidth);
    ASSERT_FLOAT_EQ(cameraView.getFrustumLineWidth(), frustumLineWidth);

    const float frustumScale = 1.f;
    cameraView.setFrustumScale(frustumScale);
    ASSERT_FLOAT_EQ(cameraView.getFrustumScale(), frustumScale);
#else
    const int N = 1000;
    torch::Device device(torch::kCUDA);
    torch::Tensor means          = torch::rand({N, 3}, device);
    torch::Tensor quats          = torch::rand({N, 4}, device);
    torch::Tensor logScales      = torch::rand({N, 3}, device);
    torch::Tensor logitOpacities = torch::rand({N}, device);
    torch::Tensor sh0            = torch::rand({N, 1, 3}, device);
    torch::Tensor shN            = torch::rand({N, 15, 3}, device);

    fvdb::GaussianSplat3d splats(
        means, quats, logScales, logitOpacities, sh0, shN, false, false, false);

    fvdb::detail::viewer::GaussianSplat3dView &view =
        viewer.addGaussianSplat3d("test_view", splats);

    const float testNear = 0.5f;
    view.setNear(testNear);
    ASSERT_FLOAT_EQ(view.getNear(), testNear);

    const float testFar = 1000000.0f;
    view.setFar(testFar);
    ASSERT_FLOAT_EQ(view.getFar(), testFar);

    const float testEps2d = 0.5f;
    view.setEps2d(testEps2d);
    ASSERT_FLOAT_EQ(view.getEps2d(), testEps2d);

    const float testMinRadius2d = 0.5f;
    view.setMinRadius2d(testMinRadius2d);
    ASSERT_FLOAT_EQ(view.getMinRadius2d(), testMinRadius2d);

    const float testTileSize = 16;
    view.setTileSize(testTileSize);
    ASSERT_EQ(view.getTileSize(), testTileSize);

    const int testShDegree = 1;
    view.setShDegreeToUse(testShDegree);
    ASSERT_EQ(view.getShDegreeToUse(), testShDegree);
#endif

#ifdef LOCAL_TESTING
    std::this_thread::sleep_for(std::chrono::seconds(10000));
#endif
}
