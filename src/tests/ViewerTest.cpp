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
    std::string ply_paths[] = {};

    std::string view_names[] = {};

    torch::Device device(torch::kCUDA);

    std::vector<
        std::tuple<fvdb::GaussianSplat3d,
                   std::unordered_map<std::string, fvdb::GaussianSplat3d::PlyMetadataTypes>>>
        loadedData;

    for (size_t i = 0; i < std::size(ply_paths) && i < std::size(view_names); ++i) {
        const std::string &ply_path = ply_paths[i];

        printf("Loading splats from %s\n", ply_path.c_str());
        auto loaded = fvdb::detail::io::loadGaussianPly(ply_path, device);
        loadedData.push_back(loaded);
    }

    int N = 100;

    for (int i = 0; i < N; ++i) {
        printf("Iteration %d / %d\n", i + 1, N);
        for (size_t i = 0; i < std::size(ply_paths) && i < std::size(view_names); ++i) {
            const std::string &ply_path  = ply_paths[i];
            const std::string &view_name = view_names[i];

            auto [splats, metadata] = loadedData[i];

            printf("Adding splats from %s\n", ply_path.c_str());
            fvdb::detail::viewer::GaussianSplat3dView &view =
                viewer.addGaussianSplat3d(view_name, splats);
            view.setFar(1.f);

            torch::Tensor cameraToWorld =
                std::get<torch::Tensor>(metadata.at("camera_to_world_matrices"));
            torch::Tensor projectionMat =
                std::get<torch::Tensor>(metadata.at("projection_matrices"));

            if (cameraToWorld._is_zerotensor() || projectionMat._is_zerotensor()) {
                continue;
            }

            torch::Tensor imageSizes;
            if (metadata.find("image_sizes") != metadata.end()) {
                imageSizes = std::get<torch::Tensor>(metadata.at("image_sizes"));
            } else {
                imageSizes = torch::empty({0}, device);
            }

            fvdb::detail::viewer::CameraView &cameraView = viewer.addCameraView(
                view_name, cameraToWorld, projectionMat, imageSizes, 0.f, 0.5f);

            std::this_thread::sleep_for(std::chrono::seconds(5));

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
        }
    }
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
