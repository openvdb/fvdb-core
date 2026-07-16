// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/io/GaussianPlyIO.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace {

using fvdb::detail::io::PlyMetadataTypes;

class TemporaryPlyFile {
  public:
    TemporaryPlyFile() {
        static std::atomic<uint64_t> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        mPath =
            std::filesystem::temp_directory_path() / ("fvdb_gaussian_ply_" + std::to_string(stamp) +
                                                      "_" + std::to_string(counter++) + ".ply");
    }

    ~TemporaryPlyFile() {
        std::error_code error;
        std::filesystem::remove(mPath, error);
    }

    const std::string
    string() const {
        return mPath.string();
    }

  private:
    std::filesystem::path mPath;
};

struct GaussianData {
    torch::Tensor means;
    torch::Tensor quats;
    torch::Tensor logScales;
    torch::Tensor logitOpacities;
    torch::Tensor sh0;
    torch::Tensor shN;
};

GaussianData
makeGaussianData(const int64_t numHigherOrderBases = 3) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    return {
        torch::tensor({{0.1f, -0.2f, 2.0f}, {0.5f, 0.25f, 3.0f}}, options),
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}, {0.9f, 0.1f, 0.2f, 0.3f}}, options),
        torch::tensor({{-0.5f, -0.6f, -0.7f}, {-0.2f, -0.3f, -0.4f}}, options),
        torch::tensor({0.4f, -0.8f}, options),
        torch::tensor({{{0.1f, 0.2f, 0.3f}}, {{0.4f, 0.5f, 0.6f}}}, options),
        torch::arange(2 * numHigherOrderBases * 3, options).reshape({2, numHigherOrderBases, 3}) /
            10.0f,
    };
}

void
expectGaussianDataEqual(
    const GaussianData &expected,
    const std::tuple<torch::Tensor,
                     torch::Tensor,
                     torch::Tensor,
                     torch::Tensor,
                     torch::Tensor,
                     torch::Tensor,
                     std::unordered_map<std::string, PlyMetadataTypes>> &loaded) {
    EXPECT_TRUE(torch::allclose(std::get<0>(loaded), expected.means));
    EXPECT_TRUE(torch::allclose(std::get<1>(loaded), expected.quats));
    EXPECT_TRUE(torch::allclose(std::get<2>(loaded), expected.logScales));
    EXPECT_TRUE(torch::allclose(std::get<3>(loaded), expected.logitOpacities));
    EXPECT_TRUE(torch::allclose(std::get<4>(loaded), expected.sh0));
    EXPECT_TRUE(torch::allclose(std::get<5>(loaded), expected.shN));
}

void
save(const TemporaryPlyFile &file,
     const GaussianData &data,
     std::optional<std::unordered_map<std::string, PlyMetadataTypes>> metadata = std::nullopt) {
    fvdb::detail::io::saveGaussianPly(file.string(),
                                      data.means,
                                      data.quats,
                                      data.logScales,
                                      data.logitOpacities,
                                      data.sh0,
                                      data.shN,
                                      std::move(metadata));
}

} // namespace

TEST(GaussianPlyIOTest, StandardAndEmptyShNRoundTrips) {
    for (const int64_t numHigherOrderBases: {3, 0}) {
        const GaussianData expected = makeGaussianData(numHigherOrderBases);
        TemporaryPlyFile file;

        save(file, expected);
        const auto loaded =
            fvdb::detail::io::loadGaussianPly(file.string(), torch::Device(torch::kCUDA));

        expectGaussianDataEqual(expected, loaded);
        EXPECT_TRUE(std::get<6>(loaded).empty());
        EXPECT_TRUE(std::get<5>(loaded).sizes() == expected.shN.sizes());
    }
}

TEST(GaussianPlyIOTest, FiltersGaussiansContainingNan) {
    GaussianData input = makeGaussianData();
    input.means.index_put_({0, 0}, std::numeric_limits<float>::quiet_NaN());
    TemporaryPlyFile file;

    save(file, input);
    const auto loaded =
        fvdb::detail::io::loadGaussianPly(file.string(), torch::Device(torch::kCUDA));

    EXPECT_EQ(std::get<0>(loaded).size(0), 1);
    EXPECT_TRUE(
        torch::allclose(std::get<0>(loaded), input.means.index({torch::indexing::Slice(1)})));
    EXPECT_TRUE(
        torch::allclose(std::get<1>(loaded), input.quats.index({torch::indexing::Slice(1)})));
    EXPECT_TRUE(torch::allclose(std::get<5>(loaded), input.shN.index({torch::indexing::Slice(1)})));
}

TEST(GaussianPlyIOTest, MetadataTypesAndNonContiguousTensorsRoundTrip) {
    const GaussianData input = makeGaussianData();
    TemporaryPlyFile file;
    const auto tensor = torch::arange(24, torch::TensorOptions().dtype(torch::kFloat32))
                            .reshape({4, 6})
                            .transpose(0, 1);
    ASSERT_FALSE(tensor.is_contiguous());
    std::unordered_map<std::string, PlyMetadataTypes> metadata{
        {"string_value", std::string("fvdb")},
        {"integer_value", int64_t{8198767135}},
        {"double_value", 0.12124324352352465},
        {"tensor_value", tensor},
    };

    save(file, input, metadata);
    const auto loaded          = fvdb::detail::io::loadGaussianPly(file.string());
    const auto &loadedMetadata = std::get<6>(loaded);

    EXPECT_EQ(std::get<std::string>(loadedMetadata.at("string_value")), "fvdb");
    EXPECT_EQ(std::get<int64_t>(loadedMetadata.at("integer_value")), int64_t{8198767135});
    EXPECT_DOUBLE_EQ(std::get<double>(loadedMetadata.at("double_value")), 0.12124324352352465);
    EXPECT_TRUE(torch::equal(std::get<torch::Tensor>(loadedMetadata.at("tensor_value")), tensor));
}

TEST(GaussianPlyIOTest, RejectsInvalidMetadataKeys) {
    const GaussianData input = makeGaussianData();

    for (const char *key: {"invalid key", "invalid@key"}) {
        TemporaryPlyFile file;
        std::unordered_map<std::string, PlyMetadataTypes> metadata{{key, int64_t{1}}};
        EXPECT_THROW(save(file, input, metadata), c10::ValueError);
    }
}
