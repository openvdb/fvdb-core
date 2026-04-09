// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_GAUSSIANPLYIO_H
#define FVDB_DETAIL_IO_GAUSSIANPLYIO_H

#include <torch/types.h>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace fvdb::detail::io {

/// Type alias for PLY metadata values.
using PlyMetadataTypes = std::variant<std::string, int64_t, double, torch::Tensor>;

/// Magic string prepended to additional metadata properties stored in PLY files
inline static const std::string PLY_MAGIC = "fvdb_ply_af_8198767135";

/// We won't allow keys in a PLY file longer than this many characters.
inline static const size_t MAX_PLY_KEY_LENGTH = 256;

inline static const std::string PLY_VERSION_STRING = "fvdb_ply 1.0.0";

/// @brief Load a PLY file's means, quats, scales, opacities, and SH coefficients as raw tensors.
/// @param filename Filename of the PLY file
/// @param device Device to transfer the loaded tensors to
/// @return A tuple of (means, quats, logScales, logitOpacities, sh0, shN, metadata)
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::unordered_map<std::string, PlyMetadataTypes>>
loadGaussianPly(const std::string &filename, torch::Device device = torch::kCPU);

/// @brief Save Gaussian splat data and optional training metadata to a PLY file.
/// @param filename The path to save the PLY file to
/// @param means [N, 3] Gaussian means
/// @param quats [N, 4] Gaussian quaternions
/// @param logScales [N, 3] Log of Gaussian scales
/// @param logitOpacities [N] Logit of Gaussian opacities
/// @param sh0 [N, 1, D] Degree-0 SH coefficients
/// @param shN [N, K-1, D] Higher-degree SH coefficients
/// @param metadata An optional dictionary of training metadata to include in the PLY file
void saveGaussianPly(
    const std::string &filename,
    const torch::Tensor &means,
    const torch::Tensor &quats,
    const torch::Tensor &logScales,
    const torch::Tensor &logitOpacities,
    const torch::Tensor &sh0,
    const torch::Tensor &shN,
    std::optional<std::unordered_map<std::string, PlyMetadataTypes>> metadata = std::nullopt);

} // namespace fvdb::detail::io

#endif // FVDB_DETAIL_IO_GAUSSIANPLYIO_H
