// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_GAUSSIANPLYIO_H
#define FVDB_DETAIL_IO_GAUSSIANPLYIO_H

#include <c10/core/Device.h>
#include <torch/types.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace fvdb::detail::io {

/// The types of valid metadata you can save in a PLY file alongside Gaussians
using PlyMetadataTypes = std::variant<std::string, int64_t, double, torch::Tensor>;

/// Magic string prepended to additional metadata properties stored in PLY files
inline static const std::string PLY_MAGIC = "fvdb_ply_af_8198767135";

/// We won't allow keys in a PLY file longer than this many characters.
inline static const size_t MAX_PLY_KEY_LENGTH = 256;

inline static const std::string PLY_VERSION_STRING = "fvdb_ply 1.0.0";

/// @brief Load a PLY file's means, quats, scales, opacities, and SH coefficients.
/// @param filename Filename of the PLY file
/// @param device Device to transfer the loaded tensors to
/// @return A tuple of (means, quats, logScales, logitOpacities, sh0, shN, metadata).
///  The metadata dictionary can be empty if no metadata was saved in the PLY file.
///  Its keys are strings and values are either strings, int64s, doubles, or tensors.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::unordered_map<std::string, PlyMetadataTypes>>
loadGaussianPly(const std::string &filename, torch::Device device = torch::kCPU);

/// @brief Save Gaussian tensors and optional training metadata to a PLY file.
/// @param filename The path to save the PLY file to
/// @param means Gaussian means tensor of shape (N, 3)
/// @param quats Gaussian quaternions tensor of shape (N, 4)
/// @param logScales Gaussian log-scales tensor of shape (N, 3)
/// @param logitOpacities Gaussian logit-opacities tensor of shape (N,)
/// @param sh0 Zeroth-order SH coefficients tensor of shape (N, 1, D)
/// @param shN Higher-order SH coefficients tensor of shape (N, K-1, D)
/// @param metadata An optional dictionary of training metadata to include in the PLY file. The
/// keys are strings and the values are either strings, int64s, doubles, or tensors
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
