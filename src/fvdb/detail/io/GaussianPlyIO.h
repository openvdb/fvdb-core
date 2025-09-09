// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_IO_GAUSSIANPLYIO_H
#define FVDB_DETAIL_IO_GAUSSIANPLYIO_H

#include <fvdb/GaussianSplat3d.h>

namespace fvdb::detail::io {

/// The types of valid metadata you can save in a PLY file alongside Gaussians
using PlyMetadataTypes = std::variant<std::string, int64_t, double, torch::Tensor>;

/// Magic string prepended to additional metadata properties stored in PLY files
inline static const std::string PLY_MAGIC = "fvdb_ply_af_8198767135";

/// We won't allow keys in a PLY file longer than this many characters.
inline static const size_t MAX_PLY_KEY_LENGTH = 256;

inline static const std::string PLY_VERSION_STRING = "fvdb_ply 1.0.0";

/// @brief Load a PLY file's means, quats, scales, opacities, and SH coefficients as the state
/// of this GaussianSplat3d object
/// @param filename Filename of the PLY file
/// @param device Device to transfer the loaded tensors to
/// @return The loaded GaussianSplat3d class, and a dictionary of metadata (can be empty if no
//  metadata was saved in the PLY file). The metadata keys are strings and the values are either
//  strings, int64s, doubles, or tensors.
std::tuple<GaussianSplat3d, std::unordered_map<std::string, PlyMetadataTypes>>
loadGaussianPly(const std::string &filename, torch::Device device = torch::kCPU);

/// @brief Save this scene and optional training metadata to a PLY file with the given filename
/// @param filename The path to save the PLY file to
/// @param gaussians The GaussianSplat3d object containing the Gaussians to saved.
/// @param metadata An optional dictionary of training metadata to include in the PLY file. The
/// keys are strings and the values are either strings, int64s, doubles, or tensors
void saveGaussianPly(
    const std::string &filename,
    const GaussianSplat3d &gaussians,
    std::optional<std::unordered_map<std::string, PlyMetadataTypes>> metadata = std::nullopt);

} // namespace fvdb::detail::io

#endif // FVDB_DETAIL_IO_GAUSSIANPLYIO_H
