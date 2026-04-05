// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANDEDUPLICATEPIXELS_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANDEDUPLICATEPIXELS_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Deduplicate pixel coordinates in a JaggedTensor.
///
/// Encodes each pixel as a single int64 key incorporating its batch index and 2D coordinate,
/// sorts keys to find unique groups and builds an inverse mapping. Returns the deduplicated
/// pixels as a new JaggedTensor, the inverse index tensor, and a flag indicating whether any
/// duplicates were found.
///
/// @param pixelsToRender The input JaggedTensor of pixel coordinates [total_pixels, 2]
/// @param imageWidth Width of each image in pixels
/// @param imageHeight Height of each image in pixels
/// @return Tuple of (uniquePixels JaggedTensor, inverseIndices tensor, hasDuplicates bool)
std::tuple<fvdb::JaggedTensor, torch::Tensor, bool> deduplicatePixels(
    const fvdb::JaggedTensor &pixelsToRender, int64_t imageWidth, int64_t imageHeight);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANDEDUPLICATEPIXELS_H
