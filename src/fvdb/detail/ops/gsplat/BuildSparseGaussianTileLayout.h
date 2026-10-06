// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_BUILDSPARSEGAUSSIANTILELAYOUT_H
#define FVDB_DETAIL_OPS_GSPLAT_BUILDSPARSEGAUSSIANTILELAYOUT_H

#include <fvdb/JaggedTensor.h>

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace fvdb::detail::ops {

/// @brief Computes helper data structures for sparse gaussian splatting
///
/// @details This function computes four Tensors needed for sparse gaussian splatting:
///   For the following, let AP be the number of active pixels and AT be the number of active tiles.
///   1. active_tiles: An integer Tensor with shape [AT] indicating the tile_ids
///      corresponding to tiles (across all cameras) which contain active pixels.
///   2. active_tile_mask: A Boolean Tensor of shape [C, TW, TH] indicating which tiles are active,
///      where C is the number of cameras, TW is the number of tiles along the width dimension, and
///      TH is the number of tiles along the height dimension.
///   3. tile_pixel_mask: A uint64 Tensor of bitmasks with shape [AT, words_per_tile] where
///      words_per_tile is the number of uint64_t words needed to make a bitmask for a
///      tileSideLength x tileSideLength tile. Bits are stored in raster order (top left to bottom
///      right).
///   4. tile_pixel_cumsum: An int64 Tensor with shape [AT] encoding the cumulative sum of
///      active pixels in each active tile.
///   5. pixel_map: An int64 Tensor with shape [AP] specifying the write order of pixels within
///      tiles.
///        i.e. Suppose we're rendering the k^th active pixel in tile_id = active_tiles[t],
///             we write its rendered value to index pixel_map[tile_pixel_cumsum[tile_id-1] + k]
///             in the output
///
/// @param tileSideLength Number of pixels along one side of a square tile
/// @param numTilesW Number of tiles along width dimension
/// @param numTilesH Number of tiles along height dimension
/// @param pixelsToRender JaggedTensor with one [P_c, 2] list of (row, col) coordinates per camera
/// @param imageWidth Exact image width in pixels. Defaults to numTilesW*tileSideLength.
/// @param imageHeight Exact image height in pixels. Defaults to numTilesH*tileSideLength.
///
/// @throws c10::ValueError if pixelsToRender is not a list of tensors (ldim != 1), if the image
///         size does not fit the tile grid, if any pixel lies outside the imageHeight by imageWidth
///         image, or if a pixel appears more than once in the same image.
/// @return Tuple of (active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
buildSparseGaussianTileLayout(const int32_t tileSideLength,
                              const int32_t numTilesW,
                              const int32_t numTilesH,
                              const fvdb::JaggedTensor &pixelsToRender,
                              const std::optional<int32_t> imageWidth  = std::nullopt,
                              const std::optional<int32_t> imageHeight = std::nullopt);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_BUILDSPARSEGAUSSIANTILELAYOUT_H
