// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_GRIDHEADERUTILS_H
#define FVDB_DETAIL_UTILS_NANOVDB_GRIDHEADERUTILS_H

#include <nanovdb/NanoVDB.h>

#include <cstdint>

namespace fvdb::detail {

/// @brief Rewrites the header of a grid that was copied out of a batch (or built by hand) so that
///        it describes exactly the buffer it now sits in: grid 0 of 1, @p gridSize bytes, the
///        blind-metadata layout the caller laid out (none by default), checksum disabled.
///        nanovdb::GridHandle validates the index, count and size at construction, so every site
///        that wraps such bytes must make these writes; this is the one place they are spelled out.
///
///        With no blind data the offset is set to @p gridSize, NanoVDB's convention for "none"
///        (GridData::init), and the HasLongGridName flag is cleared: a name longer than the
///        header's field lives in a blind record, and with the records gone the flag would send
///        gridName() searching zero entries (an assert in debug builds). The header's own
///        truncated name is what remains. The checksum is disabled rather than recomputed: every
///        field written here is covered by it, and the callers go on to mutate more (grid class,
///        type, name) before the buffer is final.
inline void
normalizeStandaloneGridHeader(nanovdb::GridData *data,
                              uint64_t gridSize,
                              uint32_t blindMetadataCount  = 0u,
                              uint64_t blindMetadataOffset = 0u) {
    data->mGridIndex           = 0u;
    data->mGridCount           = 1u;
    data->mGridSize            = gridSize;
    data->mBlindMetadataCount  = blindMetadataCount;
    data->mBlindMetadataOffset = blindMetadataCount == 0u ? gridSize : blindMetadataOffset;
    if (blindMetadataCount == 0u) {
        data->setLongGridNameOn(false);
    }
    data->mChecksum.disable();
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_NANOVDB_GRIDHEADERUTILS_H
