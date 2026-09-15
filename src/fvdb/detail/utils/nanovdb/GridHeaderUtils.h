// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_NANOVDB_GRIDHEADERUTILS_H
#define FVDB_DETAIL_UTILS_NANOVDB_GRIDHEADERUTILS_H

#include <nanovdb/NanoVDB.h>

#include <cstdint>

namespace fvdb::detail {

/// @brief Rewrites the header of a grid that was copied out of a batch (or built by hand) so that
///        it describes exactly the buffer it now sits in: grid 0 of 1, @p gridSize bytes, no blind
///        data, checksum disabled. nanovdb::GridHandle validates all of this at construction, so
///        every site that wraps such bytes must make these writes; this is the one place they are
///        spelled out.
///
///        The blind-metadata offset is set to @p gridSize, NanoVDB's convention for "no blind data"
///        (GridData::init). The checksum is disabled rather than recomputed: every field written
///        here is covered by it, and the sites that build such headers go on to mutate more
///        (grid class, type, name) before the buffer is final.
inline void
normalizeStandaloneGridHeader(nanovdb::GridData *data, uint64_t gridSize) {
    data->mGridIndex           = 0u;
    data->mGridCount           = 1u;
    data->mGridSize            = gridSize;
    data->mBlindMetadataCount  = 0u;
    data->mBlindMetadataOffset = gridSize;
    data->mChecksum.disable();
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_NANOVDB_GRIDHEADERUTILS_H
