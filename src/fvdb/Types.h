// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_TYPES_H
#define FVDB_TYPES_H

namespace fvdb {

/// @brief Enum class for space-filling curve encoding types
enum class SpaceFillingCurveType {
    ZOrder,           ///< Regular z-order curve (xyz)
    ZOrderTransposed, ///< Transposed z-order curve (zyx)
    Hilbert,          ///< Regular Hilbert curve (xyz)
    HilbertTransposed ///< Transposed Hilbert curve (zyx)
};

} // namespace fvdb

#endif // FVDB_TYPES_H
