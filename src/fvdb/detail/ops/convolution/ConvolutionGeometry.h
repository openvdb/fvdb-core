// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file ConvolutionGeometry.h
/// @brief Canonical index-space geometry for sparse convolution.

#ifndef FVDB_DETAIL_OPS_CONVOLUTION_CONVOLUTIONGEOMETRY_H
#define FVDB_DETAIL_OPS_CONVOLUTION_CONVOLUTIONGEOMETRY_H

#include <nanovdb/NanoVDB.h>

#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Immutable geometry for the canonical sparse-convolution relation.
///
/// Dilation and lattice registration are fixed to one and zero, respectively.
/// The relation represented here is therefore
/// @code
/// fine = stride * coarse + tap - paddingBefore
/// @endcode
/// componentwise. Keep all index-space users on this value instead of
/// re-deriving an even-kernel phase locally.
class ConvolutionGeometry {
  public:
    static constexpr int64_t kSemanticsVersion = 1;

    ConvolutionGeometry(nanovdb::Coord kernelSize, nanovdb::Coord stride)
        : mKernelSize(validatedKernelSize(kernelSize)), mStride(validatedStride(stride)),
          mPaddingBefore(paddingBeforeFor(mKernelSize)),
          mPaddingAfter(paddingAfterFor(mKernelSize)),
          mKernelVolume(checkedKernelVolume(mKernelSize)) {}

    ConvolutionGeometry(ConvolutionGeometry const &)            = default;
    ConvolutionGeometry &operator=(ConvolutionGeometry const &) = delete;

    [[nodiscard]] __hostdev__ nanovdb::Coord const &
    kernelSize() const {
        return mKernelSize;
    }

    [[nodiscard]] __hostdev__ nanovdb::Coord const &
    stride() const {
        return mStride;
    }

    [[nodiscard]] __hostdev__ nanovdb::Coord const &
    paddingBefore() const {
        return mPaddingBefore;
    }

    [[nodiscard]] __hostdev__ nanovdb::Coord const &
    paddingAfter() const {
        return mPaddingAfter;
    }

    [[nodiscard]] __hostdev__ int64_t
    kernelVolume() const {
        return mKernelVolume;
    }

    [[nodiscard]] __hostdev__ static constexpr int64_t
    semanticsVersion() {
        return kSemanticsVersion;
    }

    [[nodiscard]] __hostdev__ static nanovdb::Coord
    dilation() {
        return nanovdb::Coord(1);
    }

    [[nodiscard]] __hostdev__ static nanovdb::Coord
    registrationOffset() {
        return nanovdb::Coord(0);
    }

    /// @brief Convert a linear tap index to its zero-based three-dimensional coordinate.
    [[nodiscard]] __hostdev__ nanovdb::Coord
    tapCoord(int64_t tapIndex) const {
        const int64_t yz = static_cast<int64_t>(mKernelSize[1]) * mKernelSize[2];
        return nanovdb::Coord(static_cast<int32_t>(tapIndex / yz),
                              static_cast<int32_t>((tapIndex / mKernelSize[2]) % mKernelSize[1]),
                              static_cast<int32_t>(tapIndex % mKernelSize[2]));
    }

    /// @brief Return the canonical fine-lattice offset of a zero-based tap coordinate.
    [[nodiscard]] __hostdev__ nanovdb::Coord
    tapOffset(nanovdb::Coord const &tap) const {
        return tap - mPaddingBefore;
    }

    /// @brief Return the fine coordinate connected to @p coarse through @p tap.
    [[nodiscard]] __hostdev__ nanovdb::Coord
    fineFromCoarse(nanovdb::Coord const &coarse, nanovdb::Coord const &tap) const {
        return nanovdb::Coord(coarse[0] * mStride[0] + tap[0] - mPaddingBefore[0],
                              coarse[1] * mStride[1] + tap[1] - mPaddingBefore[1],
                              coarse[2] * mStride[2] + tap[2] - mPaddingBefore[2]);
    }

    /// @brief Solve the canonical relation for a coarse coordinate when divisible.
    [[nodiscard]] __hostdev__ bool
    coarseFromFine(nanovdb::Coord const &fine,
                   nanovdb::Coord const &tap,
                   nanovdb::Coord &coarse) const {
        const nanovdb::Coord offset = tapOffset(tap);
        const int64_t x             = static_cast<int64_t>(fine[0]) - offset[0];
        const int64_t y             = static_cast<int64_t>(fine[1]) - offset[1];
        const int64_t z             = static_cast<int64_t>(fine[2]) - offset[2];
        if (!isDivisible(x, mStride[0]) || !isDivisible(y, mStride[1]) ||
            !isDivisible(z, mStride[2])) {
            return false;
        }
        coarse = nanovdb::Coord(static_cast<int32_t>(floorDiv(x, mStride[0])),
                                static_cast<int32_t>(floorDiv(y, mStride[1])),
                                static_cast<int32_t>(floorDiv(z, mStride[2])));
        return true;
    }

    /// @brief Euclidean floor division for a positive divisor.
    [[nodiscard]] __hostdev__ static int64_t
    floorDiv(int64_t dividend, int32_t divisor) {
        int64_t quotient  = dividend / divisor;
        int64_t remainder = dividend % divisor;
        if (remainder < 0) {
            quotient -= 1;
        }
        return quotient;
    }

    /// @brief Euclidean modulo in [0, divisor) for a positive divisor.
    [[nodiscard]] __hostdev__ static int64_t
    floorMod(int64_t dividend, int32_t divisor) {
        int64_t remainder = dividend % divisor;
        return remainder < 0 ? remainder + divisor : remainder;
    }

    [[nodiscard]] __hostdev__ static bool
    isDivisible(int64_t dividend, int32_t divisor) {
        return floorMod(dividend, divisor) == 0;
    }

  private:
    [[nodiscard]] static nanovdb::Coord
    validatedKernelSize(nanovdb::Coord const &kernelSize) {
        for (int d = 0; d < 3; ++d) {
            TORCH_CHECK_VALUE(kernelSize[d] > 0,
                              "kernel_size must be strictly positive, got ",
                              kernelSize[d],
                              " in dimension ",
                              d);
        }
        return kernelSize;
    }

    [[nodiscard]] static nanovdb::Coord
    validatedStride(nanovdb::Coord const &stride) {
        for (int d = 0; d < 3; ++d) {
            TORCH_CHECK_VALUE(stride[d] > 0,
                              "stride must be strictly positive, got ",
                              stride[d],
                              " in dimension ",
                              d);
        }
        return stride;
    }

    [[nodiscard]] static nanovdb::Coord
    paddingBeforeFor(nanovdb::Coord const &kernelSize) {
        return nanovdb::Coord(
            (kernelSize[0] - 1) / 2, (kernelSize[1] - 1) / 2, (kernelSize[2] - 1) / 2);
    }

    [[nodiscard]] static nanovdb::Coord
    paddingAfterFor(nanovdb::Coord const &kernelSize) {
        const nanovdb::Coord before = paddingBeforeFor(kernelSize);
        return kernelSize - nanovdb::Coord(1) - before;
    }

    [[nodiscard]] static int64_t
    checkedKernelVolume(nanovdb::Coord const &kernelSize) {
        int64_t volume = 1;
        for (int d = 0; d < 3; ++d) {
            TORCH_CHECK_VALUE(volume <= std::numeric_limits<int64_t>::max() / kernelSize[d],
                              "kernel volume overflows int64 for kernel_size [",
                              kernelSize[0],
                              ", ",
                              kernelSize[1],
                              ", ",
                              kernelSize[2],
                              "]");
            volume *= kernelSize[d];
        }
        return volume;
    }

    const nanovdb::Coord mKernelSize;
    const nanovdb::Coord mStride;
    const nanovdb::Coord mPaddingBefore;
    const nanovdb::Coord mPaddingAfter;
    const int64_t mKernelVolume;
};

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_CONVOLUTIONGEOMETRY_H
