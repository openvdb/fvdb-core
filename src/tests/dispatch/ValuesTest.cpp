// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/dispatch/Values.h>

#include <gtest/gtest.h>

#include <cstddef>

namespace fvdb {
namespace dispatch {

// =============================================================================
// AnyTypeValuePack Tests
// =============================================================================

TEST(AnyTypeValuePack, EmptyPackHasSizeZero) {
    using EmptyPack = AnyTypeValuePack<>;
    constexpr auto size = PackSize<EmptyPack>::value();
    EXPECT_EQ(size, 0);
}


} // namespace dispatch
} // namespace fvdb
