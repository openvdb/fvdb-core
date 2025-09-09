// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// @file Nvtx.h
// @brief NVTX (NVIDIA Tools Extension) utilities for fVDB profiling
//
// This header provides NVTX range utilities for profiling fVDB operations.
//
// Usage Examples:
// ```cpp
// // Use default color
// FVDB_FUNC_RANGE();
//
// // Use custom color
// FVDB_FUNC_RANGE_WITH_COLOR(nvtx3::rgb{255, 0, 0}); // Red color
//
// // For class methods, this will show just the function name (e.g., "forward")
// // To include class names, you can use a custom string:
// FVDB_FUNC_RANGE_WITH_COLOR_AND_NAME(nvtx3::rgb{255, 0, 0}, "MyClass::forward");
// ```
//

#ifndef FVDB_DETAIL_UTILS_NVTX_H
#define FVDB_DETAIL_UTILS_NVTX_H

#include <nvtx3/nvtx3.hpp>

#include <cstring>

namespace fvdb {

/// Tag type for libfvdb's NVTX domain.
struct fvdb_domain {
    static constexpr char const *name{"libfvdb"};           ///< Name of the libfvdb domain
    static constexpr auto color = nvtx3::rgb{255, 112, 67}; ///< Default color for libfvdb ranges
};

/// Alias for an NVTX range in the libfvdb domain.
/// Customizes an NVTX range with the given input.
///
/// Example:
/// ```cpp
/// void some_function(){
///    fvdb::scoped_range rng{"custom_name"}; // Customizes range name
///    ...
/// }
/// ```
using scoped_range = ::nvtx3::scoped_range_in<fvdb_domain>;

} // namespace fvdb

/// Convenience macro for generating an NVTX range with a custom name and color.
///
/// Uses a custom name instead of the function name, useful for including class names.
///
/// Example:
/// ```cpp
/// class MyClass {
///     void forward() {
///         FVDB_FUNC_RANGE_WITH_COLOR_AND_NAME(nvtx3::rgb{255, 0, 0}, "MyClass::forward");
///         // ...
///     }
/// };
/// ```
#define FVDB_FUNC_RANGE_WITH_COLOR_AND_NAME(color, name)                                   \
    static ::nvtx3::registered_string_in<fvdb::fvdb_domain> const nvtx3_func_name__{name}; \
    static ::nvtx3::event_attributes const nvtx3_func_attr__{nvtx3_func_name__, color};    \
    ::nvtx3::scoped_range_in<fvdb::fvdb_domain> const nvtx3_range__{nvtx3_func_attr__};

/// Convenience macro for generating a colored NVTX range in the `libfvdb` domain
/// from the lifetime of a function.
///
/// Uses the name of the immediately enclosing function returned by `__func__` to
/// name the range and applies the specified color. This is the base implementation
/// that FVDB_FUNC_RANGE() uses internally.
///
/// Example:
/// ```cpp
/// void some_function(){
///    FVDB_FUNC_RANGE_WITH_COLOR(nvtx3::rgb{255, 0, 0}); // Red color
///    ...
/// }
/// ```
#define FVDB_FUNC_RANGE_WITH_COLOR(color) FVDB_FUNC_RANGE_WITH_COLOR_AND_NAME(color, __func__)

/// Convenience macro for generating an NVTX range in the `libfvdb` domain
/// from the lifetime of a function.
///
/// Uses the name of the immediately enclosing function returned by `__func__` to
/// name the range and applies the default libfvdb color.
///
/// Example:
/// ```cpp
/// void some_function(){
///    FVDB_FUNC_RANGE(); // Uses default color
///    ...
/// }
/// ```
#define FVDB_FUNC_RANGE() FVDB_FUNC_RANGE_WITH_COLOR(fvdb::fvdb_domain::color)

/// Convenience macro for generating an NVTX range with a custom name.
///
/// Uses a custom name instead of the function name, useful for including class names.
///
/// Example:
/// ```cpp
/// class MyClass {
///     void forward() {
///         FVDB_FUNC_RANGE_CUSTOM("MyClass::forward");
///         // ...
///     }
/// };
/// ```
#define FVDB_FUNC_RANGE_WITH_NAME(name) \
    FVDB_FUNC_RANGE_WITH_COLOR_AND_NAME(fvdb::fvdb_domain::color, name)

#endif // FVDB_DETAIL_UTILS_NVTX_H
