// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_DISPATCH_TRAITS_H
#define FVDB_DETAIL_DISPATCH_TRAITS_H

#include <array>
#include <cstddef>
#include <utility>

namespace fvdb {
namespace dispatch {

template <typename T> struct strides_helper;

template <size_t... Dims> struct strides_helper<std::index_sequence<Dims...>> {
    static_assert(sizeof...(Dims) > 0, "strides only defined for rank > 0");
    static_assert(((Dims > 0) && ... && true), "dimensions must be greater than 0");

  private:
    static constexpr size_t N                   = sizeof...(Dims);
    static constexpr std::array<size_t, N> dims = {Dims...};

    static constexpr size_t
    suffix_product(size_t start) {
        size_t result = 1;
        for (size_t i = start; i < N; ++i) {
            result *= dims[i];
        }
        return result;
    }

    template <size_t... Is>
    static auto
        make_type(std::index_sequence<Is...>) -> std::index_sequence<suffix_product(Is + 1)...>;

  public:
    using type = decltype(make_type(std::make_index_sequence<N>{}));
};

template <typename T> using strides_helper_t = typename strides_helper<T>::type;

template <typename T> struct array_from_indices;

template <size_t... Is> struct array_from_indices<std::index_sequence<Is...>> {
    static constexpr std::array<size_t, sizeof...(Is)> value = {Is...};
};

} // namespace dispatch
} // namespace fvdb

#endif // FVDB_DETAIL_DISPATCH_TRAITS_H
