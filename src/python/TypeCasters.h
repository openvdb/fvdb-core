// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef PYTHON_TYPECASTERS_H
#define PYTHON_TYPECASTERS_H

#include <fvdb/detail/utils/nanovdb/TorchNanoConversions.h>

#include <torch/extension.h>
#include <torch/version.h>

namespace pybind11 {
namespace detail {

// Already defined in upstream pytorch: https://github.com/pytorch/pytorch/pull/126865
// (starting from version 2.4)
#if (!defined(TORCH_VERSION_MAJOR) || (TORCH_VERSION_MAJOR < 2) || \
     (TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR < 4))

const static inline pybind11::module TORCH_MODULE = py::module_::import("torch");

template <> struct type_caster<torch::ScalarType> : public type_caster_base<torch::ScalarType> {
    using base = type_caster_base<torch::ScalarType>;

  public:
    torch::ScalarType st_value;

    bool
    load(handle src, bool convert) {
        if (THPDtype_Check(src.ptr())) {
            st_value = reinterpret_cast<THPDtype *>(src.ptr())->scalar_type;
            value    = &st_value;
            return true;
        } else {
            return base::load(src, convert);
        }
    }

    static handle
    cast(const at::ScalarType &src, return_value_policy policy, handle parent) {
        auto result = TORCH_MODULE.attr(fvdb::detail::TorchScalarTypeToStr(src).c_str());
        Py_INCREF(result.ptr());
        return result;
    }
};
#endif

} // namespace detail
} // namespace pybind11

#endif // PYTHON_TYPECASTERS_H
