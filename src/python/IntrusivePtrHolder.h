// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch already provides PYBIND11_DECLARE_HOLDER_TYPE(T, c10::intrusive_ptr<T>, true)
// in torch/csrc/utils/pybind.h. This header exists to document this fact and provide
// a single include point for files that need to use c10::intrusive_ptr as a pybind11 holder.
//
#ifndef FVDB_PYTHON_INTRUSIVE_PTR_HOLDER_H
#define FVDB_PYTHON_INTRUSIVE_PTR_HOLDER_H

#include <torch/extension.h>

#endif // FVDB_PYTHON_INTRUSIVE_PTR_HOLDER_H
