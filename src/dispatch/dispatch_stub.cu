// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// Stub file to make dispatch a compiled CUDA library.
// This enables propagating compile options (like OpenMP) to consumers.
//
// Without this stub, dispatch would be an INTERFACE (header-only) library,
// which cannot propagate language-specific compile options.
