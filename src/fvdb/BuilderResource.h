// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_BUILDERRESOURCE_H
#define FVDB_BUILDERRESOURCE_H

#include <fvdb/TorchResource.h>

#include <nanovdb/cuda/Buffer.h>

namespace fvdb {

/// @brief The memory resource fvdb's ops bind as the ResourceT template
///        parameter of nanoVDB's CUDA builders (and of fvdb's own PadGrid),
///        routing their internal device scratch.
///
///        This alias is the single seam choosing that policy: call sites name
///        BuilderResource, never a concrete resource type. Today it is
///        TorchResource, which allocates from PyTorch's currently active CUDA
///        allocator (see TorchResource.h). A build that must run these
///        builders without torch (e.g. an ONNX Runtime execution provider,
///        where c10 is unavailable) retargets the alias here — behind a
///        build-time switch guarding the TorchResource include — instead of
///        touching every op.
///
///        The alias covers every device-only allocation whose lifetime is an
///        op: the builders' internal scratch, and the staging and scratch
///        buffers fvdb's ops hand them or fill for them, which are declared
///        as BuilderBuffer<T> below. Grid storage that outlives the op
///        (TorchDeviceBuffer) is torch-device aware by design and allocates
///        through TorchResource directly.
///
///        Note the seam is compile-time and relies on the resource being
///        stateless: builders bind the shared instance from
///        nanovdb::cuda::default_resource<BuilderResource>() through their
///        defaulted constructor arguments. A stateful resource (e.g. one
///        holding a per-session allocator handle) additionally needs an
///        instance plumbed through the ops' call sites.
using BuilderResource = TorchResource;

/// @brief A device-only buffer of T over BuilderResource: the type for an op's
///        own staging and scratch allocations. Retargeting BuilderResource
///        retargets these too.
template <class T> using BuilderBuffer = nanovdb::cuda::Buffer<T, BuilderResource>;

} // namespace fvdb

#endif // FVDB_BUILDERRESOURCE_H
