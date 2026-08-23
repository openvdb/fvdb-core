// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_BUILDERRESOURCE_H
#define FVDB_BUILDERRESOURCE_H

#include <fvdb/TorchResource.h>

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
///        The alias covers the builders' scratch and the device staging
///        buffers feeding them (nanovdb::cuda::Buffer<..., BuilderResource> in
///        SaveNanoVDB and ReinitializeSdf). Grid buffers that are torch-device
///        aware by design (TorchDeviceBuffer) name their allocator directly.
///
///        Note the seam is compile-time and relies on the resource being
///        stateless: builders bind the shared instance from
///        nanovdb::cuda::default_resource<BuilderResource>() through their
///        defaulted constructor arguments. A stateful resource (e.g. one
///        holding a per-session allocator handle) additionally needs an
///        instance plumbed through the ops' call sites.
using BuilderResource = TorchResource;

} // namespace fvdb

#endif // FVDB_BUILDERRESOURCE_H
