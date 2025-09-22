// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "GaussianSplat3dView.h"

#include "Viewer.h"

#include <nanovdb_editor/putil/Reflect.h>

namespace fvdb::detail::viewer {

GaussianSplat3dView::GaussianSplat3dView(const std::string &name,
                                         const GaussianSplat3d &splats,
                                         const Viewer &viewer)
    : mSplats(splats), mViewName(name) {
    mParams.near_plane_override = 0;
    mParams.far_plane_override  = 0;
    mParams.eps2d               = 0.3f;
    mParams.min_radius_2d       = 0.f;
    mParams.tile_size           = 16u;
    mParams.sh_degree_override  = 0;
    mParams.data_type           = PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_raster_shader_params_t);
}

} // namespace fvdb::detail::viewer
