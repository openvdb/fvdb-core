// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include "GaussianSplat3dView.h"

#include "Viewer.h"

#include <nanovdb_editor/putil/Reflect.h>

namespace fvdb::detail::viewer {

GaussianSplat3dView::GaussianSplat3dView(const std::string &name, const Viewer &viewer)
    : mName(name) {
    mParams.eps2d                        = 0.3f;
    mParams.min_radius_2d                = 0.f;
    mParams.tile_size                    = 16u;
    mParams.sh_degree_override           = -1;
    mParams.sh_stride_rgbrgbrgb_override = 0u;
    mParams.name                         = viewer.getToken(name.c_str())->str;
    mParams.data_type = PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_raster_shader_params_t);
    mSceneToken       = nullptr;
}

} // namespace fvdb::detail::viewer
