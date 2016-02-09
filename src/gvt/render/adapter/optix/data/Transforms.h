/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
/*
 * File:   gvt_optix.h
 * Author: jbarbosa
 *
 * Created on April 22, 2014, 12:47 PM
 */

#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H

#include <gvt/core/data/Transform.h>

#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/optix/data/Formats.h>
#include <gvt/render/data/Primitives.h>

#include <vector>

namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace data {

GVT_TRANSFORM_TEMPLATE // see gvt/core/data/Transform.h

    // clang-format off
    /// transform GraviT-compliant rays to OptiX-compliant rays using CUDA
    std::vector<OptixRay>
convertRaysToOptix(const gvt::render::actor::RayVector &rays);
// clang-format on

/// transform GraviT-compliant rays to OptiX-compliant rays using CUDA
template <> struct transform_impl<gvt::render::actor::RayVector, std::vector<OptixRay> > {
  inline static std::vector<OptixRay> transform(const gvt::render::actor::RayVector &rays) {
    return convertRaysToOptix(rays);
  }
};
}
}
}
}
}

#endif /* GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H */
