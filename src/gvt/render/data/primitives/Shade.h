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

#ifndef GVT_RENDER_DATA_PRIMITIVES_SHADE_H
#define GVT_RENDER_DATA_PRIMITIVES_SHADE_H

#include <gvt/core/Math.h>

namespace gvt {
namespace render {
namespace actor {
class Ray;
}
}
}

namespace gvt {
namespace render {
namespace data {
namespace scene {
class Light;
}
}
}
}

namespace gvt {
namespace render {
namespace data {
namespace primitives {

struct Material;

/*
 * Material proxy call implemented per adpater
 * Interfaces to the different shading materials may be significantly different
 * mainly due to light assessing and vec formats
 */
bool Shade(gvt::render::data::primitives::Material *material, const gvt::render::actor::Ray &ray,
           const glm::vec3 &sufaceNormal, const gvt::render::data::scene::Light *lightSource,
           const glm::vec3 lightPosSample, glm::vec3 &color);
}
}
}
}

#endif
