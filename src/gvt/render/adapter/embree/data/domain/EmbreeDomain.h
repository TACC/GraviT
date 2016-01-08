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
//
// EmbreeDomain.h
//

#ifndef GVT_RENDER_ADAPTER_EMBREE_DATA_DOMAIN_EMBREE_DOMAIN_H
#define GVT_RENDER_ADAPTER_EMBREE_DATA_DOMAIN_EMBREE_DOMAIN_H

#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/domain/GeometryDomain.h>
#include <gvt/render/data/Primitives.h>

// Embree includes
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
// end Embree includes

#include <string>

namespace gvt {
namespace render {
namespace adapter {
namespace embree {
namespace data {
namespace domain {
/// data adapter for Intel Embree ray tracer
/** this helper class transforms geometry data from the GraviT internal format
to the format expected by Intel's Embree ray tracer
*/
class EmbreeDomain : public gvt::render::data::domain::GeometryDomain {
public:
  EmbreeDomain(gvt::render::data::domain::GeometryDomain *domain);
  EmbreeDomain(std::string filename = "",
               gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
  EmbreeDomain(const EmbreeDomain &other);
  virtual ~EmbreeDomain();

  virtual bool load();
  virtual void free();

#if 0
                            Manta::RenderContext*   getRenderContext() { return rContext; }
                            Manta::DynBVH*          getAccelStruct() { return as; }
                            Manta::Mesh*            getMantaMesh() { return meshManta; }
#endif
  RTCScene getScene() { return scene; }
  unsigned getGeomId() { return geomId; }

  void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays);

protected:
  static bool init;

  RTCAlgorithmFlags packetSize;
  RTCScene scene;
  unsigned geomId;
#if 0
                            Manta::RenderContext* rContext;
                            Manta::DynBVH* as;
                            Manta::Mesh* meshManta;
#endif
};
}
}
}
}
}
}

#endif // GVT_RENDER_ADAPTER_EMBREE_DATA_DOMAIN_EMBREE_DOMAIN_H
