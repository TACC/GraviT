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
// MantaDomain.h
//

#ifndef GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H
#define GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H

#include <gvt/render/adapter/manta/override/DynBVH.h>
#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/domain/GeometryDomain.h>
#include <gvt/render/data/Primitives.h>

// begin Manta includes
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/BBox.h>
#include <Interface/Context.h>
#include <Interface/LightSet.h>
#include <Interface/MantaInterface.h>
#include <Interface/Object.h>
#include <Interface/Scene.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Lights/PointLight.h>
#include <Model/Materials/Phong.h>
#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Model/Readers/PlyReader.h>
// end Manta includes

#include <string>

namespace gvt {
namespace render {
namespace adapter {
namespace manta {
namespace data {
namespace domain {
/// data adapter for SCI Manta ray tracer
/** this helper class transforms geometry data from the GraviT internal format
to the format expected by SCI's Manta ray tracer
*/
class MantaDomain : public gvt::render::data::domain::GeometryDomain {
public:
  MantaDomain(gvt::render::data::domain::GeometryDomain *domain);
  MantaDomain(std::string filename = "",
              gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
  MantaDomain(const MantaDomain &other);
  virtual ~MantaDomain();

  virtual bool load();
  virtual void free();

  Manta::RenderContext *getRenderContext() { return rContext; }
  Manta::DynBVH *getAccelStruct() { return as; }
  Manta::Mesh *getMantaMesh() { return meshManta; }

  void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays);

protected:
  Manta::RenderContext *rContext;
  Manta::DynBVH *as;
  Manta::Mesh *meshManta;
};
}
}
}
}
}
}

#endif // GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H
