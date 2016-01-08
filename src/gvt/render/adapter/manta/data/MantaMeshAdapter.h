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
// MantaMeshAdapter.h
//

#ifndef GVT_RENDER_ADAPTER_MANTA_DATA_MANTA_MESH_ADAPTER_H
#define GVT_RENDER_ADAPTER_MANTA_DATA_MANTA_MESH_ADAPTER_H

#include "gvt/render/Adapter.h"

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

/// mesh adapter for SCI's Manta ray tracer
/** this helper class transforms mesh data from the GraviT internal format
to the format expected by SCI's Manta ray tracer
*/
class MantaMeshAdapter : public gvt::render::Adapter {
public:
  // MantaMeshAdapter(gvt::render::data::domain::GeometryDomain* domain);
  /**
   * Construct the Manta mesh adapter.  Convert the mesh
   * at the given node to Manta's format.
   *
   * Initializes Manta the first time it is called.
   */
  MantaMeshAdapter(gvt::core::DBNodeH node);

  // MantaMeshAdapter(std::string filename ="",gvt::core::math::AffineTransformMatrix<float> m =
  // gvt::core::math::AffineTransformMatrix<float>(true));
  // MantaMeshAdapter(const MantaMeshAdapter& other);

  /**
   * Release Manta copy of the mesh.
   */
  virtual ~MantaMeshAdapter();

  virtual bool load();
  virtual void free();

  Manta::RenderContext *getRenderContext() { return rContext; }

  /**
   * Return the Manta DynBVH acceleration structure.
   */
  Manta::DynBVH *getAccelStruct() { return as; }

  /**
   * Return pointer to the Manta mesh.
   */
  Manta::Mesh *getMantaMesh() { return meshManta; }

  /**
   * Trace rays using the Manta adapter.
   *
   * \param rayList incoming rays
   * \param moved_rays outgoing rays [rays that did not hit anything]
   * \param instNode instance db node containing dataRef and transforms
   */
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                     gvt::core::DBNodeH instNode, size_t _begin = 0, size_t _end = 0);
  // void trace(gvt::render::actor::RayVector& rayList,
  // gvt::render::actor::RayVector& moved_rays);
  //

protected:
  /**
   * Pointer to the Manta render context.
   */
  Manta::RenderContext *rContext;

  /**
   * Pointer to the Manta DynBVH acceleration structrue.
   */
  Manta::DynBVH *as;

  /**
   * Pointer to the Manta mesh.
   */
  Manta::Mesh *meshManta;
};
}
}
}
}
}

#endif // GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H
