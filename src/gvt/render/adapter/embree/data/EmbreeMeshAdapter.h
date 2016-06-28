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

#ifndef GVT_RENDER_ADAPTER_EMBREE_DATA_EMBREE_MESH_ADAPTER_H
#define GVT_RENDER_ADAPTER_EMBREE_DATA_EMBREE_MESH_ADAPTER_H

#include "gvt/render/Adapter.h"

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

#include <string>

namespace gvt {
namespace render {
namespace adapter {
namespace embree {
namespace data {
/// mesh adapter for Intel Embree ray tracer
/** this helper class transforms mesh data from the GraviT internal format
to the format expected by Intel's Embree ray tracer
*/
class EmbreeMeshAdapter : public gvt::render::Adapter {
public:
  /**
   * Construct the Embree mesh adapter.  Convert the mesh
   * at the given node to Embree's format.
   *
   * Initializes Embree the first time it is called.
   */
  EmbreeMeshAdapter(gvt::render::data::primitives::Mesh *mesh);

  /**
   * Release Embree copy of the mesh.
   */
  virtual ~EmbreeMeshAdapter();

  /**
   * Return the Embree scene handle;
   */
  RTCScene getScene() const { return scene; }

  /**
   * Return the geometry id.
   */
  unsigned getGeomId() const { return geomId; }

  /**
   * Return the packet size
   */
  RTCAlgorithmFlags getPacketSize() const { return packetSize; }

  /**
   * Trace rays using the Embree adapter.
   *
   * Creates threads and traces rays in packets defined by GVT_EMBREE_PACKET_SIZE
   * (currently set to 4).
   *
   * \param rayList incoming rays
   * \param moved_rays outgoing rays [rays that did not hit anything]
   * \param instNode instance db node containing dataRef and transforms
   */
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays, glm::mat4 *m,
                     glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
                     size_t begin = 0, size_t end = 0);

  /**
   * Handle to Embree scene.
   */
  RTCScene global_scene;

protected:
  /**
   * Static bool to initialize Embree (calling rtcInit) before use.
   *
   * // TODO: this will need to move in the future when we have different types of Embree adapters (ex: mesh + volume)
   */
  static bool init;

  RTCDevice device;

  /**
   * Currently selected packet size flag.
   */
  RTCAlgorithmFlags packetSize;

  /**
   * Handle to Embree scene.
   */
  RTCScene scene;

  /**
   * Handle to the Embree triangle mesh.
   */
  unsigned geomId;
  unsigned instID;

  size_t begin, end;
};
}
}
}
}
}

#endif // GVT_RENDER_ADAPTER_EMBREE_DATA_EMBREE_MESH_ADAPTER_H
