/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
#ifndef GVT_RENDER_ADAPTER_HETEROGENEOUS_DATA_MESH_ADAPTER_H
#define GVT_RENDER_ADAPTER_HETEROGENEOUS_DATA_MESH_ADAPTER_H

#include <gvt/render/adapter/embree/Wrapper.h>
#include <gvt/render/adapter/optix/Wrapper.h>

namespace gvt {
namespace render {
namespace adapter {
namespace heterogeneous {
namespace data {

class HeterogeneousMeshAdapter : public gvt::render::Adapter {
public:
  /**
   * Construct the Embree mesh adapter.  Convert the mesh
   * at the given node to Embree's format.
   *
   * Initializes Embree the first time it is called.
   */
  HeterogeneousMeshAdapter(gvt::core::DBNodeH node);

  /**
   * Release Embree copy of the mesh.
   */
  virtual ~HeterogeneousMeshAdapter();

  /**
   * Return the Embree scene handle;
   */
  void getScene() const {}

  /**
   * Return the geometry id.
   */
  unsigned getGeomId() const { return -1; }

  /**
   * Return the packet size
   */
  unsigned long long getPacketSize() const { return 1024; }

  /**
   * Trace rays using the Embree adapter.
   *
   * Creates threads and traces rays in packets defined by
   * GVT_EMBREE_PACKET_SIZE
   * (currently set to 4).
   *
   * \param rayList incoming rays
   * \param moved_rays outgoing rays [rays that did not hit anything]
   * \param instNode instance db node containing dataRef and transforms
   */
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                     gvt::core::DBNodeH instNode, size_t begin = 0, size_t end = 0);

protected:
  gvt::render::adapter::embree::data::EmbreeMeshAdapter *_embree;
  gvt::render::adapter::optix::data::OptixMeshAdapter *_optix;
};
}
}
}
}
}

#endif /*GVT_RENDER_ADAPTER_HETEROGENEOUS_DATA_MESH_ADAPTER_H*/
