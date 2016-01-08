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
#ifndef GVT_RENDER_ADAPTER_H
#define GVT_RENDER_ADAPTER_H

#include <gvt/core/DatabaseNode.h>
#include <gvt/render/actor/Ray.h>

#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/mutex.hpp>

namespace gvt {
namespace render {
/// base class for ray tracing engine adapters
/**  MantaMeshAdapter, EmbreeMeshAdapter, OptixMeshAdapter */
class Adapter {
protected:
  /**
   * Data node (ex: Mesh, Volume)
   */
  gvt::core::DBNodeH node;

public:
  /**
   * Construct an adapter with a given data node
   */
  Adapter(gvt::core::DBNodeH node) : node(node) {}

  /**
   * Destroy the adapter
   */
  virtual ~Adapter() {}

  /**
   * Trace rays using the adapter.
   *
   * \param rayList incoming rays
   * \param moved_rays outgoing rays [rays that did not hit anything]
   * \param instNode instance db node containing dataRef and transforms
   */
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                     gvt::core::DBNodeH instNode, size_t begin = 0, size_t end = 0) = 0;

  boost::mutex _inqueue;
  boost::mutex _outqueue;
};

} // render
} // gvt

#endif // GVT_RENDER_ADAPTER_H
