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

#ifndef GVT_RENDER_IMAGETRACER
#define GVT_RENDER_IMAGETRACER

#include <gvt/render/tracer/RayTracer.h>
#include <mutex>

namespace gvt {
namespace render {
/**
 * \Brief ray trace Image Decomposition scheduler
 *
 * Implements a image decomposition strategy to generate the final image. The image space is divided equally amount all
 * compute nodes and each node assumes that they have aceess to the entire data. The termination condition only needs to
 * check that there is no more rays to be processed localy does voting and message passing is not required.
 */
class ImageTracer : public gvt::render::RayTracer {
private:
protected:
public:
  ImageTracer(const std::string& name,std::shared_ptr<gvt::render::data::scene::gvtCameraBase> cam,
              std::shared_ptr<gvt::render::composite::ImageComposite> img);
  ~ImageTracer();

  /**
   * \brief Ray trace image decomposition implementation
   *
   * @method operator
   */
  virtual void operator()();

  /**
   * Checks if the rays intersect any of the high level bounding boxes that encapsulate each instance and places the ray
   * in the first intersected instance queue. All other rays are dropped.
   * @method processRaysAndDrop
   * @param  rays               [description]
   */
  virtual void processRaysAndDrop(gvt::render::actor::RayVector &rays);
  /**
   * Process rays returned by the adpater
   * @method processRays
   * @param  rays        List of rays returned by the adpater
   * @param  src         Instance id from where the rays originated
   * @param  dst         Instance id of rays destination (not used)
   */
  virtual void processRays(gvt::render::actor::RayVector &rays, const int src = -1, const int dst = -1);

  /**
   * Message Manager, to used since no messages are exchanged in this scheduler
   *
   * @method MessageManager
   * @param  msg            Raw message
   * @return                [description]
   */
  virtual bool MessageManager(std::shared_ptr<gvt::comm::Message> msg);

  /**
   * Checks id all queues are empty
   * @method isDone
   * @return return true if all instance queues are empty
   */
  virtual bool isDone();
  /**
   * Checks if at least one instance queue has work
   * @method hasWork
   */
  virtual bool hasWork();

  /**
   * Reset BVH and update
   * @method resetBVH
   */
  virtual void resetBVH();
};
}; // namespace render
}; // namespace gvt

#endif /*GVT_RENDER_IMAGETRACER*/
