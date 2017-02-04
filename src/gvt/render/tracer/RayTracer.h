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

#ifndef GVT_RENDER_RAYTRACER
#define GVT_RENDER_RAYTRACER

#include <gvt/core/tracer/tracer.h>
#include <gvt/render/Adapter.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Types.h>
#include <gvt/render/composite/IceTComposite.h>
#include <gvt/render/composite/ImageComposite.h>
#include <gvt/render/data/accel/BVH.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>

namespace gvt {
namespace render {

class RayTracer : public gvt::core::Tracer {
private:
protected:
  std::shared_ptr<gvt::render::composite::ImageComposite> img;
  std::shared_ptr<gvt::render::data::scene::gvtCameraBase> cam;
  std::shared_ptr<gvt::render::data::accel::BVH> bvh;
  gvt::render::RenderContext *cntxt;

  // Caching
  gvt::core::Map<int, gvt::render::data::primitives::Mesh *> meshRef;
  gvt::core::Map<int, glm::mat4 *> instM;
  gvt::core::Map<int, glm::mat4 *> instMinv;
  gvt::core::Map<int, glm::mat3 *> instMinvN;
  gvt::core::Vector<gvt::render::data::scene::Light *> lights;
  gvt::core::Map<gvt::render::data::primitives::Mesh *, std::shared_ptr<gvt::render::Adapter> > adapterCache;
  int adapterType;

public:
  RayTracer();
  ~RayTracer();
  virtual void operator()();
  virtual void calladapter(const int instTarget, gvt::render::actor::RayVector &toprocess,
                           gvt::render::actor::RayVector &moved_rays);
  virtual void processRays(gvt::render::actor::RayVector &rays, const int src = -1, const int dst = -1);

  virtual bool MessageManager(std::shared_ptr<gvt::comm::Message> msg);
  virtual bool isDone();
  virtual bool hasWork();
  virtual float *getImageBuffer();
  virtual void resetCamera();
  virtual void resetFilm();
  virtual void resetBVH();

  virtual std::shared_ptr<gvt::render::composite::ImageComposite> getComposite() { return img; }
};
};
};

#endif /*GVT_RENDER_RAYTRACER*/
