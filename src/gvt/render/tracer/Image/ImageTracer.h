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
class ImageTracer : public gvt::render::RayTracer {
private:
protected:
  std::mutex *queue_mutex = nullptr;
  gvt::core::Map<int, gvt::render::actor::RayVector> queue;

public:
  ImageTracer();
  ~ImageTracer();

  virtual void operator()();
  virtual void processRaysAndDrop(gvt::render::actor::RayVector &rays);
  virtual void processRays(gvt::render::actor::RayVector &rays, const int src = -1, const int dst = -1);

  virtual bool MessageManager(std::shared_ptr<gvt::comm::Message> msg);
  virtual bool isDone();
  virtual bool hasWork();

  virtual void resetBVH();
};
};
};

#endif /*GVT_RENDER_IMAGETRACER*/
