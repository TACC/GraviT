/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
   at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use
   this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software
   distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
//
//  RayTracer.h
//

#ifndef GVTAPPS_RENDER_EMBREE_RAY_TRACER_H
#define GVTAPPS_RENDER_EMBREE_RAY_TRACER_H

#include <gvt/render/Attributes.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <apps/render/ConfigFileLoader.h>

#include <pthread.h>
#include <semaphore.h>

#include <algorithm>
#include <set>

namespace gvtapps {
namespace render {
/// Render using the Intel Embree ray tracer
/**
    Ray traces a scene using the Intel Embree ray tracing engine
   (http://embree.github.io).
    Ray tracer is initialized using a configuration file and rendered using
   RenderImage

    \sa ConfigFileLoader, MantaRayTracer, OptixRayTracer
*/
class EmbreeRayTracer {
public:
  EmbreeRayTracer(gvtapps::render::ConfigFileLoader &cl);

  void RenderImage(std::string);
  gvt::render::actor::RayVector rays;
  gvt::render::data::Dataset *scene;

protected:
  struct LoadBalancer {
    LoadBalancer(size_t size_, int granularity_ = 16)
        : size(size_), granularity(granularity_) {
      blockSize = std::max(size_t(1), size / granularity);
      last = 0;
    }

    void GetWork(size_t &begin, size_t &end) {
      begin = std::min(last, size);
      last += blockSize;
      end = last - 1;
    }

    size_t size, blockSize, last;
    int granularity;
  };

  void IntersectQueueHandler(void *);
  std::vector<pthread_t> _threads;
  sem_t mutex;
  LoadBalancer *loadBalancer;

private:
  void parallel_comm_2(gvt::render::Attributes &rta,
                       gvt::render::actor::RayVector &rays,
                       gvt::render::data::scene::Image &image);
};
}
}

#endif // GVTAPPS_RENDER_EMBREE_RAY_TRACER_H
