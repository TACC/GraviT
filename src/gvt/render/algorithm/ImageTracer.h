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
/*
 * ImageTracer.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_RENDER_ALGORITHM_IMAGE_TRACER_H
#define GVT_RENDER_ALGORITHM_IMAGE_TRACER_H

#include <mpi.h>

#include <gvt/core/Types.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/core/utils/timer.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/algorithm/TracerBase.h>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/Wrapper.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/Wrapper.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/Wrapper.h>
#endif

#if defined(GVT_RENDER_ADAPTER_OPTIX) && defined(GVT_RENDER_ADAPTER_EMBREE)
#include <gvt/render/adapter/heterogeneous/Wrapper.h>
#endif

#include <boost/timer/timer.hpp>

namespace gvt {
namespace render {
namespace algorithm {
/// work scheduler that strives to keep rays resident and load data domains as needed
/**
  The Image scheduler strives to schedule work such that rays remain resident on their
  initial process and domains are loaded as necessary to retire those rays. Rays are never
  sent to other processes. Domains can be loaded at multiple processes, depending on the
  requirements of the rays at each process.

  This scheduler can become unbalanced when:
   - certain rays require more time to process than others
   - rays at a process require many domains, which can cause memory thrashing
   - when there are few rays remaining to render, other processes can remain idle

     \sa DomainTracer, HybridTracer
   */
template <> class Tracer<gvt::render::schedule::ImageScheduler> : public AbstractTrace {
public:
  // ray range [used when mpi is enabled]
  size_t rays_start, rays_end;

  // caches meshes that are converted into the adapter's format
  std::map<gvt::render::data::primitives::Mesh *, gvt::render::Adapter *> adapterCache;

  Tracer(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image) : AbstractTrace(rays, image) {
    int ray_portion = rays.size() / mpi.world_size;
    rays_start = mpi.rank * ray_portion;
    rays_end = (mpi.rank + 1) == mpi.world_size ? rays.size()
                                                : (mpi.rank + 1) * ray_portion; // tack on any odd rays to last proc
  }

  // organize the rays into queues
  // if using mpi, only keep the rays for the current rank
  inline void FilterRaysLocally() {
    auto nullNode = gvt::core::DBNodeH(); // temporary workaround until shuffleRays is fully replaced

    if (mpi) {
      GVT_DEBUG(DBG_ALWAYS, "image scheduler: filter locally mpi: [" << rays_start << ", " << rays_end << "]");
      gvt::render::actor::RayVector lrays;
      lrays.assign(rays.begin() + rays_start, rays.begin() + rays_end);
      rays.clear();
      shuffleRays(lrays, -1);
    } else {
      GVT_DEBUG(DBG_ALWAYS, "image scheduler: filter locally non mpi: " << rays.size());
      shuffleRays(rays, -1);
    }
  }

  inline void operator()() {

    gvt::core::time::timer t_diff(false, "image tracer: diff timers/frame:");
    gvt::core::time::timer t_all(false, "image tracer: all timers:");
    gvt::core::time::timer t_frame(true, "image tracer: frame :");
    gvt::core::time::timer t_gather(false, "image tracer: gather :");
    gvt::core::time::timer t_shuffle(false, "image tracer: shuffle :");
    gvt::core::time::timer t_trace(false, "image tracer: trace :");
    gvt::core::time::timer t_sort(false, "image tracer: select :");
    gvt::core::time::timer t_adapter(false, "image tracer: adapter :");
    gvt::core::time::timer t_filter(false, "image tracer: filter :");

    GVT_DEBUG(DBG_ALWAYS, "image scheduler: starting, num rays: " << rays.size());
    gvt::core::DBNodeH root = gvt::render::RenderContext::instance()->getRootNode();

    GVT_ASSERT((instancenodes.size() > 0), "image scheduler: instance list is null");
    int adapterType = root["Schedule"]["adapter"].value().toInteger();

    clearBuffer();

    // sort rays into queues
    t_filter.resume();
    FilterRaysLocally();
    t_filter.stop();

    gvt::render::actor::RayVector moved_rays;
    int instTarget = -1, instTargetCount = 0;
    for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) {
      if (q->second.size() > (size_t)instTargetCount) {
        instTargetCount = q->second.size();
        instTarget = q->first;
      }
    }

    if (instTargetCount == 0) {
      //std::cout << "No work to start with" << std::endl;
    }

    // process domains until all rays are terminated
    do {
      // process domain with most rays queued
      instTarget = -1;
      instTargetCount = 0;

      t_sort.resume();
      GVT_DEBUG(DBG_ALWAYS, "image scheduler: selecting next instance, num queues: " << this->queue.size());
      for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end();
           ++q) {
        if (q->second.size() > (size_t)instTargetCount) {
          instTargetCount = q->second.size();
          instTarget = q->first;
        }
      }
      t_sort.stop();

      //std::cout << "Instance[" << instTarget << "] " << instTargetCount << std::endl;

      GVT_DEBUG(DBG_ALWAYS, "image scheduler: next instance: " << instTarget << ", rays: " << instTargetCount);

      if (instTarget >= 0) {
        t_adapter.resume();
        gvt::render::Adapter *adapter = 0;
        // gvt::core::DBNodeH meshNode = instancenodes[instTarget]["meshRef"].deRef();

        gvt::render::data::primitives::Mesh *mesh = meshRef[instTarget];
        // TODO: Make cache generic needs to accept any kind of adpater

        // 'getAdapterFromCache' functionality
        auto it = adapterCache.find(mesh);
        if (it != adapterCache.end()) {
          adapter = it->second;
        } else {
          adapter = 0;
        }
        if (!adapter) {
          GVT_DEBUG(DBG_ALWAYS, "image scheduler: creating new adapter");
          switch (adapterType) {
#ifdef GVT_RENDER_ADAPTER_EMBREE
          case gvt::render::adapter::Embree:
            adapter = new gvt::render::adapter::embree::data::EmbreeMeshAdapter(mesh);
            break;
#endif
#ifdef GVT_RENDER_ADAPTER_MANTA
          case gvt::render::adapter::Manta:
            adapter = new gvt::render::adapter::manta::data::MantaMeshAdapter(mesh);
            break;
#endif
#ifdef GVT_RENDER_ADAPTER_OPTIX
          case gvt::render::adapter::Optix:
            adapter = new gvt::render::adapter::optix::data::OptixMeshAdapter(mesh);
            break;
#endif

#if defined(GVT_RENDER_ADAPTER_OPTIX) && defined(GVT_RENDER_ADAPTER_EMBREE)
          case gvt::render::adapter::Heterogeneous:
            adapter = new gvt::render::adapter::heterogeneous::data::HeterogeneousMeshAdapter(mesh);
            break;
#endif
          default:
            GVT_DEBUG(DBG_SEVERE, "image scheduler: unknown adapter type: " << adapterType);
          }

          adapterCache[mesh] = adapter;
        }
        t_adapter.stop();
        GVT_ASSERT(adapter != nullptr, "image scheduler: adapter not set");
        // end getAdapterFromCache concept

        GVT_DEBUG(DBG_ALWAYS, "image scheduler: calling process queue");
        {
          t_trace.resume();
          moved_rays.reserve(this->queue[instTarget].size() * 10);
#ifdef GVT_USE_DEBUG
          boost::timer::auto_cpu_timer t("Tracing rays in adapter: %w\n");
#endif
          adapter->trace(this->queue[instTarget], moved_rays, instM[instTarget], instMinv[instTarget],
                         instMinvN[instTarget], lights);

          this->queue[instTarget].clear();

          t_trace.stop();
        }

        GVT_DEBUG(DBG_ALWAYS, "image scheduler: marching rays");
        t_shuffle.resume();
        shuffleRays(moved_rays, instTarget);
        moved_rays.clear();
        t_shuffle.stop();
      } else {
        //std::cout << "No more work" << std::endl;
      }
    } while (instTarget != -1);
    GVT_DEBUG(DBG_ALWAYS, "image scheduler: gathering buffers");
    t_gather.resume();
    this->gatherFramebuffers(this->rays.size());

    t_gather.stop();
    t_frame.stop();
    GVT_DEBUG(DBG_ALWAYS, "image scheduler: adapter cache size: " << adapterCache.size());
    // std::cout << "Timers ---------------------------------------------------------------" << std::endl;
    // std::cout << "image scheduler: filter time: " << t_filter.format() << std::endl;
    // std::cout << "image scheduler: select time: " << t_sort.format() << std::endl;
    // std::cout << "image scheduler: adapter time: " << t_adapter.format() << std::endl;
    // std::cout << "image scheduler: trace time: " << t_trace.format() << std::endl;
    // std::cout << "image scheduler: shuffle time: " << t_shuffle.format() << std::endl;
    // std::cout << "image scheduler: gather time: " << t_gather.format() << std::endl;
    // std::cout << "image scheduler: frame time: " << t_frame.format() << std::endl;

    t_all = t_sort + t_trace + t_shuffle + t_gather + t_adapter + t_filter;
    t_diff = t_frame - t_all;
    // std::cout << "image scheduler: added time: " << a.format() << " { unaccounted time: " << (t_frame - a).format()
    //           << " }" << std::endl;
  }
};
}
}
}
#endif /* GVT_RENDER_ALGORITHM_IMAGE_TRACER_H */
