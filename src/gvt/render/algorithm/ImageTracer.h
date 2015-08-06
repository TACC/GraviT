/*
 * ImageTracer.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_RENDER_ALGORITHM_IMAGE_TRACER_H
#define GVT_RENDER_ALGORITHM_IMAGE_TRACER_H

#include <mpi.h>


#include <gvt/core/mpi/Wrapper.h>
#include <gvt/core/Types.h>
#include <gvt/render/Types.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/algorithm/TracerBase.h>
#include <gvt/render/adapter/embree/Wrapper.h>

#include <boost/timer/timer.hpp>

namespace gvt {
    namespace render {
        namespace algorithm {
            /// Tracer Image (ImageSchedule) based decomposition implementation

            template<> class Tracer<gvt::render::schedule::ImageScheduler> : public AbstractTrace
            {
            public:

                size_t rays_start, rays_end ;

                Tracer(gvt::render::actor::RayVector& rays, gvt::render::data::scene::Image& image) 
                : AbstractTrace(rays, image) 
                {
                    GVT_DEBUG(DBG_ALWAYS, "image trace: constructor start");
                    int ray_portion = rays.size() / mpi.world_size;
                    rays_start = mpi.rank * ray_portion;
                    rays_end = (mpi.rank + 1) == mpi.world_size ? rays.size() : (mpi.rank + 1) * ray_portion; // tack on any odd rays to last proc
                    GVT_DEBUG(DBG_ALWAYS, "abstract trace: num instances [third time]: " << instancenodes.size());
                    GVT_DEBUG(DBG_ALWAYS, "image trace: constructor end");
                }

                virtual void FilterRaysLocally() {
                    if(mpi) {
                        GVT_DEBUG(DBG_ALWAYS,"filter locally mpi: " << rays.size());
                        gvt::render::actor::RayVector lrays;
                        lrays.assign(rays.begin() + rays_start,
                                     rays.begin() + rays_end);
                        rays.clear();
                        shuffleRays(lrays);        
                    } else {
                        GVT_DEBUG(DBG_ALWAYS,"filter locally non mpi: " << rays.size());
                        shuffleRays(rays);
                    }
                }

                virtual void operator()() 
                {
                    GVT_DEBUG(DBG_ALWAYS,"Using Image schedule");
                    boost::timer::auto_cpu_timer t;

                    GVT_DEBUG(DBG_ALWAYS, "image trace: operator(): num instances: " << instancenodes.size());
                    long ray_counter = 0, domain_counter = 0;

                    GVT_DEBUG(DBG_ALWAYS,"number of rays: " << rays.size());

                    FilterRaysLocally();

                    GVT_DEBUG(DBG_ALWAYS, "image trace: operator(): num instances after filter ray: " << instancenodes.size());

                    // buffer for color accumulation
                    gvt::render::actor::RayVector moved_rays;
                    int domTarget = -1, domTargetCount = 0;
                    // process domains until all rays are terminated
                    do 
                    {
                        // process domain with most rays queued
                        domTarget = -1;
                        domTargetCount = 0;

                        GVT_DEBUG(DBG_ALWAYS, "Selecting new domain: num queues: " << this->queue.size());
                        for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) 
                        {
                            if (q->second.size() > domTargetCount) 
                            {
                                domTargetCount = q->second.size();
                                domTarget = q->first;
                            }
                        }
                        GVT_DEBUG(DBG_ALWAYS, "new domain: " << domTarget);

                        if (domTarget >= 0) 
                        {
                            gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
                            gvt::core::DBNodeH root = ctx->getRootNode();

                            GVT_DEBUG(DBG_ALWAYS, "Getting domain " << domTarget << std::endl);

                            if(instancenodes.size() == 0) {
                                GVT_ASSERT((instancenodes.size() == 0), "instance list is null");
                            }

                            //
                            // 'getAdapter' functionality [without the cache part]
                            //
                            gvt::core::DBNodeH meshnode = instancenodes[0]["meshRef"].deRef();
                            int adapterType = gvt::core::variant_toInteger(root["Schedule"]["adapter"].value());

                            gvt::render::data::domain::GeometryDomain* geoDomain = 0; // TODO: remove this geodomain and have embree build from mesh directly
                            gvt::render::data::domain::AbstractDomain* adapter = 0; // TODO: rename to 'Adapter'
                            // here we would do something like Adapter *a = cache.find(meshnode, adaptertype);

                            if(!adapter) {
                                //   'getMesh' - right now in SimpleApp mesh is hard coded and pointer is valid
                                auto m = gvt::core::variant_toMeshPtr(meshnode["ptr"].value());

                                switch(adapterType) {
                                    case gvt::render::adapter::Embree:
                                        {
                                            geoDomain = new gvt::render::data::domain::GeometryDomain(m);
                                            adapter = new gvt::render::adapter::embree::data::domain::EmbreeDomain(geoDomain);
                                            break;
                                        }
                                    default:
                                        {
                                            GVT_DEBUG(DBG_SEVERE, "image scheduler: unknown adapter type: " << adapterType);
                                        }
                                }
                            }

                            GVT_ASSERT(adapter != nullptr, "image scheduler: adapter not set");


#if 0
                            gvt::render::data::domain::GeometryDomain* geoDomain = new gvt::render::data::domain::GeometryDomain(m);


                            gvt::render::data::domain::AbstractDomain* dom = 0;
                            if(firstAdapter == gvt::render::adapter::Embree) {
                                dom = new gvt::render::adapter::embree::data::domain::EmbreeDomain(geoDomain);
                            } else {
                                // TODO: for now, just default to embree
                                dom = new gvt::render::adapter::embree::data::domain::EmbreeDomain(geoDomain);
                            }
                            dom->load();
                            GVT_DEBUG(DBG_ALWAYS, "dom: " << domTarget << std::endl);
#endif

                            // track domain loads
                            ++domain_counter;

                            GVT_DEBUG(DBG_ALWAYS, "Calling process queue");
                            {
                                moved_rays.reserve(this->queue[domTarget].size()*10);
                                boost::timer::auto_cpu_timer t("Tracing domain rays %t\n");
                                adapter->trace(this->queue[domTarget], moved_rays);
                            }
                            GVT_DEBUG(DBG_ALWAYS, "Marching rays");
                            shuffleRays(moved_rays, adapter);
                            moved_rays.clear();

                            //delete adapter; // TODO: later this will be cached by the scheduler
                            delete geoDomain;
                        }
                    } while (domTarget != -1);
                    GVT_DEBUG(DBG_ALWAYS, "Gathering buffers");
                    this->gatherFramebuffers(this->rays.size());
                }
            };
        }
    }
}
#endif /* GVT_RENDER_ALGORITHM_IMAGE_TRACER_H */
