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
#include <gvt/render/Schedulers.h>
#include <gvt/render/algorithm/TracerBase.h>
#include <gvt/render/algorithm/MetaProcessQueue.h>

#include <boost/timer/timer.hpp>

namespace gvt {
    namespace render {
        namespace algorithm {
            /// Tracer Image (ImageSchedule) based decomposition implementation

            template<class DomainType, class MPIW> class Tracer<DomainType, MPIW, gvt::render::schedule::ImageScheduler> : public TracerBase<MPIW> 
            {
            public:

                Tracer(gvt::render::actor::RayVector& rays, gvt::render::data::scene::Image& image) 
                : TracerBase<MPIW>(rays, image) 
                {
                    int ray_portion = this->rays.size() / this->world_size;
                    this->rays_start = this->rank * ray_portion;
                    this->rays_end = (this->rank + 1) == this->world_size ? this->rays.size() : (this->rank + 1) * ray_portion; // tack on any odd rays to last proc
                }

                virtual void operator()() 
                {
                    GVT_DEBUG(DBG_ALWAYS,"Using Image schedule");
                    boost::timer::auto_cpu_timer t;

                    long ray_counter = 0, domain_counter = 0;

                    this->FilterRaysLocally();

                    // buffer for color accumulation
                    gvt::render::actor::RayVector moved_rays;
                    int domTarget = -1, domTargetCount = 0;
                    // process domains until all rays are terminated
                    do 
                    {
                        // process domain with most rays queued
                        domTarget = -1;
                        domTargetCount = 0;

                        GVT_DEBUG(DBG_ALWAYS, "Selecting new domain");
                        for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) 
                        {
                            if (q->second.size() > domTargetCount) 
                            {
                                domTargetCount = q->second.size();
                                domTarget = q->first;
                            }
                        }
                        GVT_DEBUG(DBG_ALWAYS, "Selecting new domain");
                        //if (domTarget != -1) std::cout << "Domain " << domTarget << " size " << this->queue[domTarget].size() << std::endl;
                        GVT_DEBUG_CODE(DBG_ALWAYS,if (DEBUG_RANK) std::cerr << this->rank << ": selected domain " << domTarget << " (" << domTargetCount << " rays)" << std::endl);
                        GVT_DEBUG_CODE(DBG_ALWAYS,if (DEBUG_RANK) std::cerr << this->rank << ": currently processed " << ray_counter << " rays across " << domain_counter << " domains" << std::endl);

                        if (domTarget >= 0) 
                        {

                            GVT_DEBUG(DBG_ALWAYS, "Getting domain " << domTarget << std::endl);
                            gvt::render::data::domain::AbstractDomain* dom = gvt::render::Attributes::rta->dataset->getDomain(domTarget);
                            dom->load();
                            GVT_DEBUG(DBG_ALWAYS, "dom: " << domTarget << std::endl);

                            // track domain loads
                            ++domain_counter;

                            GVT_DEBUG(DBG_ALWAYS, "Calling process queue");
                            //GVT::Backend::ProcessQueue<DomainType>(new GVT::Backend::adapt_param<DomainType>(this->queue, moved_rays, domTarget, dom, this->colorBuf, ray_counter, domain_counter))();
                            {
                                moved_rays.reserve(this->queue[domTarget].size()*10);
                                boost::timer::auto_cpu_timer t("Tracing domain rays %t\n");
                                dom->trace(this->queue[domTarget], moved_rays);
                            }
                            GVT_DEBUG(DBG_ALWAYS, "Marching rays");
                            //                        BOOST_FOREACH( gvt::render::actor::Ray* mr,  moved_rays) {
                            //                            dom->marchOut(mr);
                            //                            gvt::core::schedule::asyncExec::instance()->run_task(processRay(this,mr));
                            //                        }

                            boost::atomic<int> current_ray(0);
                            size_t workload = std::max((size_t)1,(size_t)(moved_rays.size() / (gvt::core::schedule::asyncExec::instance()->numThreads * 2)));
                            {
                                boost::timer::auto_cpu_timer t("Scheduling rays %t\n");
                                for (int rc = 0; rc < gvt::core::schedule::asyncExec::instance()->numThreads; ++rc) {
                                    gvt::core::schedule::asyncExec::instance()->run_task(processRayVector(this, moved_rays, current_ray, moved_rays.size(),workload, dom));
                                }
                                gvt::core::schedule::asyncExec::instance()->sync();

                            }
                            GVT_DEBUG(DBG_ALWAYS, "Finished queueing");
                            gvt::core::schedule::asyncExec::instance()->sync();
                            GVT_DEBUG(DBG_ALWAYS, "Finished marching");
                            //dom->free();
                            moved_rays.clear();
                            //this->queue.erase(domTarget); // TODO: for secondary rays, rays may have been added to this domain queue
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
