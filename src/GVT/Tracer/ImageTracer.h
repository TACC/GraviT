/*
 * ImageTracer.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_IMAGETRACER_H
#define GVT_IMAGETRACER_H

#include <mpi.h>


#include <GVT/MPI/mpi_wrappers.h>
#include <GVT/Tracer/Tracer.h>
#include <GVT/Backend/MetaProcessQueue.h>
#include <GVT/Scheduler/schedulers.h>

namespace GVT {

    namespace Trace {
/// Tracer Image (ImageSchedule) based decomposition implementation

        template<class DomainType, class MPIW> class Tracer<DomainType, MPIW, ImageSchedule> : public Tracer_base<MPIW> {
        public:

            Tracer(GVT::Env::RayTracerAttributes& rta, GVT::Data::RayVector& rays, Image& image) : Tracer_base<MPIW>(rta, rays, image) {
                int ray_portion = this->rays.size() / this->world_size;
                this->rays_start = this->rank * ray_portion;
                this->rays_end = (this->rank + 1) == this->world_size ? this->rays.size() : (this->rank + 1) * ray_portion; // tack on any odd rays to last proc
            }

            virtual void operator()() {

                long ray_counter = 0, domain_counter = 0;

                this->generateRays();

                // buffer for color accumulation
                GVT::Data::RayVector moved_rays;
                int domTarget = -1, domTargetCount = 0;
                // process domains until all rays are terminated
                while (!this->queue.empty()) {
                    // process domain with most rays queued
                    domTarget = -1;
                    domTargetCount = 0;

                    for (map<int, GVT::Data::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) {
                        if (q->second.size() > domTargetCount) {
                            domTargetCount = q->second.size();
                            domTarget = q->first;
                        }
                    }


                    if (DEBUG_RANK) GVT_DEBUG(DBG_ALWAYS, this->rank << ": selected domain " << domTarget << " (" << domTargetCount << " rays)");
                    if (DEBUG_RANK) GVT_DEBUG(DBG_ALWAYS, this->rank << ": currently processed " << ray_counter << " rays across " << domain_counter << " domains");

                    if (domTarget >= 0) {

                        GVT_DEBUG(DBG_ALWAYS,"Getting domain " << domTarget << endl);
                        GVT::Domain::Domain* dom = this->rta.dataset->getDomain(domTarget);
                        dom->load();
                        GVT_DEBUG(DBG_ALWAYS,"dom: " << domTarget << endl);

                        // track domain loads
                        ++domain_counter;
                        GVT::Backend::ProcessQueue<DomainType>(new GVT::Backend::adapt_param<DomainType>(this->queue, moved_rays, domTarget, dom, this->rta, this->colorBuf, ray_counter, domain_counter))();
                        while (!moved_rays.empty()) {
                            GVT::Data::ray& mr = moved_rays.back();
                            moved_rays.pop_back();
                            dom->marchOut(mr);
                            if(!mr.domains.empty()) {
                                int target = mr.domains.back();
                                this->queue[target].push_back(mr);
                                this->rta.dataset->getDomain(target)->marchIn(mr);
                                mr.domains.pop_back();
                            } else {
                                this->addRay(mr);
                            }   
                        }
                        dom->free();
                        this->queue.erase(domTarget); // TODO: for secondary rays, rays may have been added to this domain queue
                    }

                }
                this->gatherFramebuffers(this->rays.size());
            }
        };
    };
};
#endif /* IMAGETRACER_H_ */
