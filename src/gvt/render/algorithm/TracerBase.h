/*
 * Tracer.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_RENDER_ALGORITHM_TRACER_BASE_H
#define GVT_RENDER_ALGORITHM_TRACER_BASE_H

#include <gvt/core/Debug.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Image.h>

#include <boost/foreach.hpp>

#include <mpi.h>

#include <map>

namespace gvt {
    namespace render {
        namespace algorithm {

            /// Tracer base class


            struct processRay;

            class AbstractTrace 
            {
            public:

                gvt::render::actor::RayVector& rays; ///< Rays to trace
                gvt::render::data::scene::Image& image; ///< Final image buffer

                unsigned char *vtf;
                float sample_ratio;

                boost::mutex raymutex;
                boost::mutex* queue_mutex;
                std::map<int, gvt::render::actor::RayVector> queue; ///< Node rays working queue

                // buffer for color accumulation
                boost::mutex* colorBuf_mutex;
                GVT_COLOR_ACCUM* colorBuf;

                AbstractTrace(gvt::render::actor::RayVector& rays, gvt::render::data::scene::Image& image) : rays(rays), image(image) 
                {
                    this->vtf = gvt::render::Attributes::rta->GetTransferFunction();
                    this->sample_ratio = gvt::render::Attributes::rta->sample_ratio;
                    this->colorBuf = new GVT_COLOR_ACCUM[gvt::render::RTA::instance()->view.width * gvt::render::RTA::instance()->view.height];
                    this->queue_mutex = new boost::mutex[gvt::render::Attributes::rta->dataset->size()];
                    this->colorBuf_mutex = new boost::mutex[gvt::render::RTA::instance()->view.width];
                    
                }

                virtual ~AbstractTrace() 
                {
                    delete[] colorBuf;
                };
                virtual void operator()(void) = 0;
                virtual void FilterRaysLocally(void) = 0;
                virtual bool SendRays() = 0;
                virtual void gatherFramebuffers(int rays_traced) = 0;

                virtual void addRay(gvt::render::actor::Ray& ray) 
                {
                    gvt::render::actor::isecDomList len2List;
                    if (len2List.empty()) gvt::render::Attributes::rta->dataset->intersect(ray, len2List);
                    if (!len2List.empty()) {
                        ray.domains.assign(len2List.begin() + 1, len2List.end());
                        int domTarget = (*len2List.begin());
                        gvt::render::Attributes::rta->dataset->getDomain(domTarget)->marchIn(ray);
                        queue[domTarget].push_back(ray);
                        return;
                    } else {
                        for (int i = 0; i < 3; i++) colorBuf[ray.id].rgba[i] += ray.color.rgba[i];
                        colorBuf[ray.id].rgba[3] = 1.f;
                        colorBuf[ray.id].clamp();
                    }
                }
            };

            struct processRayVector 
            {
                AbstractTrace* tracer;
                gvt::render::actor::RayVector& rays;
                boost::atomic<int>& current_ray;
                int last;
                const size_t split;
                gvt::render::data::domain::AbstractDomain* dom;

                processRayVector(AbstractTrace* tracer, gvt::render::actor::RayVector& rays, boost::atomic<int>& current_ray, int last, 
                    const int split,  gvt::render::data::domain::AbstractDomain* dom =NULL) 
                : tracer(tracer), rays(rays), current_ray(current_ray), last(last), split(split), dom(dom) 
                {}

                void operator()() 
                {    
                    gvt::render::actor::RayVector localQueue;
                    while (!rays.empty()) 
                    {
                        localQueue.clear();
                        boost::unique_lock<boost::mutex> lock(tracer->raymutex);
                        std::size_t range = std::min(split, rays.size());
			
                        GVT_DEBUG(DBG_ALWAYS,"processRayVector: current_ray " << current_ray << " last ray " << last << " split " << split << " rays.size()" << rays.size());

                        localQueue.assign(rays.begin(), rays.begin() + range);
                        rays.erase(rays.begin(), rays.begin() + range);
                        lock.unlock();

                        for (int i = 0; i < localQueue.size(); i++) 
                        {
                            gvt::render::actor::Ray& ray = localQueue[i];
                            gvt::render::actor::isecDomList& len2List = ray.domains;
                            if (len2List.empty() && dom) dom->marchOut(ray);
                            if (len2List.empty()) gvt::render::Attributes::rta->dataset->intersect(ray, len2List);
                            if (!len2List.empty()) 
                            {
                                int domTarget = (*len2List.begin());
                                len2List.erase(len2List.begin());
                                boost::mutex::scoped_lock qlock(tracer->queue_mutex[domTarget]);
                                tracer->queue[domTarget].push_back(ray);
                            } 
                            else 
                            {
                                boost::mutex::scoped_lock fbloc(tracer->colorBuf_mutex[ray.id % gvt::render::Attributes::instance()->view.width]);
                                for (int i = 0; i < 3; i++) tracer->colorBuf[ray.id].rgba[i] += ray.color.rgba[i];
                                tracer->colorBuf[ray.id].rgba[3] = 1.f;
                                tracer->colorBuf[ray.id].clamp();
                            }
                        }
                    }
                }
            };

            /*!
             * Defines properties for inheritance  
             * 
             * \tparam MPIW MPI communication world (SINGLE NODE, MPI_Comm)
             */
            template<class MPIW>
            class TracerBase : public MPIW, public AbstractTrace 
            {
            public:

                TracerBase(gvt::render::actor::RayVector& rays, 
                    gvt::render::data::scene::Image& image) 
                : MPIW(rays.size()), AbstractTrace(rays, image) 
                {}

                virtual ~TracerBase() 
                {}

                /*! Trace operation
                 * 
                 * Implements the code for the tracer 
                 * 
                 * 
                 */

                virtual void operator()(void) 
                {
                    GVT_ASSERT_BACKTRACE(0, "Not supported");
                }

                /*! Generate rays 
                 * 
                 * Used inside the operator to generate the tree of rays.
                 * 
                 * 
                 */

                virtual void FilterRaysLocally() 
                {
                    boost::atomic<int> current_ray(this->rays_start);
                    GVT_DEBUG(DBG_ALWAYS,"Generate rays filtering : " << this->rays.size());
                    size_t workload = this->rays.size(); 
                    //size_t workload = this->rays_end - this->rays_start;
                    //std::max((size_t)1,(size_t)(this->rays.size() / (GVT::Concurrency::asyncExec::instance()->numThreads * 2)));
    //                //for (int rc = 0; rc < GVT::Concurrency::asyncExec::instance()->numThreads; ++rc) {
    //                //    GVT::Concurrency::asyncExec::instance()->run_task(processRayVector(this, this->rays,current_ray,this->rays_end,workload));
    //                //}
    //                // GVT::Concurrency::asyncExec::instance()->sync();
    //                
    //                //GVT_DEBUG(DBG_ALWAYS,"Current ray: " << (int)current_ray);
                    processRayVector(this, this->rays,current_ray,this->rays_end,workload)();  
                    GVT_DEBUG(DBG_ALWAYS,"Filtered rays : " << this->rays.size());
    //                
                }



                //GVT::Concurrency::asyncExec::singleton->syncAll();

                /*! Gather buffers from all distributed nodes
                 *  
                 * \param colorBuf Local color buffer
                 * \param rays_traced number of rays traced
                 * 
                 * \note Should be called at the end of the operator
                 */
                virtual void gatherFramebuffers(int rays_traced) 
                {


                    for (int i = 0; i < gvt::render::Attributes::rta->view.width * gvt::render::Attributes::rta->view.height; ++i) 
                    {
                        this->image.Add(i, colorBuf[i]);
                    }

                    unsigned char* rgb = this->image.GetBuffer();

                    int rgb_buf_size = 3 * gvt::render::Attributes::rta->view.width * gvt::render::Attributes::rta->view.height;

                    unsigned char *bufs = (this->rank == 0) ? new unsigned char[this->world_size * rgb_buf_size] : NULL;

                    MPI_Gather(rgb, rgb_buf_size, MPI_UNSIGNED_CHAR, bufs, rgb_buf_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

                    // XXX TODO: find a better way to merge the color buffers
                    if (this->rank == 0) 
                    {
                        // merge into root proc rgb

                        GVT_DEBUG(DBG_ALWAYS, "Gathering buffers");
                        for (int i = 1; i < this->world_size; ++i) 
                        {
                            for (int j = 0; j < rgb_buf_size; j += 3) 
                            {
                                int p = i * rgb_buf_size + j;
                                // assumes black background, so adding is fine (r==g==b== 0)
                                rgb[j + 0] += bufs[p + 0];
                                rgb[j + 1] += bufs[p + 1];
                                rgb[j + 2] += bufs[p + 2];
                                //GVT_DEBUG(DBG_ALWAYS,"r:" << rgb[j + 0]  << " g:"<< rgb[j + 1] << " b:" << rgb[j + 2]  );
                            }
                        }

                        // clean up
                    }

                    DEBUG(if (DEBUG_RANK) cerr << this->rank << ": rgb buffer merge done" << endl);

                    delete[] bufs;

                }

                /*! Communication with other nodes
                 * 
                 * Implements the communication and scheduling strategy among nodes.
                 * 
                 * \note Called from the operator
                 * 
                 */
                virtual bool SendRays() 
                {
                    GVT_ASSERT_BACKTRACE(false, "Not supported");
                    return false;
                }
            };

            /// Generic Tracer interface for a base scheduling strategy with static inner scheduling policy

            /*! Tracer implementation generic interface
             * 
             * \tparam DomainType Data domain type. Besides defining the domain behavior defines the procedure to process the current queue of rays
             * \tparam MPIW MPI Communication World (Single node or Multiple Nodes) 
             * \tparam BSCHEDUDER Base tracer scheduler (e.g. Image, Domain or Hybrid)
             * 
             */

            template<class DomainType, class MPIW, class BSCHEDULER>
            class Tracer : public TracerBase<MPIW> 
            {
            public:

                Tracer(gvt::render::actor::RayVector& rays, gvt::render::data::scene::Image& image) 
                : TracerBase<MPIW>(rays, image) 
                {}

                virtual ~Tracer() 
                {}

            };
            /// Generic Tracer interface for a base scheduling strategy with mutable inner scheduling policy

            /*! Tracer implementation generic interface for scheduler with mutable inner scheduling policy
             * 
             * \tparam DomainType Data domain type. Besides defining the domain behavior defines the procedure to process the current queue of rays
             * \tparam MPIW MPI Communication World (Single node or Multiple Nodes) 
             * \tparam BSCHEDUDER Base tracer scheduler (e.g.Hybrid)
             * \tparam ISCHEDUDER Inner scheduler for base scheduler (Greedy, Spread, ...)
             * 
             */
            template<class DomainType, class MPIW, template<typename> class BSCHEDULER, class ISCHEDULER>
            class Tracer<DomainType, MPIW, BSCHEDULER<ISCHEDULER> > : public TracerBase<MPIW> 
            {
            public:

                Tracer(gvt::render::actor::RayVector& rays, gvt::render::data::scene::Image& image) 
                : TracerBase<MPIW>(rays, image) 
                {}

                virtual ~Tracer() {}
            };

        }
    }
}

#endif /* GVT_RENDER_ALGORITHM_TRACER_BASE_H */
