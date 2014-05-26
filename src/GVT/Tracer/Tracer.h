/*
 * Tracer.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_TRACER_H
#define GVT_TRACER_H

#include <mpi.h>
#include <GVT/Data/primitives.h>
#include <GVT/Environment/RayTracerAttributes.h>
#include <GVT/Data/scene/Image.h>
#include <GVT/common/debug.h>



namespace GVT {

    namespace Trace {

        /// Tracer base class

        class abstract_trace {
        public:

            GVT::Env::RayTracerAttributes& rta; ///< Ray tracer attributes
            GVT::Data::RayVector& rays; ///< Rays to trace
            Image& image; ///< Final image buffer

            unsigned char *vtf;
            float sample_ratio;

            map<int, GVT::Data::RayVector> queue; ///< Node rays working queue

            // buffer for color accumulation
            COLOR_ACCUM* colorBuf;

            abstract_trace(GVT::Env::RayTracerAttributes& rta, GVT::Data::RayVector& rays, Image& image) : rta(rta), rays(rays), image(image) {
                // get transfer function and set opacity map
                this->vtf = this->rta.GetTransferFunction();
                this->sample_ratio = this->rta.sample_ratio;
                this->colorBuf = new COLOR_ACCUM[this->rays.size()];
                //memset(this->colorBuf, 0, sizeof (this->colorBuf) * this->rays.size());
            }

            virtual ~abstract_trace() {
                delete[] colorBuf;
            };
            virtual void operator()(void) = 0;
            virtual void generateRays(void) = 0;
            virtual bool SendRays() = 0;
            virtual void gatherFramebuffers(int rays_traced) = 0;

            virtual void addRay(GVT::Data::ray& r) {
                vector<int> len2List;
                this->rta.dataset->Intersect(r, len2List);
                if (!len2List.empty()) {
                    r.domains.assign(len2List.begin(),len2List.end());
                    queue[len2List[0]].push_back(r);
                    return;
                }
                for (int i = 0; i < 3; i++) colorBuf[r.id].rgba[i] += r.color.rgba[i];
                colorBuf[r.id].rgba[3] = 1.f;
                colorBuf[r.id].clamp();
            }
        };

        /*!
         * Defines properties for inheritance  
         * 
         * \tparam MPIW MPI communication world (SINGLE NODE, MPI_Comm)
         */
        template<class MPIW>
        class Tracer_base : public MPIW, public abstract_trace {
        public:

            Tracer_base(GVT::Env::RayTracerAttributes& rta, GVT::Data::RayVector& rays, Image& image) : MPIW(rays.size()), abstract_trace(rta, rays, image) {
            }

            virtual ~Tracer_base() {
            };

            /*! Trace operation
             * 
             * Implements the code for the tracer 
             * 
             * 
             */

            virtual void operator()(void) {
                GVT_ASSERT_BACKTRACE(0, "Not supported");
            }

            /*! Generate rays 
             * 
             * Used inside the operator to generate the tree of rays.
             * 
             * 
             */

            virtual void generateRays() {
                for (int rc = this->rays_start; rc < this->rays_end; ++rc) {
                    DEBUG(cerr << endl << "Seeding ray " << rc << ": " << this->rays[rc] << endl);
                    vector<int> len2List;
                    this->rta.dataset->Intersect(this->rays[rc], len2List);
                    if (!len2List.empty()) {
                        for (int i = len2List.size() - 1; i >= 0; --i)
                            this->rays[rc].domains.push_back(len2List[i]); // insert domains in reverse order
                        queue[len2List[0]].push_back(this->rays[rc]); // TODO: make this a ref?
                    }
                }
            }

            /*! Gather buffers from all distributed nodes
             *  
             * \param colorBuf Local color buffer
             * \param rays_traced number of rays traced
             * 
             * \note Should be called at the end of the operator
             */
            virtual void gatherFramebuffers(int rays_traced) {


                for (int i = 0; i < rta.view.width * rta.view.height; ++i) {
                    this->image.Add(i, colorBuf[i]);
                }

                unsigned char* rgb = this->image.GetBuffer();

                int rgb_buf_size = 3 * rta.view.width * rta.view.height;

                unsigned char *bufs = (this->rank == 0) ? new unsigned char[this->world_size * rgb_buf_size] : NULL;

                MPI_Gather(rgb, rgb_buf_size, MPI_UNSIGNED_CHAR, bufs, rgb_buf_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

                // XXX TODO: find a better way to merge the color buffers
                if (this->rank == 0) {
                    // merge into root proc rgb

                    GVT_DEBUG(DBG_ALWAYS, "Gathering buffers");
                    for (int i = 1; i < this->world_size; ++i) {
                        for (int j = 0; j < rgb_buf_size; j += 3) {
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
            virtual bool SendRays() {
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
        class Tracer : public Tracer_base<MPIW> {
        public:

            Tracer(GVT::Env::RayTracerAttributes& rta, GVT::Data::RayVector& rays, Image& image) : Tracer_base<MPIW>(rta, rays, image) {
            }

            virtual ~Tracer() {
            };

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
        class Tracer<DomainType, MPIW, BSCHEDULER<ISCHEDULER> > : public Tracer_base<MPIW> {
        public:

            Tracer(GVT::Env::RayTracerAttributes& rta, GVT::Data::RayVector& rays, Image& image) : Tracer_base<MPIW>(rta, rays, image) {
            }

            virtual ~Tracer() {
            };

        };

    };
};




#endif /* GVT_TRACER_H */
