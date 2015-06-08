/*
 * DomainTracer.h
 *
 *  Created on: Dec 8, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_RENDER_ALGORITHM_DOMAIN_TRACER_H
#define GVT_RENDER_ALGORITHM_DOMAIN_TRACER_H

#include <gvt/core/mpi/Wrapper.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/algorithm/TracerBase.h>
#include <gvt/render/algorithm/MetaProcessQueue.h>
#include <gvt/render/Schedulers.h>

#include <boost/foreach.hpp>

#include <mpi.h>
#include <set>

#define RAY_BUF_SIZE 10485760  // 10 MB per neighbor

 namespace gvt {
    namespace render {
        namespace algorithm {
            /// Tracer Domain (DomainSchedule) based decomposition implementation

            template<> class Tracer<gvt::render::schedule::DomainScheduler> : public AbstractTrace 
            {
            public:

                std::set<int> neighbors;

                size_t rays_start, rays_end ;

                Tracer(gvt::render::actor::RayVector& rays, gvt::render::data::scene::Image& image) 
                : AbstractTrace(rays, image) 
                {}

                virtual ~Tracer() { }

                virtual void FilterRaysLocally() 
                {
                    AbstractTrace::FilterRaysLocally();

                    // for (int rc = rays_start; rc < rays_end; ++rc) 
                    // {
                    //     GVT_DEBUG(DBG_LOW,"Seeding ray " << rc << ": " << this->rays[rc]);
                    //     gvt::render::actor::isecDomList len2List;
                    //     gvt::render::Attributes::rta->dataset->intersect(this->rays[rc], len2List);
                    //     // only keep rays that are meant for domains on this processor
                    //     if (!len2List.empty() && ((int) len2List[0] % mpi.world_size) == mpi.rank) 
                    //     {

                    //         this->rays[rc].domains.assign(len2List.rbegin(), len2List.rend());
                    //         gvt::render::Attributes::rta->dataset->getDomain(len2List[0])->marchIn(this->rays[rc]);
                    //         this->queue[len2List[0]].push_back(this->rays[rc]); // TODO: make this a ref?

                    //     }
                    // }
                }

                virtual void operator()() 
                {
                    long ray_counter = 0, domain_counter = 0;

                    FindNeighbors();

                    GVT_DEBUG(DBG_LOW,"generating camera rays");

                    this->FilterRaysLocally();


                    GVT_DEBUG(DBG_LOW,"tracing rays");


                    // process domains until all rays are terminated
                    bool all_done = false;
                    std::set<int> doms_to_send;
                    int lastDomain = -1;
                    gvt::render::data::domain::AbstractDomain* dom = NULL;
                    gvt::render::actor::RayVector moved_rays;
                    moved_rays.reserve(1000);

                    int domTarget = -1, domTargetCount = 0;

                    while (!all_done) 
                    {
                        if (!this->queue.empty()) 
                        {
                            // process domain assigned to this proc with most rays queued
                            domTarget = -1;
                            domTargetCount = 0;
                            for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) 
                            {
                                if ((q->first % mpi.world_size) == mpi.rank
                                    && q->second.size() > domTargetCount) 
                                {
                                    domTargetCount = q->second.size();
                                domTarget = q->first;
                            }
                        }
                        GVT_DEBUG(DBG_LOW,"selected domain " << domTarget << " (" << domTargetCount << " rays)");
                        GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) std::cerr << "selected domain " << domTarget << " (" << domTargetCount << " rays)" << std::endl);

                        doms_to_send.clear();
                            // pnav: use this to ignore domain x:        int domi=0;if (0)
                        if (domTarget >= 0) 
                        {
                            GVT_DEBUG(DBG_LOW,"Getting domain " << domTarget);
                            if (domTarget != lastDomain)
                                if (dom != NULL) dom->free();

                            dom = gvt::render::Attributes::rta->dataset->getDomain(domTarget);

                                // track domains loaded
                            if (domTarget != lastDomain) 
                            {
                                ++domain_counter;
                                lastDomain = domTarget;
                                dom->load();
                            }

                                //GVT::Backend::ProcessQueue<DomainType>(new GVT::Backend::adapt_param<DomainType>(this->queue, moved_rays, domTarget, dom, this->colorBuf, ray_counter, domain_counter))();
                            {
                                moved_rays.reserve(this->queue[domTarget].size()*10);
                                boost::timer::auto_cpu_timer t("Tracing domain rays %t\n");
                                dom->trace(this->queue[domTarget], moved_rays);
                            }
                            boost::atomic<int> current_ray(0);
                            size_t workload = std::max((size_t) 1, (size_t) (moved_rays.size() / (gvt::core::schedule::asyncExec::instance()->numThreads * 2)));
                            {
                                boost::timer::auto_cpu_timer t("Scheduling rays %t\n");
                                for (int rc = 0; rc < gvt::core::schedule::asyncExec::instance()->numThreads; ++rc) 
                                {
                                    gvt::core::schedule::asyncExec::instance()->run_task(processRayVector(this, moved_rays, current_ray, moved_rays.size(), workload, dom));
                                }
                                gvt::core::schedule::asyncExec::instance()->sync();
                            }

                            this->queue.erase(domTarget);
                        }
                        //DEBUG( else if (GVT_DEBUG_RANK) cerr << mpi.rank << ": no assigned domains have rays" << endl);
                    }
                    //DEBUG( else if (GVT_DEBUG_RANK) cerr << mpi.rank << ": skipped queues.  empty:" << this->queue.empty() << " size:" << this->queue.size() << endl);

                        // done with current domain, send off rays to their proper processors.


                    SendRays();
                        // are we done?

                        // root proc takes empty flag from all procs
                    int not_done = (int) (!this->queue.empty());
                    int *empties = (mpi.rank == 0) ? new int[mpi.world_size] : NULL;
                    MPI_Gather(&not_done, 1, MPI_INT, empties, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (mpi.rank == 0) 
                    {
                        not_done = 0;
                        for (int i = 0; i < mpi.world_size; ++i)
                            not_done += empties[i];
                        for (int i = 0; i < mpi.world_size; ++i)
                            empties[i] = not_done;
                    }

                    MPI_Scatter(empties, 1, MPI_INT, &not_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": " << not_done << " procs still have rays" << " (my q:" << this->queue.size() << ")" << endl);
                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK)
                        for (map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q)
                            cerr << "    q(" << q->first << "):" << q->second.size() << endl
                        );

                    all_done = (not_done == 0);

                    delete[] empties;

                }

                    // add colors to the framebuffer
                this->gatherFramebuffers(this->rays_end - this->rays_start);

            }

            virtual void FindNeighbors() {

                    int total = gvt::render::Attributes::rta->GetTopology()[2], 
                        plane = gvt::render::Attributes::rta->GetTopology()[1], 
                        row   = gvt::render::Attributes::rta->GetTopology()[0]; // XXX TODO: assumes grid layout
                    int offset[3] = {-1, 0, 1};
                    std::set<int> n_doms;

                    // find all domains that neighbor my domains
                    for (int i = 0; i < total; ++i) 
                    {
                        if (i % mpi.world_size != mpi.rank) continue;

                        // down, left
                        int n = i - 1;
                        if (n >= 0 && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i - row;
                        if (n >= 0 && (n % plane) < (i % plane))
                            n_doms.insert(n);
                        n = i - row - 1;
                        if (n >= 0 && (n % plane) < (i % plane) && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i - row + 1;
                        if (n >= 0 && (n % plane) < (i % plane) && (n % row) > (i % row))
                            n_doms.insert(n);

                        // up, right
                        n = i + 1;
                        if (n < total && (n % row) > (i % row))
                            n_doms.insert(n);
                        n = i + row;
                        if (n < total && (n % plane) > (i % plane))
                            n_doms.insert(n);
                        n = i + row - 1;
                        if (n < total && (n % plane) > (i % plane) && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i + row + 1;
                        if (n < total && (n % plane) > (i % plane) && (n % row) > (i % row))
                            n_doms.insert(n);

                        // bottom
                        n = i - plane;
                        if (n >= 0)
                            n_doms.insert(n);
                        // bottom: down, left
                        n = i - plane - 1;
                        if (n >= 0 && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i - plane - row;
                        if (n >= 0 && (n % plane) < (i % plane))
                            n_doms.insert(n);
                        n = i - plane - row - 1;
                        if (n >= 0 && (n % plane) < (i % plane) && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i - plane - row + 1;
                        if (n >= 0 && (n % plane) < (i % plane) && (n % row) > (i % row))
                            n_doms.insert(n);
                        // bottom: up, right
                        n = i - plane + 1;
                        if (n >= 0 && (n % row) > (i % row))
                            n_doms.insert(n);
                        n = i - plane + row;
                        if (n >= 0 && (n % plane) > (i % plane))
                            n_doms.insert(n);
                        n = i - plane + row - 1;
                        if (n >= 0 && (n % plane) > (i % plane) && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i - plane + row + 1;
                        if (n >= 0 && (n % plane) > (i % plane) && (n % row) > (i % row))
                            n_doms.insert(n);

                        // top
                        n = i + plane;
                        if (n < total)
                            n_doms.insert(n);
                        // down, left
                        n = i + plane - 1;
                        if (n < total && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i + plane - row;
                        if (n < total && (n % plane) < (i % plane))
                            n_doms.insert(n);
                        n = i + plane - row - 1;
                        if (n < total && (n % plane) < (i % plane) && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i + plane - row + 1;
                        if (n < total && (n % plane) < (i % plane) && (n % row) > (i % row))
                            n_doms.insert(n);
                        // up, right
                        n = i + plane + 1;
                        if (n < total && (n % row) > (i % row))
                            n_doms.insert(n);
                        n = i + plane + row;
                        if (n < total && (n % plane) > (i % plane))
                            n_doms.insert(n);
                        n = i + plane + row - 1;
                        if (n < total && (n % plane) > (i % plane) && (n % row) < (i % row))
                            n_doms.insert(n);
                        n = i + plane + row + 1;
                        if (n < total && (n % plane) > (i % plane) && (n % row) > (i % row))
                            n_doms.insert(n);
                    }

                    // find which proc owns each neighboring domain
                    for (std::set<int>::iterator it = n_doms.begin(); it != n_doms.end(); ++it)
                        if (*it % mpi.world_size != mpi.rank) neighbors.insert(*it % mpi.world_size);


                }

                virtual bool SendRays() 
                {

                    int* outbound = new int[2 * mpi.world_size];
                    int* inbound = new int[2 * mpi.world_size];
                    MPI_Request* reqs = new MPI_Request[2 * mpi.world_size];
                    MPI_Status* stat = new MPI_Status[2 * mpi.world_size];
                    unsigned char** send_buf = new unsigned char*[mpi.world_size];
                    unsigned char** recv_buf = new unsigned char*[mpi.world_size];
                    int* send_buf_ptr = new int[mpi.world_size];

                    // init bufs
                    for (int i = 0; i < 2 * mpi.world_size; ++i) 
                    {
                        inbound[i] = outbound[i] = 0;
                        reqs[i] = MPI_REQUEST_NULL;
                    }

                    // count how many rays are to be sent to each neighbor
                    for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) 
                    {
                        int n = q->first % mpi.world_size;
                        GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": domain " << q->first << " maps to proc " << n);
                        if (this->neighbors.find(n) != this->neighbors.end()) 
                        {
                            int n_ptr = 2 * n;
                            int buf_size = 0;

                            outbound[n_ptr] += q->second.size();
                            for (int r = 0; r < q->second.size(); ++r) 
                            {
                                GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ":  " << (q->second)[r] << endl);
                                buf_size += (q->second)[r].packedSize(); // rays can have diff packed sizes
                            }
                            outbound[n_ptr + 1] += buf_size;
                            GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << " neighbor! Added " << q->second.size() << " rays (" << buf_size << " bytes)" << endl);
                        }
                        //DEBUG( else if (GVT_DEBUG_RANK) cerr << " not neighbor" << endl);
                    }

                    // let the neighbors know what's coming
                    // and find out what's coming here
                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": sending neighbor info" << endl);

                    int tag = 0;
                    for (std::set<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n)
                        MPI_Irecv(&inbound[2 * (*n)], 2, MPI_INT, *n, tag, MPI_COMM_WORLD, &reqs[2 * (*n)]);
                    for (std::set<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n)
                        MPI_Isend(&outbound[2 * (*n)], 2, MPI_INT, *n, tag, MPI_COMM_WORLD, &reqs[2 * (*n) + 1]);

                    MPI_Waitall(2 * mpi.world_size, reqs, stat);
                    GVT_DEBUG_CODE(DBG_LOW,
                        if (GVT_DEBUG_RANK) {
                        cerr << mpi.rank << ": sent neighbor info" << endl;
                        cerr << mpi.rank << ": inbound ";
                        for (int i = 0; i < mpi.world_size; ++i)
                            cerr << "(" << inbound[2 * i] << "," << inbound[2 * i + 1] << ") ";
                        cerr << endl << mpi.rank << ": outbound ";
                        for (int i = 0; i < mpi.world_size; ++i)
                            cerr << "(" << outbound[2 * i] << "," << outbound[2 * i + 1] << ") ";
                        cerr << endl;
                    });

                    // set up send and recv buffers
                    for (int i = 0, j = 0; i < mpi.world_size; ++i, j += 2) 
                    {
                        send_buf_ptr[i] = 0;
                        if (outbound[j] > 0)
                            send_buf[i] = new unsigned char[outbound[j + 1]];
                        else send_buf[i] = 0;
                        if (inbound[j] > 0)
                            recv_buf[i] = new unsigned char[inbound[j + 1]];
                        else recv_buf[i] = 0;
                    }
                    for (int i = 0; i < 2 * mpi.world_size; ++i) reqs[i] = MPI_REQUEST_NULL;

                    // now send and receive rays (and associated color buffers)
                        tag = tag + 1;
                    for (std::set<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n) 
                    {
                        if (inbound[2 * (*n)] > 0) 
                        {
                            GVT_DEBUG_CODE(DBG_LOW,
                                if (GVT_DEBUG_RANK)
                                cerr << mpi.rank << ": recv " << inbound[2 * (*n)] << " rays ("
                                    << inbound[2 * (*n) + 1] << " bytes) from " << *n << endl
                                );
                            MPI_Irecv(recv_buf[*n], inbound[2 * (*n) + 1], MPI_UNSIGNED_CHAR, *n, tag, MPI_COMM_WORLD, &reqs[2 * (*n)]);
                            GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": recv done from " << *n << endl);
                        }
                    }

                    std::vector<int> to_del;
                    for (std::map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) 
                    {
                        int n = q->first % mpi.world_size;
                        if (outbound[2 * n] > 0) {
                            GVT_DEBUG_CODE(DBG_LOW,
                                if (GVT_DEBUG_RANK)
                                cerr << mpi.rank << ": send " << outbound[2 * n] << " rays ("
                                    << outbound[2 * n + 1] << " bytes) to " << n << endl
                            );
                            for (int r = 0; r < q->second.size(); ++r) {
                                gvt::render::actor::Ray ray = (q->second)[r];
                                GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ":  " << ray << endl);
                                send_buf_ptr[n] += ray.pack(send_buf[n] + send_buf_ptr[n]);
                            }
                            to_del.push_back(q->first);
                        }
                    }
                    for (std::set<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n) {
                        if (outbound[2 * (*n)] > 0) {
                            MPI_Isend(send_buf[*n], outbound[2 * (*n) + 1], MPI_UNSIGNED_CHAR, *n, tag, MPI_COMM_WORLD, &reqs[2 * (*n) + 1]);
                            GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": send done to " << *n << endl);
                        }
                    }

                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": q(" << this->queue.size() << ") erasing " << to_del.size());
                    for (int i = 0; i < to_del.size(); ++i)
                        this->queue.erase(to_del[i]);
                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << " q(" << this->queue.size() << ")" << endl);

                    MPI_Waitall(2 * mpi.world_size, reqs, stat); // XXX TODO refactor to use Waitany?


                    for (std::set<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n) 
                    {
                        if (inbound[2 * (*n)] > 0) 
                        {
                            GVT_DEBUG_CODE(DBG_LOW,
                                if (GVT_DEBUG_RANK) {
                                cerr << mpi.rank << ": adding " << inbound[2 * (*n)] << " rays (" << inbound[2 * (*n) + 1] << " B) from " << *n << endl;
                                cerr << "    recv buf: " << (long) recv_buf[*n] << endl;
                            });
                            int ptr = 0;
                            for (int c = 0; c < inbound[2 * (*n)]; ++c) 
                            {
                                gvt::render::actor::Ray r(recv_buf[*n] + ptr);
                                GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ":  " << r << endl);
                                this->queue[r.domains.back()].push_back(r);
                                ptr += r.packedSize();
                            }
                        }
                    }
                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << mpi.rank << ": sent and received rays" << endl);

                    // clean up
                    for (int i = 0; i < mpi.world_size; ++i) {
                        delete[] send_buf[i];
                        delete[] recv_buf[i];
                    }
                    delete[] send_buf_ptr;
                    delete[] send_buf;
                    delete[] recv_buf;
                    delete[] inbound;
                    delete[] outbound;
                    delete[] reqs;
                    delete[] stat;
                    GVT_DEBUG_CODE(DBG_LOW,if (GVT_DEBUG_RANK) cerr << "done with DomainSendRays" << endl);
                    return false;
                }
            };
        }
    }
}
#endif /* GVT_RENDER_ALGORITHM_DOMAIN_TRACER_H */
