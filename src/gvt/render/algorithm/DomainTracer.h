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
 * DomainTracer.h
 *
 *  Created on: Dec 8, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_RENDER_ALGORITHM_DOMAIN_TRACER_H
#define GVT_RENDER_ALGORITHM_DOMAIN_TRACER_H

#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/Types.h>
#ifdef GVT_USE_MPE
#include "mpe.h"
#endif
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/Types.h>
#include <gvt/render/algorithm/TracerBase.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/RenderContext.h>

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

#include <boost/foreach.hpp>

#include <set>

#define RAY_BUF_SIZE 10485760 // 10 MB per neighbor

using namespace gvt::core::mpi;
namespace gvt {
namespace render {
namespace algorithm {

/// work scheduler that strives to keep domains loaded and send rays
/**
  The Domain scheduler strives to schedule work such that loaded domains remain loaded
  and rays are sent to the process that contains the loaded domain (or is responsible
  for loading the domain). A domain is loaded in at most one process at any time. If
  there are sufficent processes to load all domains, the entire render will proceed
  in-core.

  This scheduler can become unbalanced when:
   - there are more processes than domains, excess processes will remain idle
   - rays are concentrated at a few domains, processes with other domains loaded
  can remain idle
   - when there are few rays remaining to render, other processes can remain
  idle

     \sa HybridTracer, ImageTracer
   */
template <> class Tracer<gvt::render::schedule::DomainScheduler> : public AbstractTrace {
public:
  std::set<int> neighbors;

  size_t rays_start, rays_end;

  // caches meshes that are converted into the adapter's format
  std::map<gvt::core::Uuid, gvt::render::Adapter *> adapterCache;
  gvt::core::Vector<gvt::core::DBNodeH> dataNodes;
  std::map<gvt::core::Uuid, size_t> mpiInstanceMap;
#ifdef GVT_USE_MPE
  int tracestart, traceend;
  int shufflestart, shuffleend;
  int framebufferstart, framebufferend;
  int localrayfilterstart, localrayfilterend;
  int intersectbvhstart, intersectbvhend;
  int marchinstart, marchinend;
#endif

  Tracer(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image) : AbstractTrace(rays, image) {
#ifdef GVT_USE_MPE
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPE_Log_get_state_eventIDs(&tracestart, &traceend);
    MPE_Log_get_state_eventIDs(&shufflestart, &shuffleend);
    MPE_Log_get_state_eventIDs(&framebufferstart, &framebufferend);
    MPE_Log_get_state_eventIDs(&localrayfilterstart, &localrayfilterend);
    MPE_Log_get_state_eventIDs(&intersectbvhstart, &intersectbvhend);
    MPE_Log_get_state_eventIDs(&marchinstart, &marchinend);
    if (mpi.rank == 0) {
      MPE_Describe_state(tracestart, traceend, "Process Queue", "blue");
      MPE_Describe_state(shufflestart, shuffleend, "Shuffle Rays", "green");
      MPE_Describe_state(framebufferstart, framebufferend, "Gather Framebuffer", "orange");
      MPE_Describe_state(localrayfilterstart, localrayfilterend, "Filter Rays Local", "coral");
      MPE_Describe_state(intersectbvhstart, intersectbvhend, "Intersect BVH", "azure");
      MPE_Describe_state(marchinstart, marchinend, "March Ray in", "LimeGreen");
    }
#endif
    dataNodes = rootnode["Data"].getChildren();

    // create a map of instances to mpi rank
    for (size_t i = 0; i < instancenodes.size(); i++) {
      gvt::core::DBNodeH meshNode = instancenodes[i]["meshRef"].deRef();

      size_t dataIdx = -1;
      for (size_t d = 0; d < dataNodes.size(); d++) {
        if (dataNodes[d].UUID() == meshNode.UUID()) {
          dataIdx = d;
          break;
        }
      }

      // NOTE: mpi-data(domain) assignment strategy
      size_t mpiNode = dataIdx % mpi.world_size;

      GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] domain scheduler: instId: " << i << ", dataIdx: " << dataIdx
                                << ", target mpi node: " << mpiNode << ", world size: " << mpi.world_size);

      GVT_ASSERT(dataIdx != -1, "domain scheduler: could not find data node");
      mpiInstanceMap[instancenodes[i].UUID()] = mpiNode;
    }
  }

  virtual ~Tracer() {}

  virtual void FilterRaysLocally() {
    auto nullNode = gvt::core::DBNodeH(); // temporary workaround until
                                          // shuffleRays is fully replaced
    shuffleRays(rays, nullNode);

    for (auto e : queue) {
      if (mpiInstanceMap[instancenodes[e.first].UUID()] != mpi.rank) {
        GVT_DEBUG(DBG_ALWAYS, " rank[" << mpi.rank << "] FILTERRAYS: removing queue " << e.first);
        queue[e.first].clear();
      }
    }
  }

  virtual void operator()() {
    boost::timer::cpu_timer t_sched;
    t_sched.start();
    boost::timer::cpu_timer t_trace;
    GVT_DEBUG(DBG_ALWAYS, "domain scheduler: starting, num rays: " << rays.size());
    gvt::core::DBNodeH root = gvt::render::RenderContext::instance()->getRootNode();

    int adapterType = root["Schedule"]["adapter"].value().toInteger();

    long domain_counter = 0;

// FindNeighbors();

// sort rays into queues
// note: right now throws away rays that do not hit any domain owned by the current
// rank
#ifdef GVT_USE_MPE
    MPE_Log_event(localrayfilterstart, 0, NULL);
#endif
    FilterRaysLocally();
#ifdef GVT_USE_MPE
    MPE_Log_event(localrayfilterend, 0, NULL);
#endif

    GVT_DEBUG(DBG_LOW, "tracing rays");

    // process domains until all rays are terminated
    bool all_done = false;
    std::set<int> doms_to_send;
    int lastInstance = -1;
    // gvt::render::data::domain::AbstractDomain* dom = NULL;

    gvt::render::actor::RayVector moved_rays;
    moved_rays.reserve(1000);

    int instTarget = -1;
    size_t instTargetCount = 0;

    gvt::render::Adapter *adapter = 0;

    while (!all_done) {

      if (!queue.empty()) {
        // process domain assigned to this proc with most rays queued
        // if there are queues for instances that are not assigned
        // to the current rank, erase those entries
        instTarget = -1;
        instTargetCount = 0;

        std::vector<int> to_del;
        GVT_DEBUG(DBG_ALWAYS, "domain scheduler: selecting next instance, num queues: " << this->queue.size());
        for (auto &q : queue) {
          const bool inRank = mpiInstanceMap[instancenodes[q.first].UUID()] == mpi.rank;

          if (q.second.empty() || !inRank) {
            to_del.push_back(q.first);
            continue;
          }

          if (inRank && q.second.size() > instTargetCount) {
            instTargetCount = q.second.size();
            instTarget = q.first;
          }
        }

        // erase empty queues
        for (int instId : to_del) {
          GVT_DEBUG(DBG_ALWAYS, "rank[" << mpi.rank << "] DOMAINTRACER: deleting queue for instance " << instId);
          queue.erase(instId);
        }

        if (instTarget == -1) {
          continue;
        }

        GVT_DEBUG(DBG_ALWAYS, "domain scheduler: next instance: " << instTarget << ", rays: " << instTargetCount << " ["
                                                                  << mpi.rank << "]");

        doms_to_send.clear();
        // pnav: use this to ignore domain x:        int domi=0;if (0)
        if (instTarget >= 0) {
          GVT_DEBUG(DBG_LOW, "Getting instance " << instTarget);
          // gvt::render::Adapter *adapter = 0;
          gvt::core::DBNodeH meshNode = instancenodes[instTarget]["meshRef"].deRef();

          if (instTarget != lastInstance) {
            // TODO: before we would free the previous domain before loading the
            // next
            // this can be replicated by deleting the adapter
            delete adapter;
            adapter = 0;
          }

          // track domains loaded
          if (instTarget != lastInstance) {
            ++domain_counter;
            lastInstance = instTarget;

            //
            // 'getAdapterFromCache' functionality
            if (!adapter) {
              GVT_DEBUG(DBG_ALWAYS, "domain scheduler: creating new adapter");
              switch (adapterType) {
#ifdef GVT_RENDER_ADAPTER_EMBREE
              case gvt::render::adapter::Embree:
                adapter = new gvt::render::adapter::embree::data::EmbreeMeshAdapter(meshNode);
                break;
#endif
#ifdef GVT_RENDER_ADAPTER_MANTA
              case gvt::render::adapter::Manta:
                adapter = new gvt::render::adapter::manta::data::MantaMeshAdapter(meshNode);
                break;
#endif
#ifdef GVT_RENDER_ADAPTER_OPTIX
              case gvt::render::adapter::Optix:
                adapter = new gvt::render::adapter::optix::data::OptixMeshAdapter(meshNode);
                break;
#endif
#if defined(GVT_RENDER_ADAPTER_OPTIX) && defined(GVT_RENDER_ADAPTER_EMBREE)
              case gvt::render::adapter::Heterogeneous:
                adapter = new gvt::render::adapter::heterogeneous::data::HeterogeneousMeshAdapter(meshNode);
                break;
#endif
              default:
                GVT_DEBUG(DBG_SEVERE, "domain scheduler: unknown adapter type: " << adapterType);
              }
              // adapterCache[meshNode.UUID()] = adapter; // note: cache logic
              // comes later when we implement hybrid
            }
            // end 'getAdapterFromCache' concept
            //
          }
          GVT_ASSERT(adapter != nullptr, "domain scheduler: adapter not set");

          GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] domain scheduler: calling process queue");
          {
            t_trace.resume();
            moved_rays.reserve(this->queue[instTarget].size() * 10);
#ifdef GVT_USE_DEBUG
            boost::timer::auto_cpu_timer t("Tracing rays in adapter: %w\n");
#endif
#ifdef GVT_USE_MPE
            MPE_Log_event(tracestart, 0, NULL);
#endif
            adapter->trace(this->queue[instTarget], moved_rays, instancenodes[instTarget]);
#ifdef GVT_USE_MPE
            MPE_Log_event(traceend, 0, NULL);
#endif
            this->queue[instTarget].clear();
            t_trace.stop();
          }

#ifdef GVT_USE_MPE
          MPE_Log_event(shufflestart, 0, NULL);
#endif
          shuffleRays(moved_rays, instancenodes[instTarget]);
#ifdef GVT_USE_MPE
          MPE_Log_event(shuffleend, 0, NULL);
#endif
          moved_rays.clear();
        }
      }

#if GVT_USE_DEBUG
      if (!queue.empty()) {
        std::cout << "[" << mpi.rank << "] Queue is not empty" << std::endl;
        for (auto q : queue) {
          std::cout << "[" << mpi.rank << "] [" << q.first << "] : " << q.second.size() << std::endl;
        }
      }
#endif

      // done with current domain, send off rays to their proper processors.
      GVT_DEBUG(DBG_ALWAYS, "Rank [ " << mpi.rank << "]  calling SendRays");
      SendRays();
      // are we done?

      // root proc takes empty flag from all procs
      int not_done = (int)(!queue.empty());
      int *empties = (mpi.rank == 0) ? new int[mpi.world_size] : NULL;
      MPI_Gather(&not_done, 1, MPI_INT, empties, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (mpi.rank == 0) {
        not_done = 0;
        for (size_t i = 0; i < mpi.world_size; ++i) not_done += empties[i];
        for (size_t i = 0; i < mpi.world_size; ++i) empties[i] = not_done;
      }

      MPI_Scatter(empties, 1, MPI_INT, &not_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
      GVT_DEBUG_CODE(DBG_ALWAYS, if (DEBUG_RANK) cerr << mpi.rank << ": " << not_done << " procs still have rays"
                                                      << " (my q:" << queue.size() << ")");
      all_done = (not_done == 0);

      delete[] empties;
    }

// add colors to the framebuffer
#ifdef GVT_USE_MPE
    MPE_Log_event(framebufferstart, 0, NULL);
#endif
    this->gatherFramebuffers(this->rays_end - this->rays_start);
#ifdef GVT_USE_MPE
    MPE_Log_event(framebufferend, 0, NULL);
#endif
  }

  // FIXME: update FindNeighbors to use mpiInstanceMap
  virtual void FindNeighbors() {
    gvt::core::math::Vector3f topo;
    topo = rootnode["Dataset"]["topology"].value().toVector3f();
    int total = topo[2], plane = topo[1], row = topo[0]; // XXX TODO:
    // int total = gvt::render::Attributes::rta->GetTopology()[2],
    //   plane = gvt::render::Attributes::rta->GetTopology()[1],
    //  row = gvt::render::Attributes::rta->GetTopology()[0];  // XXX TODO:
    // assumes grid
    // layout
    int offset[3] = { -1, 0, 1 };
    std::set<int> n_doms;

    // find all domains that neighbor my domains
    for (int i = 0; i < total; ++i) {
      if (i % mpi.world_size != mpi.rank) continue;

      // down, left
      int n = i - 1;
      if (n >= 0 && (n % row) < (i % row)) n_doms.insert(n);
      n = i - row;
      if (n >= 0 && (n % plane) < (i % plane)) n_doms.insert(n);
      n = i - row - 1;
      if (n >= 0 && (n % plane) < (i % plane) && (n % row) < (i % row)) n_doms.insert(n);
      n = i - row + 1;
      if (n >= 0 && (n % plane) < (i % plane) && (n % row) > (i % row)) n_doms.insert(n);

      // up, right
      n = i + 1;
      if (n < total && (n % row) > (i % row)) n_doms.insert(n);
      n = i + row;
      if (n < total && (n % plane) > (i % plane)) n_doms.insert(n);
      n = i + row - 1;
      if (n < total && (n % plane) > (i % plane) && (n % row) < (i % row)) n_doms.insert(n);
      n = i + row + 1;
      if (n < total && (n % plane) > (i % plane) && (n % row) > (i % row)) n_doms.insert(n);

      // bottom
      n = i - plane;
      if (n >= 0) n_doms.insert(n);
      // bottom: down, left
      n = i - plane - 1;
      if (n >= 0 && (n % row) < (i % row)) n_doms.insert(n);
      n = i - plane - row;
      if (n >= 0 && (n % plane) < (i % plane)) n_doms.insert(n);
      n = i - plane - row - 1;
      if (n >= 0 && (n % plane) < (i % plane) && (n % row) < (i % row)) n_doms.insert(n);
      n = i - plane - row + 1;
      if (n >= 0 && (n % plane) < (i % plane) && (n % row) > (i % row)) n_doms.insert(n);
      // bottom: up, right
      n = i - plane + 1;
      if (n >= 0 && (n % row) > (i % row)) n_doms.insert(n);
      n = i - plane + row;
      if (n >= 0 && (n % plane) > (i % plane)) n_doms.insert(n);
      n = i - plane + row - 1;
      if (n >= 0 && (n % plane) > (i % plane) && (n % row) < (i % row)) n_doms.insert(n);
      n = i - plane + row + 1;
      if (n >= 0 && (n % plane) > (i % plane) && (n % row) > (i % row)) n_doms.insert(n);

      // top
      n = i + plane;
      if (n < total) n_doms.insert(n);
      // down, left
      n = i + plane - 1;
      if (n < total && (n % row) < (i % row)) n_doms.insert(n);
      n = i + plane - row;
      if (n < total && (n % plane) < (i % plane)) n_doms.insert(n);
      n = i + plane - row - 1;
      if (n < total && (n % plane) < (i % plane) && (n % row) < (i % row)) n_doms.insert(n);
      n = i + plane - row + 1;
      if (n < total && (n % plane) < (i % plane) && (n % row) > (i % row)) n_doms.insert(n);
      // up, right
      n = i + plane + 1;
      if (n < total && (n % row) > (i % row)) n_doms.insert(n);
      n = i + plane + row;
      if (n < total && (n % plane) > (i % plane)) n_doms.insert(n);
      n = i + plane + row - 1;
      if (n < total && (n % plane) > (i % plane) && (n % row) < (i % row)) n_doms.insert(n);
      n = i + plane + row + 1;
      if (n < total && (n % plane) > (i % plane) && (n % row) > (i % row)) n_doms.insert(n);
    }

    // find which proc owns each neighboring domain
    for (std::set<int>::iterator it = n_doms.begin(); it != n_doms.end(); ++it)
      if (*it % mpi.world_size != mpi.rank) neighbors.insert(*it % mpi.world_size);
  }
  virtual bool SendRays() {
    int *outbound = new int[2 * mpi.world_size];
    int *inbound = new int[2 * mpi.world_size];
    MPI_Request *reqs = new MPI_Request[2 * mpi.world_size];
    MPI_Status *stat = new MPI_Status[2 * mpi.world_size];
    unsigned char **send_buf = new unsigned char *[mpi.world_size];
    unsigned char **recv_buf = new unsigned char *[mpi.world_size];
    int *send_buf_ptr = new int[mpi.world_size];

    // if there is only one rank we dont need to go through this routine.
    if (mpi.world_size < 2) return false;
    // init bufs
    for (size_t i = 0; i < 2 * mpi.world_size; ++i) {
      inbound[i] = outbound[i] = 0;
      reqs[i] = MPI_REQUEST_NULL;
    }

    // count how many rays are to be sent to each neighbor
    for (std::map<int, gvt::render::actor::RayVector>::iterator q = queue.begin(); q != queue.end(); ++q) {
      // n is the rank this vector of rays (q.second) belongs on.
      size_t n = mpiInstanceMap[instancenodes[q->first].UUID()]; // bds
      GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: instance " << q->first << " maps to proc " << n);
      if (n != mpi.rank) { // bds if instance n is not this rank send rays to it.
        int n_ptr = 2 * n;
        int buf_size = 0;

        outbound[n_ptr] += q->second.size(); // outbound[n_ptr] has number of rays going
        for (size_t r = 0; r < q->second.size(); ++r) {
          buf_size += (q->second)[r].packedSize(); // rays can have diff packed sizes
        }
        outbound[n_ptr + 1] += buf_size;    // size of buffer needed to hold rays
        outbound[n_ptr + 1] += sizeof(int); // bds add space for the queue number
        outbound[n_ptr + 1] += sizeof(int); // bds add space for the number of rays in queue
        GVT_DEBUG(DBG_ALWAYS, " neighbor! Added " << q->second.size() << " rays (" << buf_size << " bytes)"
                                                  << std::endl);
      }
    }

    // let the neighbors know what's coming
    // and find out what's coming here
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: sending neighbor info" << std::endl);

    int tag = 0;
    for (size_t n = 0; n < mpi.world_size; ++n) // bds sends to self?
      MPI_Irecv(&inbound[2 * n], 2, MPI_INT, n, tag, MPI_COMM_WORLD, &reqs[2 * n]);
    for (size_t n = 0; n < mpi.world_size; ++n) // bds send to self
      MPI_Isend(&outbound[2 * n], 2, MPI_INT, n, tag, MPI_COMM_WORLD, &reqs[2 * n + 1]);

    MPI_Waitall(2 * mpi.world_size, reqs, stat);
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]:GOT HEADS UP " << std::endl);
#ifdef GVT_DEBUG
    std::cerr << mpi.rank << ": sent neighbor info" << std::endl;
    std::cerr << mpi.rank << ": inbound ";
    for (size_t i = 0; i < mpi.world_size; ++i) std::cerr << "(" << inbound[2 * i] << "," << inbound[2 * i + 1] << ") ";
    std::cerr << std::endl << mpi.rank << ": outbound ";
    for (size_t i = 0; i < mpi.world_size; ++i)

      std::cerr << "(" << outbound[2 * i] << "," << outbound[2 * i + 1] << ") ";
    std::cerr << std::endl;
#endif

    // set up send and recv buffers
    for (size_t i = 0, j = 0; i < mpi.world_size; ++i, j += 2) {
      send_buf_ptr[i] = 0;
      if (outbound[j] > 0)
        send_buf[i] = new unsigned char[outbound[j + 1]];
      else
        send_buf[i] = 0;
      if (inbound[j] > 0)
        recv_buf[i] = new unsigned char[inbound[j + 1]];
      else
        recv_buf[i] = 0;
    }
    for (size_t i = 0; i < 2 * mpi.world_size; ++i) reqs[i] = MPI_REQUEST_NULL;

    //  ************************ post non-blocking receive *********************
    tag = tag + 1;
    for (size_t n = 0; n < mpi.world_size; ++n) { // bds loop through all ranks
      if (inbound[2 * n] > 0) {
        GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: recv " << inbound[2 * n] << " rays (" << inbound[2 * n + 1]
                                  << " bytes) from " << n << std::endl);
        MPI_Irecv(recv_buf[n], inbound[2 * n + 1], MPI_UNSIGNED_CHAR, n, tag, MPI_COMM_WORLD, &reqs[2 * n]);
        GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << ": recv posted from " << n << std::endl);
      }
    }
    // ******************** pack the send buffers *********************************
    std::vector<int> to_del;
    for (std::map<int, gvt::render::actor::RayVector>::iterator q = queue.begin(); q != queue.end(); ++q) {
      int n = mpiInstanceMap[instancenodes[q->first].UUID()]; // bds use instance map
      if (outbound[2 * n] > 0) {
        *((int *)(send_buf[n] + send_buf_ptr[n])) = q->first;         // bds load queue number into send buffer
        send_buf_ptr[n] += sizeof(int);                               // bds advance pointer
        *((int *)(send_buf[n] + send_buf_ptr[n])) = q->second.size(); // bds load number of rays into send buffer
        send_buf_ptr[n] += sizeof(int);                               // bds advance pointer
        GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: loading queue " << q->first << std::endl);
        for (size_t r = 0; r < q->second.size(); ++r) { // load the rays in this queue
          gvt::render::actor::Ray ray = (q->second)[r];
          send_buf_ptr[n] += ray.pack(send_buf[n] + send_buf_ptr[n]);
        }
        to_del.push_back(q->first);
      }
    }
    for (size_t n = 0; n < mpi.world_size; ++n) { // bds loop over all
      if (outbound[2 * n] > 0) {
        MPI_Isend(send_buf[n], outbound[2 * n + 1], MPI_UNSIGNED_CHAR, n, tag, MPI_COMM_WORLD, &reqs[2 * n + 1]);
        GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: send done to " << n << std::endl);
      }
    }

    // GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << ": q(" << queue.size()
    //                            << ") erasing " << to_del.size());
    // for (int i = 0; i < to_del.size(); ++i) queue.erase(to_del[i]);
    // GVT_DEBUG(DBG_ALWAYS,  " q(" << queue.size() << ")" << std::endl);

    MPI_Waitall(2 * mpi.world_size, reqs, stat); // XXX TODO refactor to use Waitany?

    // ******************* unpack rays into the queues **************************
    for (int n = 0; n < mpi.world_size; ++n) { // bds loop over all
      GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] " << n << " inbound[2*n] " << inbound[2 * n] << std::endl);
      if (inbound[2 * n] > 0) {
        // clang-format off
        GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: adding " << inbound[2 * n] << " rays (" << inbound[2 * n + 1]
                                  << " B) from " << n << std::endl << "    recv buf: " << (long)recv_buf[n]
                                  << std::endl);
        // clang-format on
        int ptr = 0;
        while (ptr < inbound[2 * n + 1]) {
          int q_number = *((int *)(recv_buf[n] + ptr)); // bds get queue number
          ptr += sizeof(int);
          int raysinqueue = *((int *)(recv_buf[n] + ptr)); // bds get rays in this queue
          ptr += sizeof(int);
          GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: unpacking queue " << q_number << std::endl);
          for (int c = 0; c < raysinqueue; ++c) {
            gvt::render::actor::Ray r(recv_buf[n] + ptr);
            queue[q_number].push_back(r);
            ptr += r.packedSize();
          }
        }
      }
    }
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "]: sent and received rays" << std::endl);

    // clean up
    for (size_t i = 0; i < mpi.world_size; ++i) {
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
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] done with DomainSendRays");
    return false;
  }
};
}
}
}
#endif /* GVT_RENDER_ALGORITHM_DOMAIN_TRACER_H */
