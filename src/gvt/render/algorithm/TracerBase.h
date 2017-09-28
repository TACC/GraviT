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
 * Tracer.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_RENDER_ALGORITHM_TRACER_BASE_H
#define GVT_RENDER_ALGORITHM_TRACER_BASE_H

#include <gvt/core/Debug.h>
#include <gvt/core/utils/timer.h>
#include <gvt/render/Adapter.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/accel/BVH.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/actor/ORays.h>

#include <gvt/render/composite/composite.h>

#include <boost/foreach.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/timer/timer.hpp>

#include <algorithm>
#include <numeric>

#include <mpi.h>

#include <deque>
#include <map>

// uncomment this to restrict tbb to serial operation
// must also add "serial" to tbb::parallel_for
// like this tbb::serial::parallel_for. Do this to get reasonable 
// prints out of parallel for loop. Of course it doesent run in parallel
// any more. 
#define TBB_PREVIEW_SERIAL_SUBSET 1
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>

namespace gvt {
namespace render {
namespace algorithm {

/// Tracer base class

struct GVT_COMM {
  size_t rank;
  size_t world_size;

  GVT_COMM() {
    rank = MPI::COMM_WORLD.Get_rank();
    world_size = MPI::COMM_WORLD.Get_size();
  }

  operator bool() { return (world_size > 1); }
  bool root() { return rank == 0; }

  template <typename B> B *gatherbuffer(B *buf, size_t size) {

    if (world_size <= 1) return buf;

    // std::cout << "World size : " << world_size << std::endl;

    size_t partition_size = size / world_size;
    size_t next_neighbor = rank;
    size_t prev_neighbor = rank;

    B *acc = &buf[rank * partition_size];
    B gather[] = new B[partition_size * world_size];
    gvt::core::Vector<MPI::Request> Irecv_requests_status;

    for (int round = 0; round < world_size; round++) {
      next_neighbor = (next_neighbor + 1) % world_size;

      if (next_neighbor != rank) {
        // std::cout << "Node[" << rank << "] send to Node[" << next_neighbor << "]" << std::endl;
        B *send = &buf[next_neighbor * partition_size];
        MPI::COMM_WORLD.Isend(send, sizeof(B) * partition_size, MPI::BYTE, next_neighbor, rank | 0xF00000000000000);
      }

      prev_neighbor = (prev_neighbor > 0 ? prev_neighbor - 1 : world_size - 1);

      if (prev_neighbor != rank) {
        // std::cout << "Node[" << rank << "] recv to Node[" << prev_neighbor << "]" << std::endl;
        B *recv = &gather[prev_neighbor * partition_size];
        Irecv_requests_status.push_back(MPI::COMM_WORLD.Irecv(recv, sizeof(B) * partition_size, MPI::BYTE,
                                                              prev_neighbor, prev_neighbor | 0xF00000000000000));
      }
    }

    MPI::Request::Waitall(Irecv_requests_status.size(), &Irecv_requests_status[0]);

    for (int source = 0; source < world_size; ++source) {
      if (source == rank) continue;
      size_t chunksize =
          MAX(GVT_SIMD_WIDTH,
              partition_size / (gvt::core::CoreContext::instance()->getRootNode()["threads"].value().toInteger()));

      static tbb::simple_partitioner ap;
      tbb::parallel_for(tbb::blocked_range<size_t>(0, partition_size, chunksize),
                        [&](tbb::blocked_range<size_t> chunk) {
#ifndef __clang__
#pragma simd
#endif
                          for (int i = chunk.begin(); i < chunk.end(); ++i) {
                            acc[i] += gather[source * partition_size + i];
                          }
                        },
                        ap);
    }

    B *newbuf = (rank == 0) ? new B[size] : nullptr;
    MPI::COMM_WORLD.Gather(acc, sizeof(B) * partition_size, MPI::BYTE, newbuf, sizeof(B) * partition_size, MPI::BYTE,
                           0);

    if (newbuf) std::memcpy(buf, newbuf, sizeof(B) * size);
    delete[] gather;
    delete newbuf;
    return newbuf;
  }
};

/// base tracer class for GraviT ray tracing framework
/**
  This is the base class for the GraviT ray tracing framework on which the work
  schedulers are implemented.
  \sa DomainTracer, HybridTracer, ImageTracer
  */
class AbstractTrace {
public:
  ///! Define mpi communication world
  GVT_COMM mpi;

  gvt::render::actor::RayVector &rays;    ///< Rays to trace
  gvt::render::data::scene::Image &image; ///< Final image buffer
  gvt::render::RenderContext &cntxt = *gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootnode;
  gvt::core::Vector<gvt::core::DBNodeH> instancenodes;
  gvt::core::Map<int, gvt::render::data::primitives::Mesh *> meshRef;
  gvt::core::Map<int, glm::mat4 *> instM;
  gvt::core::Map<int, glm::mat4 *> instMinv;
  gvt::core::Map<int, glm::mat3 *> instMinvN;
  gvt::core::Vector<gvt::render::data::scene::Light *> lights;

  gvt::render::data::accel::AbstractAccel *acceleration;

  int width;
  int height;

  float sample_ratio;

  tbb::mutex *queue_mutex; // array of mutexes - one per instance
  gvt::core::Map<int, gvt::render::actor::RayVector> queue; ///< Node rays working
  tbb::mutex *colorBuf_mutex;                               ///< buffer for color accumulation
  glm::vec4 *colorBuf;

  gvt::render::composite::composite img;
  bool require_composite;

  AbstractTrace(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image)
      : rays(rays), image(image) {

    rootnode = cntxt.getRootNode();

    width = rootnode["Film"]["width"].value().toInteger();
    height = rootnode["Film"]["height"].value().toInteger();

    sample_ratio = 1.f;

    require_composite = false;
    colorBuf = new glm::vec4[width * height];
    require_composite = img.initIceT();

    // TODO: alim: this queue is on the number of domains in the dataset
    // if this is on the number of domains, then it will be equivalent to the
    // number
    // of instances in the database

    Initialize();
  }

  void resetBufferSize(const size_t &w, const size_t &h) {
    width = w;
    height = h;
    if (colorBuf != nullptr) {
      delete[] colorBuf;
      delete[] colorBuf_mutex;
    }
    colorBuf_mutex = new tbb::mutex[width];
    colorBuf = new glm::vec4[width * height];
    // std::cout << "Resized buffer" << std::endl;
  }

  virtual void resetInstances() {

    if (acceleration) delete acceleration;

    if (queue_mutex) delete[] queue_mutex;
    meshRef.clear();
    instM.clear();
    instMinv.clear();
    instMinvN.clear();

    for (auto &l : lights) {
      delete l;
    }

    lights.clear();

    Initialize();
  }

  void Initialize() {
    instancenodes = rootnode["Instances"].getChildren();

    int numInst = instancenodes.size();

    queue_mutex = new tbb::mutex[numInst];
    colorBuf_mutex = new tbb::mutex[width];

    acceleration = new gvt::render::data::accel::BVH(instancenodes);

    for (int i = 0; i < instancenodes.size(); i++) {
      meshRef[i] =
          (gvt::render::data::primitives::Mesh *)instancenodes[i]["meshRef"].deRef()["ptr"].value().toULongLong();
      instM[i] = (glm::mat4 *)instancenodes[i]["mat"].value().toULongLong();
      instMinv[i] = (glm::mat4 *)instancenodes[i]["matInv"].value().toULongLong();
      instMinvN[i] = (glm::mat3 *)instancenodes[i]["normi"].value().toULongLong();
    }

    auto lightNodes = rootnode["Lights"].getChildren();

    lights.reserve(2);
    for (auto lightNode : lightNodes) {
      auto color = lightNode["color"].value().tovec3();

      if (lightNode.name() == std::string("PointLight")) {
        auto pos = lightNode["position"].value().tovec3();
        lights.push_back(new gvt::render::data::scene::PointLight(pos, color));
      } else if (lightNode.name() == std::string("AmbientLight")) {
        lights.push_back(new gvt::render::data::scene::AmbientLight(color));
      } else if (lightNode.name() == std::string("AreaLight")) {
        auto pos = lightNode["position"].value().tovec3();
        auto normal = lightNode["normal"].value().tovec3();
        auto width = lightNode["width"].value().toFloat();
        auto height = lightNode["height"].value().toFloat();
        lights.push_back(new gvt::render::data::scene::AreaLight(pos, color, normal, width, height));
      }
    }
  }

  void clearBuffer() { std::memset(colorBuf, 0, sizeof(glm::vec4) * width * height); }

  // clang-format off
  virtual ~AbstractTrace() {};
  virtual void operator()(void) {
    GVT_ASSERT_BACKTRACE(0, "Not supported");
  };
  // clang-format on

  inline void FilterRaysLocally(void) { shuffleRays(rays, -1); }

  /**
   * Given a queue of rays, intersects them against the accel structure
   * to find out what instance they will hit next
   */
  inline void shuffleRays(gvt::render::actor::RayVector &rays, const int domID) {

    size_t chunksize =
        MAX(4096, rays.size() / (gvt::core::CoreContext::instance()->getRootNode()["threads"].value().toInteger() * 4));


    gvt::render::data::accel::BVH &acc = *dynamic_cast<gvt::render::data::accel::BVH *>(acceleration);
    static tbb::auto_partitioner ap;

    tbb::serial::parallel_for(tbb::blocked_range<gvt::render::actor::RayVector::iterator>(rays.begin(), rays.end(), chunksize),
      [&](tbb::blocked_range<gvt::render::actor::RayVector::iterator> raysit) {

      gvt::core::Vector<gvt::render::data::accel::BVH::hit> hits =
      acc.intersect<GVT_SIMD_WIDTH>(raysit.begin(), raysit.end(), domID);

      gvt::core::Map<int, gvt::render::actor::RayVector> local_queue;

      for (size_t i = 0; i < hits.size(); i++) {
        gvt::render::actor::Ray &r = *(raysit.begin() + i);
        bool write_to_fb = false;
        int target_queue = -1;
#ifdef GVT_RENDER_ADAPTER_OSPRAY
         if(r.depth & RAY_BOUNDARY){ 
         // check to see if this ray hit anything in bvh
           if(hits[i].next != -1) {
             r.depth &= ~RAY_BOUNDARY;
             r.origin = r.origin + r.direction *(hits[i].t * (1.0f+std::numeric_limits<float>::epsilon()));
             target_queue = hits[i].next;
             //local_queue[hits[i].next].push_back(r);
             } else {
               r.depth &= ~RAY_BOUNDARY;
               r.depth |= RAY_EXTERNAL_BOUNDARY;
               target_queue = -1;
             }
           }
         // check types
         if(r.type == RAY_PRIMARY) {
           if((r.depth & RAY_OPAQUE) | (r.depth & RAY_EXTERNAL_BOUNDARY)) {
             write_to_fb = true;
             target_queue = -1;
           } else if(r.depth & ~RAY_BOUNDARY) {
             target_queue = domID;
           }
        } else if(r.type == RAY_SHADOW) {
          if(r.depth & RAY_EXTERNAL_BOUNDARY) {
            tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
            colorBuf[r.id] += glm::vec4(r.color,r.w);
          }else if(r.depth & RAY_BOUNDARY) {
            r.origin = r.origin + r.direction *(hits[i].t * 1.00f);
            local_queue[hits[i].next].push_back(r);
          }
        } else if(r.type == RAY_AO) {
          if(r.depth &(RAY_EXTERNAL_BOUNDARY | RAY_TIMEOUT)) {
            tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
            colorBuf[r.id] += glm::vec4(r.color,r.w);
          } else if (r.depth & RAY_BOUNDARY) {
            r.origin = r.origin + r.direction *(hits[i].t * 1.00f);
            local_queue[hits[i].next].push_back(r);
          }
        }
        if(write_to_fb) {
          tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
          colorBuf[r.id] += glm::vec4(r.color,r.w);
        }
        if(target_queue != -1) {
          local_queue[target_queue].push_back(r);
        }
#else
        if (hits[i].next != -1) {
          r.origin = r.origin + r.direction * (hits[i].t * 0.95f);
          local_queue[hits[i].next].push_back(r);
        } else if(r.type == gvt::render::actor::Ray::SHADOW && glm::length(r.color)>0) {
          tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
          colorBuf[r.id] += glm::vec4(r.color, r.w);
        }
#endif
      }

      for (auto &q : local_queue) {

        queue_mutex[q.first].lock();
        queue[q.first].insert(queue[q.first].end(),
        std::make_move_iterator(local_queue[q.first].begin()),
        std::make_move_iterator(local_queue[q.first].end()));
        queue_mutex[q.first].unlock();
      }
    },
  ap);

    rays.clear();
  }

  inline bool SendRays() { GVT_ASSERT_BACKTRACE(0, "Not supported"); }

  inline void gatherFramebuffers(int rays_traced) {

    glm::vec4 *final;

    if (require_composite)
      final = img.execute(colorBuf, width, height);
    else
      final = colorBuf;

    const size_t size = width * height;
    size_t chunksize = MAX(
        GVT_SIMD_WIDTH, size / (gvt::core::CoreContext::instance()->getRootNode()["threads"].value().toInteger() * 4));

    static tbb::simple_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size, chunksize),
    [&](tbb::blocked_range<size_t> chunk) {
    for (size_t i = chunk.begin(); i < chunk.end(); i++) image.Add(i, final[i]);
      },
    ap);
    if (require_composite) delete[] final;
  }
};

/// Generic Tracer interface for a base scheduling strategy with static inner
/// scheduling policy

/*! Tracer implementation generic interface
 *
 * \tparam DomainType Data domain type. Besides defining the domain behavior
 *defines the procedure to process the current queue of rays
 * \tparam MPIW MPI Communication World (Single node or Multiple Nodes)
 * \tparam BSCHEDUDER Base tracer scheduler (e.g. Image, Domain or Hybrid)
 *
 */
template <class BSCHEDULER> class Tracer : public AbstractTrace {
public:
  Tracer(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image) : AbstractTrace(rays, image) {}

  virtual ~Tracer() {}
};

/// Generic Tracer interface for a base scheduling strategy with mutable inner
/// scheduling policy

/*! Tracer implementation generic interface for scheduler with mutable inner
 *scheduling policy
 *
 * \tparam DomainType Data domain type. Besides defining the domain behavior
 *defines the procedure to process the current queue of rays
 * \tparam MPIW MPI Communication World (Single node or Multiple Nodes)
 * \tparam BSCHEDUDER Base tracer scheduler (e.g.Hybrid)
 * \tparam ISCHEDUDER Inner scheduler for base scheduler (Greedy, Spread, ...)
 *
 */
template <template <typename> class BSCHEDULER, class ISCHEDULER>
class Tracer<BSCHEDULER<ISCHEDULER> > : public AbstractTrace {
public:
  Tracer(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image) : AbstractTrace(rays, image) {}

  virtual ~Tracer() {}
};
}
}
}

#endif /* GVT_RENDER_ALGORITHM_TRACER_BASE_H */
