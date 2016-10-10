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

//#define TBB_PREVIEW_STATIC_PARTITIONER 1

#include <gvt/core/Debug.h>
#include <gvt/core/utils/timer.h>
#include <gvt/render/Adapter.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/accel/BVH.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Image.h>

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
    B *gather = new B[partition_size * world_size];
    std::vector<MPI::Request> Irecv_requests_status;

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
      const size_t chunksize = MAX(GVT_SIMD_WIDTH, partition_size / (std::thread::hardware_concurrency()));
      static tbb::simple_partitioner ap;
      tbb::parallel_for(tbb::blocked_range<size_t>(0, partition_size, chunksize),
                        [&](tbb::blocked_range<size_t> chunk) {
#pragma simd
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
    delete gather;
    delete newbuf;
    return newbuf;
  }
};

struct processRay;

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
  gvt::core::DBNodeH rootnode = cntxt.getRootNode();
  gvt::core::Vector<gvt::core::DBNodeH> instancenodes;
  std::map<int, gvt::render::data::primitives::Mesh *> meshRef;
  std::map<int, glm::mat4 *> instM;
  std::map<int, glm::mat4 *> instMinv;
  std::map<int, glm::mat3 *> instMinvN;
  std::vector<gvt::render::data::scene::Light *> lights;

  gvt::render::data::accel::AbstractAccel *acceleration;

  int width = rootnode["Film"]["width"].value().toInteger();
  int height = rootnode["Film"]["height"].value().toInteger();

  float sample_ratio = 1.f;

  tbb::mutex *queue_mutex;                            // array of mutexes - one per instance
  std::map<int, gvt::render::actor::RayVector> queue; ///< Node rays working
  tbb::mutex *colorBuf_mutex;                         ///< buffer for color accumulation
  glm::vec4 *colorBuf;

  gvt::render::composite::composite img;
  bool require_composite = false;

  AbstractTrace(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image)
      : rays(rays), image(image) {
    GVT_DEBUG(DBG_ALWAYS, "initializing abstract trace: num rays: " << rays.size());
    colorBuf = new glm::vec4[width * height];
    require_composite = img.initIceT();
    // TODO: alim: this queue is on the number of domains in the dataset
    // if this is on the number of domains, then it will be equivalent to the
    // number
    // of instances in the database

    Initialize();

    GVT_DEBUG(DBG_ALWAYS, "abstract trace: constructor end");
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
    //std::cout << "Resized buffer" << std::endl;
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

  void Initialize(){

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

  inline void FilterRaysLocally(void) {
    GVT_DEBUG(DBG_ALWAYS, "Generate rays filtering : " << rays.size());
    shuffleRays(rays, -1);
  }

  /**
   * Given a queue of rays, intersects them against the accel structure
   * to find out what instance they will hit next
   */
  inline void shuffleRays(gvt::render::actor::RayVector &rays, const int domID) {

    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] Shuffle: start");
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] Shuffle: rays: " << rays.size());

   // std::cout << "Suffle rays" << rays.size() << std::endl;

    const size_t chunksize = MAX(2, rays.size() / (std::thread::hardware_concurrency() * 4));
    gvt::render::data::accel::BVH &acc = *dynamic_cast<gvt::render::data::accel::BVH *>(acceleration);
    static tbb::simple_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<gvt::render::actor::RayVector::iterator>(rays.begin(), rays.end(), chunksize),
                      [&](tbb::blocked_range<gvt::render::actor::RayVector::iterator> raysit) {
                        std::vector<gvt::render::data::accel::BVH::hit> hits =
                            acc.intersect<GVT_SIMD_WIDTH>(raysit.begin(), raysit.end(), domID);
                        std::map<int, gvt::render::actor::RayVector> local_queue;
                        for (size_t i = 0; i < hits.size(); i++) {
                          gvt::render::actor::Ray &r = *(raysit.begin() + i);
                          if (hits[i].next != -1) {
                            r.origin = r.origin + r.direction * (hits[i].t * 0.95f);
                            local_queue[hits[i].next].push_back(r);
                          } else if (r.type == gvt::render::actor::Ray::SHADOW && glm::length(r.color) > 0) {
                            tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
                            colorBuf[r.id] += glm::vec4(r.color, r.w);
                            // colorBuf[r.id][3] += r.w;
                          }
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

    //std::cout << "Finished shuffle" << std::endl;
    rays.clear();
  }

  inline bool SendRays() { GVT_ASSERT_BACKTRACE(0, "Not supported"); }

  inline void localComposite() {
    // const size_t size = width * height;
    // const size_t chunksize = MAX(2, size / (std::thread::hardware_concurrency() * 4));
    // static tbb::simple_partitioner ap;
    // tbb::parallel_for(tbb::blocked_range<size_t>(0, size, chunksize),
    //                   [&](tbb::blocked_range<size_t> chunk) {
    //                     for (size_t i = chunk.begin(); i < chunk.end(); i++) image.Add(i, colorBuf[i]);
    //                   },
    //                   ap);
  }

  inline void gatherFramebuffers(int rays_traced) {

    glm::vec4 * final;

    if (require_composite)
      final = img.execute(colorBuf, width, height);
    else
      final = colorBuf;

    const size_t size = width * height;
    const size_t chunksize = MAX(2, size / (std::thread::hardware_concurrency() * 4));
    static tbb::simple_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size, chunksize),
                      [&](tbb::blocked_range<size_t> chunk) {
                        for (size_t i = chunk.begin(); i < chunk.end(); i++) image.Add(i, final[i]);
                      },
                      ap);
    if (require_composite) delete[] final;
    // localComposite();
    // mpi.gatherbuffer<unsigned char>(image.GetBuffer(), width * height * 3);

    // size_t size = width * height;
    // unsigned char *rgb = image.GetBuffer();
    //
    // int rgb_buf_size = 3 * size;
    //
    // unsigned char *bufs = mpi.root() ? new unsigned char[mpi.world_size * rgb_buf_size] : NULL;
    //
    // // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Gather(rgb, rgb_buf_size, MPI_UNSIGNED_CHAR, bufs, rgb_buf_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    // if (mpi.root()) {
    //   const size_t chunksize = MAX(2, size / (std::thread::hardware_concurrency() * 4));
    //   static tbb::simple_partitioner ap;
    //   tbb::parallel_for(tbb::blocked_range<size_t>(0, size, chunksize), [&](tbb::blocked_range<size_t> chunk) {
    //
    //     for (int j = chunk.begin() * 3; j < chunk.end() * 3; j += 3) {
    //       for (size_t i = 1; i < mpi.world_size; ++i) {
    //         int p = i * rgb_buf_size + j;
    //         // assumes black background, so adding is fine (r==g==b== 0)
    //         rgb[j + 0] += bufs[p + 0];
    //         rgb[j + 1] += bufs[p + 1];
    //         rgb[j + 2] += bufs[p + 2];
    //       }
    //     }
    //   });
    // }
    //
    // delete[] bufs;
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
