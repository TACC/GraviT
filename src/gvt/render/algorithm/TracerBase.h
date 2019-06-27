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
#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/actor/ORays.h>
#endif
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/accel/BVH.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <gvt/render/composite/ImageComposite.h>

#include <algorithm>
#include <numeric>

#include <mpi.h>

#include <deque>
#include <map>
#include <bitset>

// uncomment this to restrict tbb to serial operation
// must also add "serial" to tbb::parallel_for
// like this tbb::serial::parallel_for. Do this to get reasonable
// prints out of parallel for loop. Of course it doesent run in parallel
// any more.
//#define TBB_PREVIEW_SERIAL_SUBSET 1
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>

namespace gvt {
namespace render {
namespace algorithm {
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif

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


    size_t partition_size = size / world_size;
    size_t next_neighbor = rank;
    size_t prev_neighbor = rank;

    B *acc = &buf[rank * partition_size];
    B gather[] = new B[partition_size * world_size];
    gvt::core::Vector<MPI::Request> Irecv_requests_status;

    for (int round = 0; round < world_size; round++) {
      next_neighbor = (next_neighbor + 1) % world_size;

      if (next_neighbor != rank) {
        B *send = &buf[next_neighbor * partition_size];
        MPI::COMM_WORLD.Isend(send, sizeof(B) * partition_size, MPI::BYTE, next_neighbor, rank | 0xF00000000000000);
      }

      prev_neighbor = (prev_neighbor > 0 ? prev_neighbor - 1 : world_size - 1);

      if (prev_neighbor != rank) {
        B *recv = &gather[prev_neighbor * partition_size];
        Irecv_requests_status.push_back(MPI::COMM_WORLD.Irecv(recv, sizeof(B) * partition_size, MPI::BYTE,
                                                              prev_neighbor, prev_neighbor | 0xF00000000000000));
      }
    }

    MPI::Request::Waitall(Irecv_requests_status.size(), &Irecv_requests_status[0]);

    for (int source = 0; source < world_size; ++source) {
      if (source == rank) continue;
      const size_t chunksize =
          MAX(GVT_SIMD_WIDTH, partition_size / (cntx::rcontext::instance().getUnique("threads").to<unsigned>() * 4));
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

  // Just a ppointer to the database so that we don't have to keep going
  cntx::rcontext &db;
  std::string camname;
  std::string filmname;
  std::string schedulername;

  gvt::render::actor::RayVector &rays; ///< Rays to trace
  std::shared_ptr<gvt::render::data::scene::gvtCameraBase> camera;
  std::shared_ptr<gvt::render::composite::ImageComposite> image; ///< Final image buffer

  gvt::core::Map<size_t, std::shared_ptr<gvt::render::data::primitives::Data> > meshRef;
  gvt::core::Map<size_t, std::shared_ptr<glm::mat4> > instM;
  gvt::core::Map<size_t, std::shared_ptr<glm::mat4> > instMinv;
  gvt::core::Map<size_t, std::shared_ptr<glm::mat3> > instMinvN;
  gvt::core::Vector<std::shared_ptr<gvt::render::data::scene::Light> > lights;

  std::shared_ptr<gvt::render::data::accel::AbstractAccel> acceleration;

  int width;
  int height;
  int adapterType;

  float sample_ratio;

  // array of mutexes - one per instance
  gvt::core::Map<size_t, gvt::render::actor::RayVector> queue; ///< Node rays working

  // TODO: Change the pointers into a smart pointer semantic
  // NOTE: Just not doing that now for time constrainst.

  tbb::mutex *queue_mutex;
  tbb::mutex *colorBuf_mutex; ///< buffer for color accumulation
                              //  glm::vec4 *colorBuf;

  //  gvt::render::composite::composite img;
  bool require_composite;

  AbstractTrace(std::shared_ptr<gvt::render::data::scene::gvtCameraBase> camera,
                std::shared_ptr<gvt::render::composite::ImageComposite> image, std::string const &camname = "Camera",
                std::string const &filmname = "Film", std::string const &schedulername = "Scheduler")
      : camera(camera), rays(camera->rays), image(image), db(cntx::rcontext::instance()), camname(camname),
        filmname(filmname), schedulername(schedulername) {

    width = db.getChild(db.getUnique(filmname), "width");
    height = db.getChild(db.getUnique(filmname), "height");

    // sample_ratio = db.getChild(db.getUnique(camname), "raySamples");

    //    require_composite = false;
    //    require_composite = img.initIceT();
    // NOTE : Replaced by smat pointer
    // colorBuf = new glm::vec4[width * height];
    Initialize();
  }

  void resetBufferSize(const size_t &w, const size_t &h) {
    width = w;
    height = h;

    // Create buffers using smart pointers
    _colorBuf_mutex = std::shared_ptr<tbb::mutex>(new tbb::mutex[width], std::default_delete<tbb::mutex[]>());

    // Alias buffers

    colorBuf_mutex = _colorBuf_mutex.get();
    //    colorBuf = _colorBuf.get();

  }

  virtual void resetInstances() {

    // Replace by smart pointer semantics
    // if (acceleration) delete acceleration;
    // if (queue_mutex) delete[] queue_mutex;

    meshRef.clear();
    instM.clear();
    instMinv.clear();
    instMinvN.clear();
    lights.clear();

    Initialize();
  }

  void Initialize() {

    // TODO: Get all instances

    // instancenodes = rootnode["Instances"].getChildren();

    auto inst = db.getChildren(db.getUnique("Instances"));
    const size_t numInst = inst.size();

    _queue_mutex = std::shared_ptr<tbb::mutex>(new tbb::mutex[numInst], std::default_delete<tbb::mutex[]>());

    queue_mutex = _queue_mutex.get();

    resetBufferSize(width, height);

    acceleration = std::make_shared<gvt::render::data::accel::BVH>(inst);

    for (auto &nref : acceleration->instanceSet) {
      auto &n = nref.get();
      size_t id = db.getChild(n, "id");

      if (db.getChild(db.deRef(db.getChild(n, "meshRef")), "ptr")
              .v.is<std::shared_ptr<gvt::render::data::primitives::Mesh> >()) {
        meshRef[id] = db.getChild(db.deRef(db.getChild(n, "meshRef")), "ptr")
                          .to<std::shared_ptr<gvt::render::data::primitives::Mesh> >();
      } 
#ifdef GVT_BUILD_VOLUME      
      else if (db.getChild(db.deRef(db.getChild(n, "meshRef")), "ptr")
                     .v.is<std::shared_ptr<gvt::render::data::primitives::Volume> >()) {
        meshRef[id] = db.getChild(db.deRef(db.getChild(n, "meshRef")), "ptr")
                          .to<std::shared_ptr<gvt::render::data::primitives::Volume> >();
      } 
#endif       
      else {
        meshRef[id] = nullptr;
      }
      instM[id] = db.getChild(n, "mat");
      instMinv[id] = db.getChild(n, "matinv");
      instMinvN[id] = db.getChild(n, "normi");
    }

    auto lightNodes = db.getChildren(db.getUnique("Lights"));

    lights.reserve(lightNodes.size());
    for (auto lightNode : lightNodes) {
      auto &light = lightNode.get();
      glm::vec3 color = db.getChild(light, "color");
      std::string type = db.getChild(light, "type");
      if (type == std::string("PointLight")) {
        glm::vec3 pos = db.getChild(light, "position");
        lights.push_back(std::make_shared<gvt::render::data::scene::PointLight>(pos, color));
      } else if (type == std::string("AmbientLight")) {
        lights.push_back(std::make_shared<gvt::render::data::scene::AmbientLight>(color));
      } else if (type == std::string("AreaLight")) {
        glm::vec3 pos = db.getChild(light, "position");
        glm::vec3 normal = db.getChild(light, "normal");
        auto width = db.getChild(light, "width");
        auto height = db.getChild(light, "height");
        lights.push_back(std::make_shared<gvt::render::data::scene::AreaLight>(pos, color, normal, width, height));
      }
    }
  }

  void clearBuffer() { image->reset(); /*std::memset(colorBuf, 0, sizeof(glm::vec4) * width * height);*/ }

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

    const size_t chunksize = MAX(4096, rays.size() / (db.getUnique("threads").to<unsigned>() * 4));
    gvt::render::data::accel::BVH &acc = *dynamic_cast<gvt::render::data::accel::BVH *>(acceleration.get());

    static tbb::auto_partitioner ap;

    //tbb::serial::parallel_for(
tbb::parallel_for(
        tbb::blocked_range<gvt::render::actor::RayVector::iterator>(rays.begin(), rays.end(), chunksize),
        [&](tbb::blocked_range<gvt::render::actor::RayVector::iterator> raysit) {
          gvt::core::Vector<gvt::render::data::accel::BVH::hit> hits =
              acc.intersect<GVT_SIMD_WIDTH>(raysit.begin(), raysit.end(), domID);

          gvt::core::Map<int, gvt::render::actor::RayVector> local_queue;
          for (size_t i = 0; i < hits.size(); i++) {
            gvt::render::actor::Ray &r = *(raysit.begin() + i);
            bool write_to_fb = false;
            int target_queue = -1;
            if ( adapterType == gvt::render::adapter::Pvol 
              || adapterType == gvt::render::adapter::Ospray ) {
              if (r.mice.depth & RAY_BOUNDARY) {
                // check to see if this ray hit anything in bvh
                if (hits[i].next != -1) {
                  r.mice.depth &= ~RAY_BOUNDARY; // turn off boundary term type
                  r.mice.origin = r.mice.origin + r.mice.direction * (hits[i].t * (1.0f + std::numeric_limits<float>::epsilon())); // bump the origin into next volume
                  target_queue = hits[i].next; 
                  // local_queue[hits[i].next].push_back(r);
                } else { // elvis is leaving the building
                  r.mice.depth &= ~RAY_BOUNDARY; // turn off ray boundary
                  r.mice.depth |= RAY_EXTERNAL_BOUNDARY; // set term type to external
                  target_queue = -1;
                }
              }
              if (r.mice.type == RAY_PRIMARY) {
                if ((r.mice.depth & RAY_OPAQUE) | (r.mice.depth & RAY_EXTERNAL_BOUNDARY)) {
                  write_to_fb = true;
                  target_queue = -1;
                } else if (r.mice.depth & ~RAY_BOUNDARY) {
                  target_queue = domID;
                }
              } else if (r.mice.type == RAY_SHADOW) {
                if (r.mice.depth & RAY_EXTERNAL_BOUNDARY) {
                  tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.mice.id % width]);
                  // colorBuf[r.id] += glm::vec4(r.color, r.w);
                  image->localAdd(r.mice.id, r.mice.color * r.mice.w, 1.f, r.mice.t);
                } else if (r.mice.depth & RAY_BOUNDARY) {
                  r.mice.origin = r.mice.origin + r.mice.direction * (hits[i].t * 1.00f);
                  local_queue[hits[i].next].push_back(r);
                }
              } else if (r.mice.type == RAY_AO) {
                if (r.mice.depth & (RAY_EXTERNAL_BOUNDARY | RAY_TIMEOUT)) {
                  tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.mice.id % width]);
                  //                colorBuf[r.id] += glm::vec4(r.color, r.w);
                  image->localAdd(r.mice.id, r.mice.color * r.mice.w, 1.f, r.mice.t);
                } else if (r.mice.depth & RAY_BOUNDARY) {
                  r.mice.origin = r.mice.origin + r.mice.direction * (hits[i].t * 1.00f);
                  local_queue[hits[i].next].push_back(r);
                }
              }
              if (write_to_fb) {
                tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.mice.id % width]);
                image->localAdd(r.mice.id, r.mice.color * r.mice.w, 1.f, r.mice.t);
              }
              if (target_queue != -1) {
                local_queue[target_queue].push_back(r);
              }
            } else {
              if (hits[i].next != -1) {
                r.mice.origin = r.mice.origin + r.mice.direction * (hits[i].t * 0.95f);
                local_queue[hits[i].next].push_back(r);
              } else if (r.mice.type == gvt::render::actor::Ray::SHADOW && glm::length(r.mice.color) > 0) {
                tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.mice.id % width]);
                // colorBuf[r.id] += glm::vec4(r.color, r.w);
                image->localAdd(r.mice.id, r.mice.color * r.mice.w, 1.f, r.mice.t);
              }
            }
          }
          for (auto &q : local_queue) {

            queue_mutex[q.first].lock();
            queue[q.first].insert(queue[q.first].end(), std::make_move_iterator(local_queue[q.first].begin()),
                                  std::make_move_iterator(local_queue[q.first].end()));
            queue_mutex[q.first].unlock();
          }
        },
        ap);

    rays.clear();
  }

  inline bool SendRays() { GVT_ASSERT_BACKTRACE(0, "Not supported"); }

  inline void gatherFramebuffers(int rays_traced) {
    image->composite();
    //    glm::vec4 *final;
    //
    //    if (require_composite)
    //      final = img.execute(colorBuf, width, height);
    //    else
    //      final = colorBuf;
    //
    //    const size_t size = width * height;
    //    const size_t chunksize = MAX(
    //        GVT_SIMD_WIDTH, size / (db.getUnique("threads").to<unsigned>() * 4));
    //    static tbb::simple_partitioner ap;
    //    tbb::parallel_for(tbb::blocked_range<size_t>(0, size, chunksize),
    //                      [&](tbb::blocked_range<size_t> chunk) {
    //                        for (size_t i = chunk.begin(); i < chunk.end(); i++) image->Add(i, final[i]);
    //                      },
    //                      ap);
    //    if (require_composite) delete[] final;
  }

private:
  // NOTE: Creating smart pointer so that I don't have to delete them later.
  // NOTE: Not necessary, just don't want to manage memory manually

  std::shared_ptr<tbb::mutex> _queue_mutex;
  std::shared_ptr<tbb::mutex> _colorBuf_mutex;
  //  std::shared_ptr<glm::vec4> _colorBuf;

protected:
  /// caches meshes that are converted into the adapter's format
  gvt::core::Map<std::shared_ptr<gvt::render::data::primitives::Data>, std::shared_ptr<gvt::render::Adapter> >
      adapterCache;
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
  Tracer(std::shared_ptr<gvt::render::data::scene::gvtCameraBase> camera,
         std::shared_ptr<gvt::render::composite::ImageComposite> image, std::string const &camname = "Camera",
         std::string const &filmname = "Film", std::string const &schedulername = "Scheduler")
      : AbstractTrace(camera, image, camname, filmname, schedulername) {}

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
  Tracer(std::shared_ptr<gvt::render::data::scene::gvtCameraBase> camera,
         std::shared_ptr<gvt::render::composite::ImageComposite> image, std::string const &camname = "Camera",
         std::string const &filmname = "Film", std::string const &schedulername = "Scheduler")
      : AbstractTrace(camera, image, camname, filmname, schedulername) {}

  virtual ~Tracer() {}
};
} // namespace algorithm
} // namespace render
} // namespace gvt

#endif /* GVT_RENDER_ALGORITHM_TRACER_BASE_H */
