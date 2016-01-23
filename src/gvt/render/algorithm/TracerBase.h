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
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/accel/BVH.h>
#include <gvt/render/Adapter.h>

#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/timer/timer.hpp>
#include <boost/range/algorithm.hpp>

#include <algorithm>
#include <future>
#include <numeric>

#include <mpi.h>

#include <map>

#include <tbb/parallel_for_each.h>
#include <tbb/tick_count.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>

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

  gvt::render::data::accel::AbstractAccel *acceleration;

  int width = rootnode["Film"]["width"].value().toInteger();
  int height = rootnode["Film"]["height"].value().toInteger();

  float sample_ratio;

  tbb::mutex raymutex;
  tbb::mutex *queue_mutex;                            // array of mutexes - one per instance
  std::map<int, gvt::render::actor::RayVector> queue; ///< Node rays working
  tbb::mutex *colorBuf_mutex;                         ///< buffer for color accumulation
  GVT_COLOR_ACCUM *colorBuf;

  AbstractTrace(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image)
      : rays(rays), image(image) {
    GVT_DEBUG(DBG_ALWAYS, "initializing abstract trace: num rays: " << rays.size());
    colorBuf = new GVT_COLOR_ACCUM[width * height];

    // TODO: alim: this queue is on the number of domains in the dataset
    // if this is on the number of domains, then it will be equivalent to the
    // number
    // of instances in the database
    instancenodes = rootnode["Instances"].getChildren();
    int numInst = instancenodes.size();
    GVT_DEBUG(DBG_ALWAYS, "abstract trace: num instances: " << numInst);
    queue_mutex = new tbb::mutex[numInst];
    colorBuf_mutex = new tbb::mutex[width];
    acceleration = new gvt::render::data::accel::BVH(instancenodes);

    GVT_DEBUG(DBG_ALWAYS, "abstract trace: constructor end");
  }

  // clang-format off
  virtual ~AbstractTrace() {};
  virtual void operator()(void) {
    GVT_ASSERT_BACKTRACE(0, "Not supported");
  };
  // clang-format on

  virtual void FilterRaysLocally(void) {
    GVT_DEBUG(DBG_ALWAYS, "Generate rays filtering : " << rays.size());
    shuffleRays(rays, gvt::core::DBNodeH());
  }

#if 0
  /**
   * Given a queue of rays, intersects them against the accel structure
   * to find out what instance they will hit next
   */
  virtual void shuffleRays(gvt::render::actor::RayVector &rays,
                           gvt::core::DBNodeH instNode) {

    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] Shuffle: start");
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank
                              << "] Shuffle: rays: " << rays.size());

#ifdef GVT_USE_DEBUG
    boost::timer::auto_cpu_timer t("Ray shuflle %t\n");
#endif
    int nchunks = 1; // std::thread::hardware_concurrency();
    int chunk_size = rays.size() / nchunks;
    std::vector<std::pair<int, int>> chunks;
    std::vector<std::future<void>> futures;
    for (int ii = 0; ii < nchunks - 1; ii++) {
      chunks.push_back(
          std::make_pair(ii * chunk_size, ii * chunk_size + chunk_size));
    }
    int ii = nchunks - 1;
    chunks.push_back(std::make_pair(ii * chunk_size, rays.size()));
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank
                              << "] Shuffle: chunks: " << chunks.size());

    int idnode = (instNode)
                     ? gvt::core::variant_toInteger(instNode["id"].value())
                     : 0xFFFFFFFF;
    gvt::render::data::primitives::Box3D &wBox =
        *gvt::core::variant_toBox3DPtr(instNode["bbox"].value());

    for (auto limit : chunks) {
      futures.push_back(std::async(std::launch::deferred, [&]() {
        int chunk = limit.second - limit.first;
        std::map<int, gvt::render::actor::RayVector> local_queue;
        gvt::render::actor::RayVector local(chunk);
        local.assign(rays.begin() + limit.first, rays.begin() + limit.second);
        GVT_DEBUG(DBG_ALWAYS,
                  "[" << mpi.rank
                      << "] Shuffle: looping through local rays: num local: "
                      << local.size());
        // go through the local list of rays and stash them in
        // local_queue[dom] where dom is the first "domain" the
        // ray intersects.

        int idnode = (instNode)
                         ? gvt::core::variant_toInteger(instNode["id"].value())
                         : 0xFFFFFFFF;
        gvt::render::data::primitives::Box3D &wBox =
            *gvt::core::variant_toBox3DPtr(instNode["bbox"].value());

        for (gvt::render::actor::Ray &r : local) {
          gvt::render::actor::isecDomList &len2List = r.domains;

          if (len2List.empty() && instNode) {
            //   // instance(?)->marchOut(r);

            float t = FLT_MAX;
            if (wBox.intersectDistance(r, t))
              r.origin += r.direction * t;
            //   while (wBox.intersectDistance(r, t)) {
            //     r.origin += r.direction * t;
            //     r.origin += r.direction *
            //     gvt::render::actor::Ray::RAY_EPSILON;
            //   }
            r.origin += r.direction * gvt::render::actor::Ray::RAY_EPSILON;
          }

          if (len2List.empty()) {
            // intersect the bvh to find the instance hit list
            acceleration->intersect(r, len2List);
            boost::sort(len2List);
          }

          if (!len2List.empty() && (int)(*len2List.begin()) == idnode) {
            len2List.erase(len2List.begin());
          }

          // TODO: alim: figure out new shuffle algorithm, as adapter is going
          // to
          // be null right now(?)
          if (!len2List.empty()) {
            int firstDomainOnList = (*len2List.begin());
            len2List.erase(len2List.begin());
            local_queue[firstDomainOnList].push_back(r);
          } else if (instNode) {
            boost::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
            for (int i = 0; i < 3; i++)
              colorBuf[r.id].rgba[i] += r.color.rgba[i];
            colorBuf[r.id].rgba[3] = 1.f;
            colorBuf[r.id].clamp();
          }
        }

        GVT_DEBUG(DBG_ALWAYS,
                  "[" << mpi.rank
                      << "] Shuffle: adding rays to queues num local: "
                      << local_queue.size());
        for (auto &q : local_queue) {
          boost::mutex::scoped_lock sl(queue_mutex[q.first]);
          GVT_DEBUG(DBG_ALWAYS, "Add " << q.second.size() << " to queue "
                                       << q.first << " width size "
                                       << queue[q.first].size() << "["
                                       << mpi.rank << "]");
          queue[q.first].insert(queue[q.first].end(), q.second.begin(),
                                q.second.end());
        }
      }));
    }
    for (auto &f : futures)
      f.wait();
    rays.clear();
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] Shuffle exit");
  }
#endif

  /**
   * Given a queue of rays, intersects them against the accel structure
   * to find out what instance they will hit next
   */
  virtual void shuffleRays(gvt::render::actor::RayVector &rays, gvt::core::DBNodeH instNode) {

    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] Shuffle: start");
    GVT_DEBUG(DBG_ALWAYS, "[" << mpi.rank << "] Shuffle: rays: " << rays.size());

    const size_t raycount = rays.size();
    const int domID = (instNode) ? instNode["id"].value().toInteger() : -1;
    const gvt::render::data::primitives::Box3D domBB =
        (instNode) ? *((gvt::render::data::primitives::Box3D *)instNode["bbox"].value().toULongLong())
                   : gvt::render::data::primitives::Box3D();

    // tbb::parallel_for(size_t(0), size_t(rays.size()),
    //      [&] (size_t index) {

    // clang-format off
    tbb::parallel_for(tbb::blocked_range<gvt::render::actor::RayVector::iterator>(rays.begin(), rays.end()),
                      [&](tbb::blocked_range<gvt::render::actor::RayVector::iterator> raysit) {
      //        gvt::render::actor::Ray &r = rays[index];

      std::map<int, gvt::render::actor::RayVector> local_queue;

      for (gvt::render::actor::Ray &r : raysit) {
        if (domID != -1) {
          float t = FLT_MAX;
          if (r.domains.empty() && domBB.intersectDistance(r, t)) {
            r.origin += r.direction * t;
          }
        }

        if (r.domains.empty()) {
          acceleration->intersect(r, r.domains);
          boost::sort(r.domains);
        }

        if (!r.domains.empty() && (int)(*r.domains.begin()) == domID) {
          r.domains.erase(r.domains.begin());
        }

        if (!r.domains.empty()) {

          int firstDomainOnList = (*r.domains.begin());
          r.domains.erase(r.domains.begin());
          // tbb::mutex::scoped_lock sl(queue_mutex[firstDomainOnList]);
          local_queue[firstDomainOnList].push_back(r);

        } else if (instNode) {

          tbb::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % width]);
          for (int i = 0; i < 3; i++) colorBuf[r.id].rgba[i] += r.color.rgba[i];
          colorBuf[r.id].rgba[3] = 1.f;
          colorBuf[r.id].clamp();
        }
      }

      std::vector<int> _doms;
      std::transform(local_queue.begin(), local_queue.end(), std::back_inserter(_doms),
                     [](const std::map<int, gvt::render::actor::RayVector>::value_type &pair) { return pair.first; });

      while (!_doms.empty()) {

        int dom = _doms.front();
        _doms.erase(_doms.begin());
        if (queue_mutex[dom].try_lock()) {
          queue[dom].insert(queue[dom].end(), std::make_move_iterator(local_queue[dom].begin()),
                            std::make_move_iterator(local_queue[dom].end()));
          queue_mutex[dom].unlock();
        } else {
          _doms.push_back(dom);
        }
      }

      // for (auto &q : local_queue) {
      //   const int dom = q.first;
      //   const size_t size = q.second.size();
      //   tbb::mutex::scoped_lock sl(queue_mutex[dom]);
      //   // queue[dom].reserve(queue[dom].size() + size);
      //   queue[dom].insert(queue[dom].end(),
      //                     std::make_move_iterator(q.second.begin()),
      //                     std::make_move_iterator(q.second.end()));

      //   // std::move(q.second.begin(), q.second.end(),
      //   // std::back_inserter(queue[dom]));
      // }
    });
    // clang-format on
    rays.clear();
  }

  virtual bool SendRays() { GVT_ASSERT_BACKTRACE(0, "Not supported"); }

  virtual void localComposite() {
    const size_t size = width * height;

    int nchunks = std::thread::hardware_concurrency() * 2;
    int chunk_size = size / nchunks;
    std::vector<std::pair<int, int> > chunks;
    std::vector<std::future<void> > futures;
    for (int ii = 0; ii < nchunks - 1; ii++) {
      chunks.push_back(std::make_pair(ii * chunk_size, ii * chunk_size + chunk_size));
    }
    int ii = nchunks - 1;
    chunks.push_back(std::make_pair(ii * chunk_size, size));

    for (auto limit : chunks) {
      GVT_DEBUG(DBG_ALWAYS, "Limits : " << limit.first << ", " << limit.second);
      futures.push_back(std::async(std::launch::deferred, [&]() {
        for (int i = limit.first; i < limit.second; i++) image.Add(i, colorBuf[i]);
      }));
    }

    for (std::future<void> &f : futures) {
      f.wait();
    }
  }

  virtual void gatherFramebuffers(int rays_traced) {

    size_t size = width * height;

    for (size_t i = 0; i < size; i++) image.Add(i, colorBuf[i]);

    if (!mpi) return;

    unsigned char *rgb = image.GetBuffer();

    int rgb_buf_size = 3 * size;

    unsigned char *bufs = mpi.root() ? new unsigned char[mpi.world_size * rgb_buf_size] : NULL;

    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(rgb, rgb_buf_size, MPI_UNSIGNED_CHAR, bufs, rgb_buf_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (mpi.root()) {
      int nchunks = std::thread::hardware_concurrency() * 2;
      int chunk_size = size / nchunks;
      std::vector<std::pair<int, int> > chunks(nchunks);
      std::vector<std::future<void> > futures;
      for (int ii = 0; ii < nchunks - 1; ii++) {
        chunks.push_back(std::make_pair(ii * chunk_size, ii * chunk_size + chunk_size));
      }
      int ii = nchunks - 1;
      chunks.push_back(std::make_pair(ii * chunk_size, size));

      for (auto &limit : chunks) {
        futures.push_back(std::async(std::launch::async, [&]() {
          // std::pair<int,int> limit = std::make_pair(0,size);
          for (size_t i = 1; i < mpi.world_size; ++i) {
            for (int j = limit.first * 3; j < limit.second * 3; j += 3) {
              int p = i * rgb_buf_size + j;
              // assumes black background, so adding is fine (r==g==b== 0)
              rgb[j + 0] += bufs[p + 0];
              rgb[j + 1] += bufs[p + 1];
              rgb[j + 2] += bufs[p + 2];
            }
          }
        }));
      }
      for (std::future<void> &f : futures) {
        f.wait();
      }
    }

    delete[] bufs;
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
