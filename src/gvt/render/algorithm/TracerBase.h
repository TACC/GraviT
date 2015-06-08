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
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <algorithm>
#include <future>
#include <numeric>

#include <mpi.h>

#include <map>

namespace gvt {
namespace render {
namespace algorithm {

/// Tracer base class

struct GVT_COMM {
  const size_t rank;
  const size_t world_size;

  GVT_COMM()
      : rank(MPI::COMM_WORLD.Get_rank()),
        world_size(MPI::COMM_WORLD.Get_size()) {}

  operator bool() { return (world_size > 1); }
};

struct processRay;

class AbstractTrace {
 public:
  ///! Define mpi communication world
  GVT_COMM mpi;

  gvt::render::actor::RayVector& rays;     ///< Rays to trace
  gvt::render::data::scene::Image& image;  ///< Final image buffer

  unsigned char* vtf;
  float sample_ratio;

  boost::mutex raymutex;
  boost::mutex* queue_mutex;
  std::map<int, gvt::render::actor::RayVector>
      queue;  ///< Node rays working queue
  // std::map<int, std::mutex> queue;
  // buffer for color accumulation
  boost::mutex* colorBuf_mutex;
  GVT_COLOR_ACCUM* colorBuf;

  AbstractTrace(gvt::render::actor::RayVector& rays,
                gvt::render::data::scene::Image& image)
      : rays(rays), image(image) {
    vtf = gvt::render::Attributes::rta->GetTransferFunction();
    sample_ratio = gvt::render::Attributes::rta->sample_ratio;
    colorBuf = new GVT_COLOR_ACCUM[gvt::render::RTA::instance()->view.width *
                                   gvt::render::RTA::instance()->view.height];
    queue_mutex =
        new boost::mutex[gvt::render::Attributes::rta->dataset->size()];
    colorBuf_mutex = new boost::mutex[gvt::render::RTA::instance()->view.width];

    for (int i = 0; i < gvt::render::Attributes::rta->dataset->size(); i++) {
      queue[i] = gvt::render::actor::RayVector();
      queue[i].reserve(gvt::render::RTA::instance()->view.width);
    }
  }

  virtual ~AbstractTrace(){
      // delete queue_mutex;
      // delete colorBuf;
      // delete colorBuf_mutex;
  };
  virtual void operator()(void) { GVT_ASSERT_BACKTRACE(0, "Not supported"); };

  virtual void FilterRaysLocally(void) {
    GVT_DEBUG(DBG_ALWAYS, "Generate rays filtering : " << rays.size());
    shuffleRays(rays);
  }

  /***
  *   Given a queue of rays:
  *     - Moves the ray to the next domain on the list
  *     - 
  *
  */

  virtual void shuffleRays(gvt::render::actor::RayVector& rays, gvt::render::data::domain::AbstractDomain* dom = NULL) {

    boost::timer::auto_cpu_timer t("Ray shuflle %t\n");

    int nchunks = 1; //std::thread::hardware_concurrency();
    int chunk_size = rays.size() / nchunks;
    std::vector<std::pair<int, int>> chunks;
    std::vector<std::future<void>> futures;
    for (int ii = 0; ii < nchunks - 1; ii++) {
      chunks.push_back(
          std::make_pair(ii * chunk_size, ii * chunk_size + chunk_size));
    }
    int ii = nchunks - 1;
    chunks.push_back(std::make_pair(ii * chunk_size, rays.size()));
    for (auto limit : chunks) {
      GVT_DEBUG(DBG_ALWAYS,"Limits : " << limit.first << ", " << limit.second);
      futures.push_back(std::async(std::launch::deferred, [&]() {
        int chunk = limit.second - limit.first;
        std::map<int, gvt::render::actor::RayVector> local_queue;
        gvt::render::actor::RayVector local(chunk);
        local.assign(rays.begin() + limit.first, rays.begin() + limit.second);
        for (gvt::render::actor::Ray& r : local) {
          gvt::render::actor::isecDomList& len2List = r.domains;
          
          if (len2List.empty() && dom) dom->marchOut(r);

          if (len2List.empty()) {
            gvt::render::Attributes::rta->dataset->intersect(r, len2List);
          }
          
          if (!len2List.empty()) {
            int firstDomainOnList = (*len2List.begin());
            len2List.erase(len2List.begin());
            local_queue[firstDomainOnList].push_back(r);

          } else if(dom) {
            boost::mutex::scoped_lock fbloc(colorBuf_mutex[r.id % gvt::render::Attributes::instance()->view.width]);
            for (int i = 0; i < 3; i++)
              colorBuf[r.id].rgba[i] += r.color.rgba[i];
            colorBuf[r.id].rgba[3] = 1.f;
            colorBuf[r.id].clamp();
          }
        }
        for (auto& q : local_queue) {
          boost::mutex::scoped_lock(queue_mutex[q.first]);
          GVT_DEBUG(DBG_ALWAYS, "Add " << q.second.size() << " to queue "
                                       << q.first << " width size "
                                       << queue[q.first].size());
          queue[q.first].insert(queue[q.first].end(), q.second.begin(),
                                q.second.end());
        }
     }));
    }
    rays.clear();
    for (auto& f : futures) f.wait();
  }

  virtual bool SendRays() { GVT_ASSERT_BACKTRACE(0, "Not supported"); }

  virtual void localComposite() {
    const size_t size = gvt::render::Attributes::rta->view.width *
                        gvt::render::Attributes::rta->view.height;


    for (int i = 0; i < size; i++)
          image.Add(i, colorBuf[i]);



    // int nchunks = std::thread::hardware_concurrency() * 2;
    // int chunk_size = size / nchunks;
    // std::vector<std::pair<int, int>> chunks;
    // std::vector<std::future<void>> futures;
    // for (int ii = 0; ii < nchunks - 1; ii++) {
    //   chunks.push_back(
    //       std::make_pair(ii * chunk_size, ii * chunk_size + chunk_size));
    // }
    // int ii = nchunks - 1;
    // chunks.push_back(std::make_pair(ii * chunk_size, size));
    
    // for (auto limit : chunks) {
    //   GVT_DEBUG(DBG_ALWAYS,"Limits : " << limit.first << ", " << limit.second);
    //   futures.push_back(std::async(std::launch::deferred, [&]() {
    //     for (int i = limit.first; i < limit.second; i++)
    //       image.Add(i, colorBuf[i]);
    //    }));
    // }

    // for (std::future<void>& f : futures) {
    //   f.wait();
    // }
  }

  virtual void gatherFramebuffers(int rays_traced) {
    localComposite();
    if (!mpi) return;

    size_t size = gvt::render::Attributes::rta->view.width *
                  gvt::render::Attributes::rta->view.height;
    unsigned char* rgb = image.GetBuffer();

    int rgb_buf_size = 3 * size;

    unsigned char* bufs = (mpi.rank == 0)
                              ? new unsigned char[mpi.world_size * rgb_buf_size]
                              : NULL;

    MPI_Gather(rgb, rgb_buf_size, MPI_UNSIGNED_CHAR, bufs, rgb_buf_size,
               MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (mpi.rank) return;

    int nchunks = std::thread::hardware_concurrency() * 2;
    int chunk_size = size / nchunks;
    std::vector<std::pair<int, int>> chunks(nchunks);
    std::vector<std::future<void>> futures;
    for (int ii = 0; ii < nchunks - 1; ii++) {
      chunks.push_back(
          std::make_pair(ii * chunk_size, ii * chunk_size + chunk_size));
    }
    int ii = nchunks - 1;
    chunks.push_back(std::make_pair(ii * chunk_size, size));

    for (auto& limit : chunks) {
      futures.push_back(std::async(std::launch::async, [&]() {
        for (int i = 1; i < mpi.world_size; i++) {
          for (int id = limit.first; id < limit.second; i++) {
            int src = i * rgb_buf_size + id;
            rgb[id + 0] = bufs[src + 0];
            rgb[id + 1] = bufs[src + 1];
            rgb[id + 2] = bufs[src + 2];
          }
        }
      }));
    }
    for (std::future<void>& f : futures) {
      f.wait();
    }
  }
};

struct processRayVector {
  AbstractTrace* tracer;
  gvt::render::actor::RayVector& rays;
  boost::atomic<int>& current_ray;
  int last;
  const size_t split;
  gvt::render::data::domain::AbstractDomain* dom;

  processRayVector(AbstractTrace* tracer, gvt::render::actor::RayVector& rays,
                   boost::atomic<int>& current_ray, int last, const int split,
                   gvt::render::data::domain::AbstractDomain* dom = NULL)
      : tracer(tracer),
        rays(rays),
        current_ray(current_ray),
        last(last),
        split(split),
        dom(dom) {}

  void operator()() {
    gvt::render::actor::RayVector localQueue;
    while (!rays.empty()) {
      localQueue.clear();
      boost::unique_lock<boost::mutex> lock(tracer->raymutex);
      std::size_t range = std::min(split, rays.size());

      GVT_DEBUG(DBG_ALWAYS, "processRayVector: current_ray "
                                << current_ray << " last ray " << last
                                << " split " << split << " rays.size()"
                                << rays.size());

      localQueue.assign(rays.begin(), rays.begin() + range);
      rays.erase(rays.begin(), rays.begin() + range);
      lock.unlock();

      for (int i = 0; i < localQueue.size(); i++) {
        gvt::render::actor::Ray& ray = localQueue[i];
        gvt::render::actor::isecDomList& len2List = ray.domains;
        if (len2List.empty() && dom) dom->marchOut(ray);
        if (len2List.empty())
          gvt::render::Attributes::rta->dataset->intersect(ray, len2List);
        if (!len2List.empty()) {
          int domTarget = (*len2List.begin());
          len2List.erase(len2List.begin());
          boost::mutex::scoped_lock qlock(tracer->queue_mutex[domTarget]);
          tracer->queue[domTarget].push_back(ray);
        } else {
          boost::mutex::scoped_lock fbloc(
              tracer->colorBuf_mutex
                  [ray.id % gvt::render::Attributes::instance()->view.width]);
          for (int i = 0; i < 3; i++)
            tracer->colorBuf[ray.id].rgba[i] += ray.color.rgba[i];
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
template <class MPIW>
class TracerBase : public MPIW, public AbstractTrace {
 public:
  TracerBase(gvt::render::actor::RayVector& rays,
             gvt::render::data::scene::Image& image)
      : MPIW(rays.size()), AbstractTrace(rays, image) {}

  virtual ~TracerBase() {}

  /*! Trace operation
   *
   * Implements the code for the tracer
   *
   *
   */

  virtual void operator()(void) { GVT_ASSERT_BACKTRACE(0, "Not supported"); }

  /*! Generate rays
   *
   * Used inside the operator to generate the tree of rays.
   *
   *
   */

  // virtual void FilterRaysLocally() {
  //   boost::atomic<int> current_ray(this->rays_start);
  //   GVT_DEBUG(DBG_ALWAYS, "Generate rays filtering : " << this->rays.size());
  //   size_t workload = this->rays.size();
  //   processRayVector(this, this->rays, current_ray, this->rays_end,
  //   workload)();
  //   GVT_DEBUG(DBG_ALWAYS, "Filtered rays : " << this->rays.size());
  //   //
  // }

  // GVT::Concurrency::asyncExec::singleton->syncAll();

  /*! Gather buffers from all distributed nodes
   *
   * \param colorBuf Local color buffer
   * \param rays_traced number of rays traced
   *
   * \note Should be called at the end of the operator
   */
  // virtual void gatherFramebuffers(int rays_traced) {
  //   for (int i = 0; i < gvt::render::Attributes::rta->view.width *
  //                           gvt::render::Attributes::rta->view.height;
  //        ++i) {
  //     this->image.Add(i, colorBuf[i]);
  //   }
  // }

  /*! Communication with other nodes
   *
   * Implements the communication and scheduling strategy among nodes.
   *
   * \note Called from the operator
   *
   */
  // virtual bool SendRays() {
  //   GVT_ASSERT_BACKTRACE(false, "Not supported");
  //   return false;
  // }
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

template <class DomainType, class MPIW, class BSCHEDULER>
class Tracer : public TracerBase<MPIW> {
 public:
  Tracer(gvt::render::actor::RayVector& rays,
         gvt::render::data::scene::Image& image)
      : TracerBase<MPIW>(rays, image) {}

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
template <class DomainType, class MPIW, template <typename> class BSCHEDULER,
          class ISCHEDULER>
class Tracer<DomainType, MPIW, BSCHEDULER<ISCHEDULER>>
    : public TracerBase<MPIW> {
 public:
  Tracer(gvt::render::actor::RayVector& rays,
         gvt::render::data::scene::Image& image)
      : TracerBase<MPIW>(rays, image) {}

  virtual ~Tracer() {}
};
}
}
}

#endif /* GVT_RENDER_ALGORITHM_TRACER_BASE_H */
