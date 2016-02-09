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
//
// OptixMeshAdapter.cpp
//

#include "gvt/render/adapter/optix/data/OptixMeshAdapter.h"
#include "gvt/core/CoreContext.h"

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>

#include <gvt/core/schedule/TaskScheduling.h> // used for threads

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Mesh.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

#include <atomic>
#include <thread>
#include <future>

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>

// TODO: add logic for other packet sizes
#define GVT_OPTIX_PACKET_SIZE 4096

using namespace gvt::render::actor;
using namespace gvt::render::adapter::optix::data;
using namespace gvt::render::data::primitives;
using namespace gvt::core::math;

static std::atomic<size_t> counter(0);

// bool OptixMeshAdapter::init = false;

struct OptixRay {
  float origin[3];
  float t_min;
  float direction[3];
  float t_max;
  friend std::ostream &operator<<(std::ostream &os, const OptixRay &r) {
    return (os << "ray  o: " << r.origin[0] << ", " << r.origin[1] << ", " << r.origin[2] << " d: " << r.direction[0]
               << ", " << r.direction[1] << ", " << r.direction[2]);
  }
};

/// OptiX hit format
struct OptixHit {
  float t;
  int triangle_id;
  float u;
  float v;
  friend std::ostream &operator<<(std::ostream &os, const OptixHit &oh) {
    return (os << "hit  t: " << oh.t << " triID: " << oh.triangle_id);
  }
};

OptixContext *OptixContext::_singleton = new OptixContext();

OptixMeshAdapter::OptixMeshAdapter(gvt::core::DBNodeH node)
    : Adapter(node), packetSize(GVT_OPTIX_PACKET_SIZE), optix_context_(OptixContext::singleton()->context()) {
  // Get GVT mesh pointer
  GVT_ASSERT(optix_context_.isValid(), "Optix Context is not valid");
  Mesh *mesh = (Mesh *)node["ptr"].value().toULongLong();
  GVT_ASSERT(mesh, "OptixMeshAdapter: mesh pointer in the database is null");

  int numVerts = mesh->vertices.size();
  int numTris = mesh->faces.size();

  // Create Optix Prime Context

  // Use all CUDA devices, if multiple are present ignore the GPU driving the
  // display
  {
    std::vector<unsigned> activeDevices;
    int devCount = 0;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&devCount);
    GVT_ASSERT(devCount, "You choose optix render, but no cuda capable devices are present");

    for (int i = 0; i < devCount; i++) {
      cudaGetDeviceProperties(&prop, i);
      if (prop.kernelExecTimeoutEnabled == 0) activeDevices.push_back(i);
      // Oversubcribe the GPU
      packetSize = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    }
    if (!activeDevices.size()) {
      activeDevices.push_back(0);
    }
    optix_context_->setCudaDeviceNumbers(activeDevices);
  }

  // Setup the buffer to hold our vertices.
  //
  std::vector<float> vertices;
  std::vector<int> faces;

  vertices.resize(numVerts * 3);
  faces.resize(numTris * 3);

  const int offset_verts = 100; // numVerts / std::thread::hardware_concurrency();

  std::vector<std::future<void> > _tasks;

  // clang-format off
  for (int i = 0; i < numVerts; i += offset_verts) {
    _tasks.push_back(std::async(std::launch::async, [&](const int ii, const int end) {
                                                      for (int jj = ii; jj < end && jj < numVerts; jj++) {
                                                        vertices[jj * 3 + 0] = mesh->vertices[jj][0];
                                                        vertices[jj * 3 + 1] = mesh->vertices[jj][1];
                                                        vertices[jj * 3 + 2] = mesh->vertices[jj][2];
                                                      }
                                                    },
                                i, i + offset_verts));
  }
  // clang-format on

  const int offset_tris = 100; // numTris / std::thread::hardware_concurrency();

  // clang-format off
  for (int i = 0; i < numTris; i += offset_tris) {
    _tasks.push_back(std::async(std::launch::async, [&](const int ii, const int end) {
                                                      for (int jj = ii; jj < end && jj < numTris; jj++) {
                                                        gvt::render::data::primitives::Mesh::Face f = mesh->faces[jj];
                                                        faces[jj * 3 + 0] = f.get<0>();
                                                        faces[jj * 3 + 1] = f.get<1>();
                                                        faces[jj * 3 + 2] = f.get<2>();
                                                      }
                                                    },
                                i, i + offset_tris));
  }
  // clang-format on

  for (auto &f : _tasks) f.wait();

  // Create and setup vertex buffer
  ::optix::prime::BufferDesc vertices_desc;
  vertices_desc = optix_context_->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, &vertices[0]);

  GVT_ASSERT(vertices_desc.isValid(), "Vertices are not valid");
  vertices_desc->setRange(0, vertices.size() / 3);
  vertices_desc->setStride(sizeof(float) * 3);

  // Create and setup triangle buffer
  ::optix::prime::BufferDesc indices_desc;
  indices_desc = optix_context_->createBufferDesc(RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, &faces[0]);

  GVT_ASSERT(indices_desc.isValid(), "Indices are not valid");
  indices_desc->setRange(0, faces.size() / 3);
  indices_desc->setStride(sizeof(int) * 3);

  // Create an Optix model.
  optix_model_ = optix_context_->createModel();
  GVT_ASSERT(optix_model_.isValid(), "Model is not valid");
  optix_model_->setTriangles(indices_desc, vertices_desc);
  optix_model_->update(RTP_MODEL_HINT_ASYNC);
  optix_model_->finish();
}

OptixMeshAdapter::~OptixMeshAdapter() {}

struct OptixParallelTrace {
  /**
   * Pointer to OptixMeshAdapter to get Embree scene information
   */
  gvt::render::adapter::optix::data::OptixMeshAdapter *adapter;

  /**
   * Shared ray list used in the current trace() call
   */
  gvt::render::actor::RayVector &rayList;

  /**
   * Shared outgoing ray list used in the current trace() call
   */
  gvt::render::actor::RayVector &moved_rays;

  /**
   * Number of rays to work on at once [load balancing].
   */
  const size_t workSize;

  /**
   * Index into the shared `rayList`.  Atomically incremented to 'grab'
   * the next set of rays.
   */
  std::atomic<size_t> &sharedIdx;

  /**
   * DB reference to the current instance
   */
  gvt::core::DBNodeH instNode;

  /**
   * Stored transformation matrix in the current instance
   */
  const gvt::core::math::AffineTransformMatrix<float> *m;

  /**
   * Stored inverse transformation matrix in the current instance
   */
  const gvt::core::math::AffineTransformMatrix<float> *minv;

  /**
   * Stored upper33 inverse matrix in the current instance
   */
  const gvt::core::math::Matrix3f *normi;

  /**
   * Stored transformation matrix in the current instance
   */
  const std::vector<gvt::render::data::scene::Light *> &lights;

  /**
   * Count the number of rays processed by the current trace() call.
   *
   * Used for debugging purposes
   */
  std::atomic<size_t> &counter;

  /**
   * Thread local outgoing ray queue
   */
  gvt::render::actor::RayVector localDispatch;

  /**
   * List of shadow rays to be processed
   */
  gvt::render::actor::RayVector shadowRays;

  /**
   * Size of Embree packet
   */
  size_t packetSize; // TODO: later make this configurable

  const size_t begin, end;

  /**
   * Construct a OptixParallelTrace struct with information needed for the
   * thread
   * to do its tracing
   */
  OptixParallelTrace(gvt::render::adapter::optix::data::OptixMeshAdapter *adapter,
                     gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                     std::atomic<size_t> &sharedIdx, const size_t workSize, gvt::core::DBNodeH instNode,
                     gvt::core::math::AffineTransformMatrix<float> *m,
                     gvt::core::math::AffineTransformMatrix<float> *minv, gvt::core::math::Matrix3f *normi,
                     std::vector<gvt::render::data::scene::Light *> &lights, std::atomic<size_t> &counter,
                     const size_t begin, const size_t end)
      : adapter(adapter), rayList(rayList), moved_rays(moved_rays), sharedIdx(sharedIdx), workSize(workSize),
        instNode(instNode), m(m), minv(minv), normi(normi), lights(lights), counter(counter),
        packetSize(adapter->getPacketSize()), begin(begin), end(end) {}

  /**
   * Convert a set of rays from a vector into a prepOptixRays ray packet.
   *
   * \param optixrays     reference of optix ray datastructure
   * \param valid         aligned array of 4 ints to mark valid rays
   * \param resetValid    if true, reset the valid bits, if false, re-use old
   * valid to know which to convert
   * \param packetSize    number of rays to convert
   * \param rays          vector of rays to read from
   * \param startIdx      starting point to read from in `rays`
   */
  void prepOptixRays(std::vector<OptixRay> &optixrays, std::vector<bool> &valid, const bool resetValid,
                     const int localPacketSize, const gvt::render::actor::RayVector &rays, const size_t startIdx) {
    for (int i = 0; i < localPacketSize; i++) {
      if (valid[i]) {
        const Ray &r = rays[startIdx + i];
        const auto origin = (*minv) * r.origin; // transform ray to local space
        const auto direction = (*minv) * r.direction;
        OptixRay optix_ray;
        optix_ray.origin[0] = origin[0];
        optix_ray.origin[1] = origin[1];
        optix_ray.origin[2] = origin[2];
        optix_ray.t_min = 0;
        optix_ray.direction[0] = direction[0];
        optix_ray.direction[1] = direction[1];
        optix_ray.direction[2] = direction[2];
        optix_ray.t_max = FLT_MAX;
        optixrays[i] = optix_ray;
      }
    }
  }

  /**
   * Generate shadow rays for a given ray
   *
   * \param r ray to generate shadow rays for
   * \param normal calculated normal
   * \param primId primitive id for shading
   * \param mesh pointer to mesh struct [TEMPORARY]
   */
  void generateShadowRays(const gvt::render::actor::Ray &r, const gvt::core::math::Vector4f &normal, int primID,
                          gvt::render::data::primitives::Mesh *mesh) {
    for (gvt::render::data::scene::Light *light : lights) {
      GVT_ASSERT(light, "generateShadowRays: light is null for some reason");
      // Try to ensure that the shadow ray is on the correct side of the
      // triangle.
      // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
      // Using about 8 * ULP(t).
      const float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
      const float t_shadow = multiplier * r.t;

      const Point4f origin = r.origin + r.direction * t_shadow;
      const Vector4f dir = light->position - origin;
      const float t_max = dir.length();

      // note: ray copy constructor is too heavy, so going to build it manually
      shadowRays.push_back(Ray(r.origin + r.direction * t_shadow, dir, r.w, Ray::SHADOW, r.depth));

      Ray &shadow_ray = shadowRays.back();
      shadow_ray.t = r.t;
      shadow_ray.id = r.id;
      shadow_ray.t_max = t_max;

      // FIXME: remove dependency on mesh->shadeFace
      gvt::render::data::Color c = mesh->shadeFace(primID, shadow_ray, normal, light);
      // gvt::render::data::Color c = adapter->getMesh()->mat->shade(shadow_ray,
      // normal, lights[lindex]);
      shadow_ray.color = GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
    }
  }

  /**
   * Test occlusion for stored shadow rays.  Add missed rays
   * to the dispatch queue.
   */
  void traceShadowRays() {
    ::optix::prime::Query query = adapter->getScene()->createQuery(RTP_QUERY_TYPE_CLOSEST);
    if (!query.isValid()) return;

    for (size_t idx = 0; idx < shadowRays.size(); idx += packetSize) {
      const size_t localPacketSize = (idx + packetSize > shadowRays.size()) ? (shadowRays.size() - idx) : packetSize;
      std::vector<OptixRay> optix_rays(localPacketSize);
      std::vector<OptixHit> hits(localPacketSize);

      std::vector<bool> valid(localPacketSize);
      std::fill(valid.begin(), valid.end(), true);

      prepOptixRays(optix_rays, valid, true, localPacketSize, shadowRays, idx);

      query->setRays(optix_rays.size(), RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_HOST,
                     &optix_rays[0]);

      // Create and pass hit results in an Optix friendly format.
      query->setHits(hits.size(), RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_HOST, &hits[0]);

      // Execute our query and wait for it to finish.
      query->execute(RTP_QUERY_HINT_ASYNC);
      query->finish();
      GVT_ASSERT(query.isValid(), "Something went wrong.");

      for (int i = hits.size() - 1; i >= 0; --i) {
        if (hits[i].triangle_id < 0) {
          // ray is valid, but did not hit anything, so add to dispatch queue
          localDispatch.push_back(shadowRays[idx + i]);
        }
      }
    }
    shadowRays.clear();
  }

  /**
   * Trace function.
   *
   * Loops through rays in `rayList`, converts them to embree format, and traces
   * against embree's scene
   *
   * Threads work on rays in chunks of `workSize` units.  An atomic add on
   * `sharedIdx` distributes
   * the ranges of rays to work on.
   *
   * After getting a chunk of rays to work with, the adapter loops through in
   * sets of `packetSize`.  Right
   * now this supports a 4 wide packet [Embree has support for 8 and 16 wide
   * packets].
   *
   * The packet is traced and re-used until all of the 4 rays and their
   * secondary rays have been traced to
   * completion.  Shadow rays are added to a queue and are tested after each
   * intersection test.
   *
   * The `while(validRayLeft)` loop behaves something like this:
   *
   * r0: primary -> secondary -> secondary -> ... -> terminated
   * r1: primary -> secondary -> secondary -> ... -> terminated
   * r2: primary -> secondary -> secondary -> ... -> terminated
   * r3: primary -> secondary -> secondary -> ... -> terminated
   *
   * It is possible to get diverging packets such as:
   *
   * r0: primary   -> secondary -> terminated
   * r1: secondary -> secondary -> terminated
   * r2: shadow    -> terminated
   * r3: primary   -> secondary -> secondary -> secondary -> terminated
   *
   * TODO: investigate switching terminated rays in the vector with active rays
   * [swap with ones at the end]
   *
   * Terminated above means:
   * - shadow ray hits object and is occluded
   * - primary / secondary ray miss and are passed out of the queue
   *
   * After a packet is completed [including its generated rays], the system
   * moves on * to the next packet
   * in its chunk. Once a chunk is completed, the thread increments `sharedIdx`
   * again to get more work.
   *
   * If `sharedIdx` grows to be larger than the incoming ray size, then the
   * thread is complete.
   */
  void operator()() {
#ifdef GVT_USE_DEBUG
    boost::timer::auto_cpu_timer t_functor("OptixMeshAdapter: thread trace time: %w\n");
#endif

    // TODO: don't use gvt mesh. need to figure out way to do per-vertex-normals
    // and shading calculations
    auto mesh = (Mesh *)instNode["meshRef"].deRef()["ptr"].value().toULongLong();

    ::optix::prime::Model scene = adapter->getScene();

    localDispatch.reserve((end - begin) * 2);

    // there is an upper bound on the nubmer of shadow rays generated per embree
    // packet
    // its embree_packetSize * lights.size()
    shadowRays.reserve(packetSize * lights.size());

    while (sharedIdx < end) {
#ifdef GVT_USE_DEBUG
// boost::timer::auto_cpu_timer t_outer_loop("OptixMeshAdapter: workSize rays
// traced: %w\n");
#endif

      // atomically get the next chunk range
      size_t workStart = sharedIdx.fetch_add(workSize);

      // have to double check that we got the last valid chunk range
      if (workStart > end) {
        break;
      }

      // calculate the end work range
      size_t workEnd = workStart + workSize;
      if (workEnd > end) {
        workEnd = end;
      }

      for (size_t localIdx = workStart; localIdx < workEnd; localIdx += packetSize) {
        // this is the local packet size. this might be less than the main
        // packetSize due to uneven amount of rays
        const size_t localPacketSize = (localIdx + packetSize > workEnd) ? (workEnd - localIdx) : packetSize;

        // trace a packet of rays, then keep tracing the generated secondary
        // rays to completion
        // tracks to see if there are any valid rays left in the packet, if so,
        // keep tracing
        // NOTE: perf issue: this will cause smaller and smaller packets to be
        // traced at a time - need to track to see effects
        bool validRayLeft = true;

        // the first time we enter the loop, we want to reset the valid boolean
        // list that was
        // modified with the previous packet
        bool resetValid = true;

        std::vector<bool> valid(localPacketSize);
        std::vector<OptixRay> optix_rays(localPacketSize);
        std::vector<OptixHit> hits(localPacketSize);

        std::fill(valid.begin(), valid.end(), true);

        while (validRayLeft) {
          validRayLeft = false;

          prepOptixRays(optix_rays, valid, false, localPacketSize, rayList, localIdx);
          ::optix::prime::Query query = adapter->getScene()->createQuery(RTP_QUERY_TYPE_CLOSEST);

          query->setRays(optix_rays.size(), RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_HOST,
                         &optix_rays[0]);

          // Create and pass hit results in an Optix friendly format.
          query->setHits(hits.size(), RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_HOST, &hits[0]);

          // Execute our query and wait faor it to finish.
          query->execute(RTP_QUERY_HINT_ASYNC);
          query->finish();

          GVT_ASSERT(query.isValid(), "Something went wrong.");
          for (size_t pi = 0; pi < localPacketSize; pi++) {
            if (valid[pi]) {
              // counter++; // tracks rays processed [atomic]
              auto &r = rayList[localIdx + pi];
              if (hits[pi].triangle_id >= 0) {
                // ray has hit something
                // shadow ray hit something, so it should be dropped
                if (r.type == gvt::render::actor::Ray::SHADOW) {
                  continue;
                }

                float t = hits[pi].t;
                r.t = t;

                Vector4f manualNormal;
                {
                  const int triangle_id = hits[pi].triangle_id;
#ifndef FLAT_SHADING
                  const float u = hits[pi].u;
                  const float v = hits[pi].v;
                  const Mesh::FaceToNormals &normals = mesh->faces_to_normals[triangle_id]; // FIXME: need to
                                                                                            // figure out
                                                                                            // to store
                  // `faces_to_normals`
                  // list
                  const Vector4f &a = mesh->normals[normals.get<0>()];
                  const Vector4f &b = mesh->normals[normals.get<1>()];
                  const Vector4f &c = mesh->normals[normals.get<2>()];
                  manualNormal = a * u + b * v + c * (1.0f - u - v);

                  manualNormal = (*normi) * (gvt::core::math::Vector3f)manualNormal;
                  manualNormal.normalize();
#else
                  int I = mesh->faces[triangle_id].get<0>();
                  int J = mesh->faces[triangle_id].get<1>();
                  int K = mesh->faces[triangle_id].get<2>();

                  Vector4f a = mesh->vertices[I];
                  Vector4f b = mesh->vertices[J];
                  Vector4f c = mesh->vertices[K];
                  Vector4f u = b - a;
                  Vector4f v = c - a;
                  Vector4f normal;
                  normal.n[0] = u.n[1] * v.n[2] - u.n[2] * v.n[1];
                  normal.n[1] = u.n[2] * v.n[0] - u.n[0] * v.n[2];
                  normal.n[2] = u.n[0] * v.n[1] - u.n[1] * v.n[0];
                  normal.n[3] = 0.0f;
                  manualNormal = normal.normalize();
#endif
                }
                const Vector4f &normal = manualNormal;

                // reduce contribution of the color that the shadow rays get
                if (r.type == gvt::render::actor::Ray::SECONDARY) {
                  t = (t > 1) ? 1.f / t : t;
                  r.w = r.w * t;
                }

                generateShadowRays(r, normal, hits[pi].triangle_id, mesh);
                int ndepth = r.depth - 1;
                float p = 1.f - (float(rand()) / RAND_MAX);
                // replace current ray with generated secondary ray
                if (ndepth > 0 && r.w > p) {
                  r.domains.clear();
                  r.type = gvt::render::actor::Ray::SECONDARY;
                  const float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon(); // TODO: move
                                                                                                 // out
                  // somewhere /
                  // make static
                  const float t_secondary = multiplier * r.t;
                  r.origin = r.origin + r.direction * t_secondary;

                  r.setDirection(mesh->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());

                  r.w = r.w * (r.direction * normal);
                  r.depth = ndepth;
                  validRayLeft = true; // we still have a valid ray in the packet to trace
                } else {
                  valid[pi] = false;
                }
              } else {
                // ray is valid, but did not hit anything, so add to dispatch
                // queue and disable it
                localDispatch.push_back(r);
              }
            }
          }

          // trace shadow rays generated by the packet
          traceShadowRays();
        }
      }
    }

#ifdef GVT_USE_DEBUG
    size_t shadow_count = 0;
    size_t primary_count = 0;
    size_t secondary_count = 0;
    size_t other_count = 0;
    for (auto &r : localDispatch) {
      switch (r.type) {
      case gvt::render::actor::Ray::SHADOW:
        shadow_count++;
        break;
      case gvt::render::actor::Ray::PRIMARY:
        primary_count++;
        break;
      case gvt::render::actor::Ray::SECONDARY:
        secondary_count++;
        break;
      default:
        other_count++;
        break;
      }
    }
    GVT_DEBUG(DBG_ALWAYS, "Local dispatch : " << localDispatch.size() << ", types: primary: " << primary_count
                                              << ", shadow: " << shadow_count << ", secondary: " << secondary_count
                                              << ", other: " << other_count);
#endif

    // copy localDispatch rays to outgoing rays queue
    boost::unique_lock<boost::mutex> moved(adapter->_outqueue);
    moved_rays.insert(moved_rays.end(), localDispatch.begin(), localDispatch.end());
    moved.unlock();
  }
};

void OptixMeshAdapter::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                             gvt::core::DBNodeH instNode, size_t _begin, size_t _end) {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t_functor("OptixMeshAdapter: trace time: %w\n");
#endif

  if (_end == 0) _end = rayList.size();

  this->begin = _begin;
  this->end = _end;

  std::atomic<size_t> sharedIdx(begin); // shared index into rayList
  const size_t numThreads = std::thread::hardware_concurrency();
  gvt::core::schedule::asyncExec::instance()->numThreads;
  const size_t workSize =
      std::max((size_t)1, std::min((size_t)GVT_OPTIX_PACKET_SIZE, (size_t)((end - begin) / numThreads)));

  const size_t numworkers = std::max((size_t)1, std::min((size_t)numThreads, (size_t)((end - begin) / workSize)));

  // pull out information out of the database, create local structs that will be
  // passed into the parallel struct
  gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

  // pull out instance transform data
  GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: getting instance transform data");
  gvt::core::math::AffineTransformMatrix<float> *m =
      (gvt::core::math::AffineTransformMatrix<float> *)instNode["mat"].value().toULongLong();
  gvt::core::math::AffineTransformMatrix<float> *minv =
      (gvt::core::math::AffineTransformMatrix<float> *)instNode["matInv"].value().toULongLong();
  gvt::core::math::Matrix3f *normi = (gvt::core::math::Matrix3f *)instNode["normi"].value().toULongLong();

  //
  // TODO: wrap this db light array -> class light array conversion in some sort
  // of helper function
  // `convertLights`: pull out lights list and convert into gvt::Lights format
  // for now
  auto lightNodes = root["Lights"].getChildren();
  std::vector<gvt::render::data::scene::Light *> lights;
  lights.reserve(2);
  for (auto lightNode : lightNodes) {
    auto color = lightNode["color"].value().toVector4f();

    if (lightNode.name() == std::string("PointLight")) {
      auto pos = lightNode["position"].value().toVector4f();
      lights.push_back(new gvt::render::data::scene::PointLight(pos, color));
    } else if (lightNode.name() == std::string("AmbientLight")) {
      lights.push_back(new gvt::render::data::scene::AmbientLight(color));
    }
  }
  GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: converted " << lightNodes.size()
                                                       << " light nodes into structs: size: " << lights.size());
  // end `convertLights`
  //

  // # notes
  // 20150819-2344: alim: boost threads vs c++11 threads don't seem to have much
  // of a runtime difference
  // - I was not re-using the c++11 threads though, was creating new ones every
  // time

  std::vector<std::future<void> > _tasks;

  for (size_t rc = 0; rc < numworkers; ++rc) {
    _tasks.push_back(std::async(std::launch::deferred, [&]() {
      OptixParallelTrace(this, rayList, moved_rays, sharedIdx, workSize, instNode, m, minv, normi, lights, counter,
                         begin, end)();
    }));
  }

  for (auto &t : _tasks) t.wait();

  // GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: Processed rays: " << counter);
  GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: Forwarding rays: " << moved_rays.size());

  // rayList.clear();
}
