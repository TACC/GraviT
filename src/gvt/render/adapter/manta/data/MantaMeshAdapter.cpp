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
// MantaMeshAdapter.cpp
//

#include "gvt/render/adapter/manta/data/MantaMeshAdapter.h"

#include "gvt/core/CoreContext.h"

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/manta/data/Transforms.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

// Manta includes
#include <Core/Math/CheapRNG.h>
#include <Core/Math/MT_RNG.h>
#include <Engine/Shadows/HardShadows.h>
#include <Engine/Shadows/NoShadows.h>
#include <Interface/Primitive.h>
#include <Model/AmbientLights/AmbientOcclusionBackground.h>
#include <Model/AmbientLights/ArcAmbient.h>
#include <Model/AmbientLights/ConstantAmbient.h>
#include <Model/AmbientLights/EyeAmbient.h>
#include <Model/Primitives/Cube.h>
// end Manta includes

#include <atomic>
#include <thread>

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>

using namespace gvt::render::actor;
using namespace gvt::render::adapter::manta::data;
// using namespace gvt::render::adapter::manta::data::domain;
// using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;

static std::atomic<size_t> counter(0);
static boost::atomic<size_t> counter_b(0);

// MantaMeshAdapter::MantaMeshAdapter(GeometryDomain* domain) : GeometryDomain(*domain)
MantaMeshAdapter::MantaMeshAdapter(gvt::render::data::prmitives::Mesh *mesh) : Adapter(mesh) {
  GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: converting mesh node " << node.UUID().toString());

  // Mesh *mesh = (Mesh *)node["ptr"].value().toULongLong();

  GVT_ASSERT(mesh, "MantaMeshAdapter: mesh pointer in the database is null");

  mesh->generateNormals();

  // Transform mesh
  // meshManta = transform<Mesh*, Manta::Mesh*>(this->mesh);
  meshManta = transform<Mesh *, Manta::Mesh *>(mesh);
  Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
  Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;

  // Create BVH
  as = new Manta::DynBVH();
  as->setGroup(meshManta);

  // Create Manta
  static Manta::MantaInterface *rtrt = Manta::createManta();

  // Create light set
  Manta::LightSet *lights = new Manta::LightSet();
  lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));

  // Create ambient light
  Manta::AmbientLight *ambient;
  ambient = new Manta::AmbientOcclusionBackground(Manta::Color::white() * 0.5, 1, 36);

  // Create context
  Manta::PreprocessContext context(rtrt, 0, 1, lights);
  std::cout << "context.global_lights : " << context.globalLights << std::endl;
  material->preprocess(context);
  as->preprocess(context);

  // Select algorithm
  Manta::ShadowAlgorithm *shadows;
  shadows = new Manta::HardShadows();
  Manta::Scene *scene = new Manta::Scene();

  scene->setLights(lights);
  scene->setObject(as);
  Manta::RandomNumberGenerator *rng = NULL;
  Manta::CheapRNG::create(rng);

  rContext =
      new Manta::RenderContext(rtrt, 0, 0 /*proc*/, 1 /*workersAnimandImage*/, 0 /*animframestate*/, 0 /*loadbalancer*/,
                               0 /*pixelsampler*/, 0 /*renderer*/, shadows /*shadowAlgorithm*/, 0 /*camera*/,
                               scene /*scene*/, 0 /*thread_storage*/, rng /*rngs*/, 0 /*samplegenerator*/);
}

// MantaMeshAdapter::MantaMeshAdapter(std::string filename, glm::mat4 m)
// : gvt::render::data::domain::GeometryDomain(filename, m)
// {
//
// }
//
// MantaMeshAdapter::MantaMeshAdapter(const MantaMeshAdapter& other)
// : gvt::render::data::domain::GeometryDomain(other)
// {
// }

MantaMeshAdapter::~MantaMeshAdapter() {
  // GeometryDomain::~GeometryDomain();
}

bool MantaMeshAdapter::load() {
#if 0
    if (domainIsLoaded()) return true;

    GVT::Domain::GeometryDomain::load();
    Manta::Mesh* mesh = GVT::Data::transform<GVT::Data::Mesh*, Manta::Mesh*>(this->mesh);


    Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
    Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
    as = new Manta::DynBVH();
    as->setGroup(mesh);

    static Manta::MantaInterface* rtrt = Manta::createManta();
    Manta::LightSet* lights = new Manta::LightSet();
    lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));
    Manta::AmbientLight* ambient;
    ambient = new Manta::AmbientOcclusionBackground(Manta::Color::white()*0.5, 1, 36);
    Manta::PreprocessContext context(rtrt, 0, 1, lights);
    std::cout << "context.global_lights : " << context.globalLights << std::endl;
    material->preprocess(context);
    as->preprocess(context);
    Manta::ShadowAlgorithm* shadows;
    shadows = new Manta::HardShadows();
    Manta::Scene* scene = new Manta::Scene();


    scene->setLights(lights);
    scene->setObject(as);
    Manta::RandomNumberGenerator* rng = NULL;
    Manta::CheapRNG::create(rng);

    rContext = new Manta::RenderContext(rtrt, 0, 0/*proc*/, 1/*workersAnimandImage*/,
            0/*animframestate*/,
            0/*loadbalancer*/, 0/*pixelsampler*/, 0/*renderer*/, shadows/*shadowAlgorithm*/, 0/*camera*/, scene/*scene*/, 0/*thread_storage*/, rng/*rngs*/, 0/*samplegenerator*/);

#endif
  return true;
}

void MantaMeshAdapter::free() {}

struct parallelTrace {
  // gvt::render::adapter::manta::data::domain::MantaDomain* dom;
  gvt::render::adapter::manta::data::MantaMeshAdapter *adapter;
  gvt::render::actor::RayVector &rayList;
  gvt::render::actor::RayVector &moved_rays;
  const size_t workSize;
  const glm::mat4 *m;
  const glm::mat4 *minv;
  const glm::mat3 *normi;
  const std::vector<gvt::render::data::scene::Light *> &lights;
  const gvt::render::data::primitives::Mesh *mesh;

  boost::atomic<size_t> &counter;

  parallelTrace(
      // gvt::render::adapter::manta::data::domain::MantaDomain* dom,
      gvt::render::adapter::manta::data::MantaMeshAdapter *adapter, gvt::render::actor::RayVector &rayList,
      gvt::render::actor::RayVector &moved_rays, const size_t workSize, boost::atomic<size_t> &counter, glm::mat4 *m,
      glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
      gvt::render::data::primitives::Mesh *mesh)
      : adapter(adapter), rayList(rayList), moved_rays(moved_rays), workSize(workSize), counter(counter), m(m),
        minv(minv), normi(normi), lights(lights), mesh(mesh) {}

  void operator()() {
    const size_t maxPacketSize = 64;

    Manta::RenderContext &renderContext = *adapter->getRenderContext();

    gvt::render::actor::RayVector rayPacket;
    gvt::render::actor::RayVector localQueue;
    gvt::render::actor::RayVector localDispatch;

    Manta::RayPacketData rpData;

    localQueue.reserve(workSize * 2);
    localDispatch.reserve(rayList.size() * 2);

    //                GVT_DEBUG(DBG_ALWAYS, dom->meshManta->vertices.size());
    //                GVT_DEBUG(DBG_ALWAYS, dom->meshManta->vertex_indices.size());
    //
    //                BOOST_FOREACH(int i, dom->meshManta->vertex_indices) {
    //                    GVT_DEBUG(DBG_ALWAYS, i);
    //                }

    while (!rayList.empty()) {
      std::unique_lock<std::mutex> queue(adapter->_inqueue);
      std::size_t range = std::min(workSize, rayList.size());
      localQueue.assign(rayList.begin(), rayList.begin() + range);
      rayList.erase(rayList.begin(), rayList.begin() + range);
      queue.unlock();

      GVT_DEBUG(DBG_ALWAYS, "Got " << localQueue.size() << " rays");
      while (!localQueue.empty()) {
        rayPacket.clear();

        while (rayPacket.size() < 64 && !localQueue.empty()) {
          rayPacket.push_back(localQueue.back());
          localQueue.pop_back();
        }

        Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, rayPacket.size(), 0,
                               Manta::RayPacket::NormalizedDirections);
        for (int i = 0; i < rayPacket.size(); i++) {
          Ray ray(rayPacket[i]);
          ray.origin = (*minv) * ray.origin; // transform ray to local space
          ray.direction = (*minv) * ray.direction;
          // mRays.setRay(i, transform<Ray, Manta::Ray>(dom->toLocal(rayPacket[i])));
          mRays.setRay(i, transform<Ray, Manta::Ray>(ray));
          // mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(rayPacket[i]));
        }

        mRays.resetHits();
        adapter->getAccelStruct()->intersect(renderContext, mRays);
        mRays.computeNormals<false>(renderContext);

        //                        GVT_DEBUG(DBG_ALWAYS,"Process packet");

        for (int pindex = 0; pindex < rayPacket.size(); pindex++) {
          if (mRays.wasHit(pindex)) {
            //                                GVT_DEBUG(DBG_ALWAYS,"Ray has hit " << pindex);
            if (rayPacket[pindex].type == gvt::render::actor::Ray::SHADOW) {
              //                                    GVT_DEBUG(DBG_ALWAYS,"Process ray in shadow");
              continue;
            }

            float t = mRays.getMinT(pindex);
            rayPacket[pindex].t = t;

            // glm::vec3 normal =
            // dom->toWorld(gvt::render::adapter::manta::data::transform<Manta::Vector,
            // glm::vec3>(mRays.getNormal(pindex)));
            glm::vec3 normal =
                glm::normalize((*normi) * (glm::vec3)(transform<Manta::Vector, glm::vec3>(mRays.getNormal(pindex))));

            if (rayPacket[pindex].type == gvt::render::actor::Ray::SECONDARY) {
              t = (t > 1) ? 1.f / t : t;
              rayPacket[pindex].w = rayPacket[pindex].w * t;
            }

            // std::vector<gvt::render::data::scene::Light*> lights = dom->getLights();
            // for (int lindex = 0; lindex < lights.size(); lindex++)
            for (gvt::render::data::scene::Light *light : lights) {
              gvt::render::actor::Ray ray(rayPacket[pindex]);
              ray.type = gvt::render::actor::Ray::SHADOW;
              ray.origin = ray.origin + ray.direction * ray.t;
              ray.setDirection(light->position - ray.origin);
              gvt::render::data::Color c = mesh->mat->shade(ray, normal, light, light->position);
              ray.color = c;
              // ray.color = GVT_COLOR_ACCUM(1.f, 1.0, c[1], c[2], 1.f);
              localQueue.push_back(ray);
            }

            int ndepth = rayPacket[pindex].depth - 1;

            float p = 1.f - (float(rand()) / RAND_MAX);

            if (ndepth > 0 && rayPacket[pindex].w > p) {
              gvt::render::actor::Ray ray(rayPacket[pindex]);
              ray.type = gvt::render::actor::Ray::SECONDARY;
              ray.origin = ray.origin + ray.direction * ray.t;
              ray.setDirection(mesh->getMaterial()->CosWeightedRandomHemisphereDirection2(normal));
              ray.w = ray.w * (ray.direction * normal);
              ray.depth = ndepth;
              localQueue.push_back(ray);
            }
            // counter++;
            continue;
          }
          // counter++;
          // GVT_DEBUG(DBG_ALWAYS,"Add to local dispatch");
          localDispatch.push_back(rayPacket[pindex]);
        }
      }
    }

    GVT_DEBUG(DBG_ALWAYS, "Local dispatch : " << localDispatch.size());

    std::unique_lock<std::mutex> moved(adapter->_outqueue);
    moved_rays.insert(moved_rays.begin(), localDispatch.begin(), localDispatch.end());
    moved.unlock();
  }
};

// TODO: Warning! the following trace structure is still buggy.
// Random stripes on the final image are seen.
struct mantaParallelTrace {
  /**
   * Pointer to MantaMeshAdapter to get Embree scene information
   */
  gvt::render::adapter::manta::data::MantaMeshAdapter *adapter;

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
  const glm::mat4 *m;

  /**
   * Stored inverse transformation matrix in the current instance
   */
  const glm::mat4 *minv;

  /**
   * Stored upper33 inverse matrix in the current instance
   */
  const glm::mat3 *normi;

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
   * Size of Manta packet
   */
  const size_t packetSize; // TODO: later make this configurable

  /**
   * Construct a mantaParallelTrace struct with information needed for the thread
   * to do its tracing
   */
  mantaParallelTrace(gvt::render::adapter::manta::data::MantaMeshAdapter *adapter,
                     gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                     std::atomic<size_t> &sharedIdx, const size_t workSize, gvt::core::DBNodeH instNode, glm::mat4 *m,
                     glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
                     std::atomic<size_t> &counter)
      : adapter(adapter), rayList(rayList), moved_rays(moved_rays), sharedIdx(sharedIdx), workSize(workSize), m(m),
        minv(minv), normi(normi), lights(lights), counter(counter),
        packetSize(64) // TODO: packetSize(adapter->getPacketSize())
  {}

  /**
   * Convert a set of rays from a vector into a RTCRay4 ray packet.
   *
   * \param ray4          reference of RTCRay4 struct to write to
   * \param valid         aligned array of 4 ints to mark valid rays
   * \param resetValid    if true, reset the valid bits, if false, re-use old valid to know which to convert
   * \param packetSize    number of rays to convert
   * \param rays          vector of rays to read from
   * \param startIdx      starting point to read from in `rays`
   */
  // void prepRTCRay4(RTCRay4 &ray4, int valid[4], const bool resetValid, const int localPacketSize,
  // gvt::render::actor::RayVector& rays, const size_t startIdx) {
  void prepRayPacket(Manta::RayPacket &mRays, int *valid, const bool resetValid, const int localPacketSize,
                     gvt::render::actor::RayVector &rays, const size_t startIdx) {
    // reset valid to match the number of active rays in the packet
    if (resetValid) {
      for (int i = 0; i < localPacketSize; i++) {
        valid[i] = -1;
      }
      for (int i = localPacketSize; i < packetSize; i++) {
        valid[i] = 0;
      }
    }

    // convert packetSize rays into manta's RayPacket class
    for (int i = 0; i < packetSize; ++i) {
      if (valid[i]) {
        Ray ray(rays[startIdx + i]);
        ray.origin = (*minv) * ray.origin; // transform ray to local space
        ray.direction = (*minv) * ray.direction;
        mRays.setRay(i, transform<Ray, Manta::Ray>(ray));
      } else {
        mRays.maskRay(i);
      }
    }
    mRays.resetHits();
  }

  /**
   * Generate shadow rays for a given ray
   *
   * \param r ray to generate shadow rays for
   * \param normal calculated normal
   * \param primId primitive id for shading
   * \param mesh pointer to mesh struct [TEMPORARY]
   */
  void generateShadowRays(const gvt::render::actor::Ray &r, const glm::vec3 &normal,
                          gvt::render::data::primitives::Mesh *mesh) {
    for (gvt::render::data::scene::Light *light : lights) {
      GVT_ASSERT(light, "generateShadowRays: light is null for some reason");
      const glm::vec3 origin = r.origin + r.direction * r.t;
      glm::vec3 dir = light->position - origin;
      const float t_max = glm::length(dir);
      shadowRays.push_back(Ray(origin, glm::normalize(dir), r.w, Ray::SHADOW, r.depth));
      Ray &shadow_ray = shadowRays.back();
      shadow_ray.t = r.t;
      shadow_ray.id = r.id;
      shadow_ray.t_max = t_max;
      gvt::render::data::Color c = mesh->mat->shade(r, normal, light, light->position);
      shadow_ray.color = c; // GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
    }
  }

  /**
   * Test occlusion for stored shadow rays.  Add missed rays
   * to the dispatch queue.
   */
  void traceShadowRays() {

    int valid[64] = { 0 };
    Manta::RayPacketData rpData;
    Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, packetSize, 0,
                           Manta::RayPacket::NormalizedDirections);

    Manta::RenderContext &renderContext = *adapter->getRenderContext();

    for (size_t idx = 0; idx < shadowRays.size(); idx += packetSize) {
      const size_t localPacketSize = (idx + packetSize > shadowRays.size()) ? (shadowRays.size() - idx) : packetSize;

      // create a shadow packet and trace with rtcOccluded
      prepRayPacket(mRays, valid, true, localPacketSize, shadowRays, idx);
      adapter->getAccelStruct()->intersect(renderContext, mRays);

      for (size_t pi = 0; pi < localPacketSize; pi++) {
        if (valid[pi] && !mRays.wasHit(pi)) {
          // if(!mRays.wasHit(pi)) {
          // ray is valid, but did not hit anything, so add to dispatch queue
          localDispatch.push_back(shadowRays[idx + pi]);
        }
      }
    }
    shadowRays.clear();
  }

  /**
   * Trace function.
   *
   * Loops through rays in `rayList`, converts them to manta format, and traces against manta's scene
   *
   * Threads work on rays in chunks of `workSize` units.  An atomic add on `sharedIdx` distributes
   * the ranges of rays to work on.
   *
   * After getting a chunk of rays to work with, the adapter loops through in sets of `packetSize`.  Right
   * now this supports a 4 wide packet [Embree has support for 8 and 16 wide packets].
   *
   * The packet is traced and re-used until all of the 4 rays and their secondary rays have been traced to
   * completion.  Shadow rays are added to a queue and are tested after each intersection test.
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
   * TODO: investigate switching terminated rays in the vector with active rays [swap with ones at the end]
   *
   * Terminated above means:
   * - shadow ray hits object and is occluded
   * - primary / secondary ray miss and are passed out of the queue
   *
   * After a packet is completed [including its generated rays], the system moves on to the next packet
   * in its chunk. Once a chunk is completed, the thread increments `sharedIdx` again to get more work.
   *
   * If `sharedIdx` grows to be larger than the incoming ray size, then the thread is complete.
   */
  void operator()() {
// const size_t maxPacketSize = 64;
//
#ifdef GVT_USE_DEBUG
    boost::timer::auto_cpu_timer t_functor("MantaMeshAdapter: thread trace time: %w\n");
#endif
    GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: started thread");

    GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: getting mesh [hack for now]");
    // TODO: don't use gvt mesh. need to figure out way to do per-vertex-normals and shading calculations
    auto mesh = (Mesh *)instNode["meshRef"].deRef()["ptr"].value().toULongLong();

    Manta::RenderContext &renderContext = *adapter->getRenderContext();

    // gvt::render::actor::RayVector rayPacket;
    // gvt::render::actor::RayVector localQueue;
    // gvt::render::actor::RayVector localDispatch;

    // localQueue.reserve(workSize * 2);
    localDispatch.reserve(rayList.size() * 2);

    // there is an upper bound on the nubmer of shadow rays generated per embree packet
    // its embree_packetSize * lights.size()
    // TODO: what about Manta?
    shadowRays.reserve(packetSize * lights.size());

    GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: starting while loop");

    while (sharedIdx < rayList.size()) {
#ifdef GVT_USE_DEBUG
// boost::timer::auto_cpu_timer t_outer_loop("EmbreeMeshAdapter: workSize rays traced: %w\n");
#endif
      // atomically get the next chunk range
      size_t workStart = sharedIdx.fetch_add(workSize);

      // have to double check that we got the last valid chunk range
      if (workStart > rayList.size()) {
        break;
      }

      // calculate the end work range
      size_t workEnd = workStart + workSize;
      if (workEnd > rayList.size()) {
        workEnd = rayList.size();
      }

      Manta::RayPacketData rpData;
      Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, packetSize, 0,
                             Manta::RayPacket::NormalizedDirections);
      int valid[64] = { 0 };

      GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: working on rays [" << workStart << ", " << workEnd << "]");

      for (size_t localIdx = workStart; localIdx < workEnd; localIdx += packetSize) {
        // this is the local packet size. this might be less than the main packetSize due to uneven amount of rays
        const size_t localPacketSize = (localIdx + packetSize > workEnd) ? (workEnd - localIdx) : packetSize;

        // trace a packet of rays, then keep tracing the generated secondary rays to completion
        // tracks to see if there are any valid rays left in the packet, if so, keep tracing
        // NOTE: perf issue: this will cause smaller and smaller packets to be traced at a time - need to track to see
        // effects
        bool validRayLeft = true;

        // the first time we enter the loop, we want to reset the valid boolean list that was
        // modified with the previous packet
        bool resetValid = true;
        while (validRayLeft) {
          validRayLeft = false;

          prepRayPacket(mRays, valid, resetValid, localPacketSize, rayList, localIdx);
          // rtcIntersect4(valid, scene, ray4);
          resetValid = false;

          adapter->getAccelStruct()->intersect(renderContext, mRays);
          mRays.computeNormals<false>(renderContext);

          for (size_t pi = 0; pi < localPacketSize; ++pi) {
            auto &r = rayList[localIdx + pi];

            if (valid[pi] && mRays.wasHit(pi)) {
              // shadow ray hit something, so it should be dropped
              if (r.type == gvt::render::actor::Ray::SHADOW) {
                continue;
              }

              float t = mRays.getMinT(pi);
              r.t = t;

              // FIXME: manta does not take vertex normal information(true?), the examples have the application
              // calculate the normal using
              // math similar to the bottom.  this means we have to keep around a 'faces_to_normals' list along with a
              // 'normals' list
              // for the manta adapter
              //
              // for some reason the manta normals aren't working, so just going to manually calculate the triangle
              // normal
              glm::vec3 normal =
                  glm::normalize((*normi) * (glm::vec3)(transform<Manta::Vector, glm::vec3>(mRays.getNormal(pi))));

              if (r.type == gvt::render::actor::Ray::SECONDARY) {
                t = (t > 1) ? 1.f / t : t;
                r.w = r.w * t;
              }

              // generateShadowRays(r, normal, mesh);
              // std::vector<gvt::render::data::scene::Light*> lights = dom->getLights();
              // for (int lindex = 0; lindex < lights.size(); lindex++)
              for (gvt::render::data::scene::Light *light : lights) {
                gvt::render::actor::Ray ray(r);
                ray.type = gvt::render::actor::Ray::SHADOW;
                ray.origin = ray.origin + ray.direction * ray.t;
                ray.setDirection(light->position - ray.origin);
                gvt::render::data::Color c = mesh->mat->shade(ray, normal, light, light->position);
                ray.color = c; // GVT_COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);
                // ray.color = GVT_COLOR_ACCUM(1.f, 1.0, c[1], c[2], 1.f);
                // localQueue.push_back(ray);
                shadowRays.push_back(ray);
              }

              int ndepth = r.depth - 1;
              float p = 1.f - (float(rand()) / RAND_MAX);

              // replace current ray with generated secondary ray
              if (ndepth > 0 && r.w > p) {
                r.type = gvt::render::actor::Ray::SECONDARY;
                r.origin = r.origin + r.direction * r.t;
                r.setDirection(mesh->getMaterial()->CosWeightedRandomHemisphereDirection2(normal));
                r.w = r.w * (r.direction * normal);
                r.depth = ndepth;
                validRayLeft = true; // we still have a valid ray in the packet to trace
                                     // localQueue.push_back(r);
              } else {
                // secondary ray is terminated, so disable its valid bit
                valid[pi] = 0;
              }
            } else {
              // ray is valid, but did not hit anything, so add to dispatch queue and disable it
              localDispatch.push_back(r);
              valid[pi] = 0;
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
    std::unique_lock<std::mutex> moved(adapter->_outqueue);
    moved_rays.insert(moved_rays.end(), localDispatch.begin(), localDispatch.end());
    moved.unlock();
  }
};

// void MantaMeshAdapter::trace(gvt::render::actor::RayVector& rayList,
// gvt::render::actor::RayVector& moved_rays)
void MantaMeshAdapter::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                             glm::mat4 *m, glm::mat4 *minv, glm::mat3 *normi,
                             std::vector<gvt::render::data::scene::Light *> &lights, size_t begin, size_t end) {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t_functor("MantaMeshAdapter: trace time: %w\n");
#endif
  // GVT_DEBUG(DBG_ALWAYS, "trace<MantaMeshAdapter>: " << rayList.size());
  // GVT_DEBUG(DBG_ALWAYS, "tracing geometry of domain " << domainID);
  // size_t workload = std::max((size_t) 1, (size_t) (rayList.size() /
  // (gvt::core::schedule::asyncExec::instance()->numThreads * 4)));

  std::atomic<size_t> sharedIdx(0); // shared index into rayList
  const size_t numThreads = gvt::core::schedule::asyncExec::instance()->numThreads;
  const size_t workSize = std::max((size_t)8, (size_t)(rayList.size() / (numThreads * 8))); // size of 'chunk'
                                                                                            // of rays to work
                                                                                            // on
  const size_t workload =
      std::max((size_t)1, (size_t)(rayList.size() / (gvt::core::schedule::asyncExec::instance()->numThreads * 4)));

  for (size_t rc = 0; rc < numThreads; ++rc) {
    // gvt::core::schedule::asyncExec::instance()->run_task(
    //         mantaParallelTrace(this, rayList, moved_rays, sharedIdx, workSize, instNode, m, minv, normi, lights,
    //         counter)
    //         );
    gvt::core::schedule::asyncExec::instance()->run_task(
        parallelTrace(this, rayList, moved_rays, workload, counter_b, m, minv, normi, lights, mesh));
  }

  gvt::core::schedule::asyncExec::instance()->sync();
  //            mantaParallelTrace(this, rayList, moved_rays, rayList.size(),counter)();

  // serial call example
  // mantaParallelTrace(this, rayList, moved_rays, sharedIdx, workSize, instNode, m, minv, normi, lights, counter)();

  // GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: Processed rays: " << counter);
  GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: Forwarding rays: " << moved_rays.size());

  rayList.clear();
}
