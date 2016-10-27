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


#include <gvt/render/adapter/manta/MantaMeshAdapter.h>

#include <gvt/core/context/CoreContext.h>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/core/math/RandEngine.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>
#include <gvt/render/data/primitives/Material.h>
#include <gvt/render/data/primitives/Shade.h>

// Manta includes
#include <Core/Math/CheapRNG.h>
#include <Core/Math/MT_RNG.h>
#include <Model/Materials/Lambertian.h>
#include <Interface/Context.h>


#include <atomic>
#include <thread>

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>
#include <boost/tuple/tuple.hpp>

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>

using namespace gvt::render::actor;
using namespace gvt::render::adapter::manta::data;
using namespace gvt::render::data::primitives;

static std::atomic<size_t> counter(0);

MantaMeshAdapter::MantaMeshAdapter(gvt::render::data::primitives::Mesh *mesh)
    : Adapter(mesh), mantaInterface(NULL), preprocessContext(NULL), renderContext(NULL), scene(NULL), bvh(NULL),
      material(NULL), mantaMesh(NULL), randGen(NULL) {
  GVT_ASSERT(mesh, "MantaMeshAdapter: mesh pointer in the database is null");

  // Create Manta (allocated in the heap)
  mantaInterface = Manta::createManta();

  // Create preprocess context
  preprocessContext =
      new Manta::PreprocessContext(mantaInterface, 0 /* proc */, 1 /* numProcs */, NULL /* LightSet* */);

  // Transform mesh (GVT to Manta) begin

  // generate surface normals
  mesh->generateNormals();

  // set material (just for placeholder)
  material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
  material->preprocess(*preprocessContext);

  mantaMesh = new Manta::Mesh();
  mantaMesh->materials.push_back(material);

  for (int i = 0; i < mesh->vertices.size(); ++i) {
    const glm::vec3 &v = mesh->vertices[i];
    mantaMesh->vertices.push_back(Manta::Vector(v[0], v[1], v[2]));
  }

  for (int i = 0; i < mesh->normals.size(); ++i) {
    const glm::vec3 &n = mesh->normals[i];
    mantaMesh->vertexNormals.push_back(Manta::Vector(n[0], n[1], n[2]));
  }

  for (int i = 0; i < mesh->faces.size(); ++i) {
    // Face being boost::tuple<int, int, int>
    gvt::render::data::primitives::Mesh::Face f = mesh->faces[i];
    // texture indices
    mantaMesh->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
    mantaMesh->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
    mantaMesh->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
    // vertex indices
    mantaMesh->vertex_indices.push_back(boost::get<0>(f));
    mantaMesh->vertex_indices.push_back(boost::get<1>(f));
    mantaMesh->vertex_indices.push_back(boost::get<2>(f));
    // normal indices
    mantaMesh->normal_indices.push_back(boost::get<0>(f));
    mantaMesh->normal_indices.push_back(boost::get<1>(f));
    mantaMesh->normal_indices.push_back(boost::get<2>(f));
    mantaMesh->face_material.push_back(0);
    // triangle objects (to be deleted inside Manta)
    mantaMesh->addTriangle(new Manta::KenslerShirleyTriangle());
  }

  // Transform mesh (GVT to Manta) end

  // Create BVH
  bvh = new Manta::DynBVH();
  bvh->setGroup(mantaMesh);
  bvh->preprocess(*preprocessContext);

  scene = new Manta::Scene();
  scene->setObject(bvh);

  Manta::CheapRNG::create(randGen);

  renderContext = new Manta::RenderContext(mantaInterface,
                                           0,       // channel index
                                           0,       // proc
                                           1,       // num procs
                                           0,       // frame state
                                           0,       // loadbalancer
                                           0,       // pixelsampler
                                           0,       // renderer
                                           0,       // shadowAlgorithm
                                           0,       // camera
                                           scene,   // scene
                                           0,       // thread_storage
                                           randGen, // rngs
                                           0        // samplegenerator
                                           );
}

MantaMeshAdapter::~MantaMeshAdapter() {
  if (renderContext) delete renderContext;
  if (randGen) delete randGen;
  if (scene) delete scene;
  if (bvh) delete bvh;
  if (mantaMesh) delete mantaMesh;
  if (material) delete material;
  if (preprocessContext) delete preprocessContext;
  if (mantaInterface) delete mantaInterface;
}

bool MantaMeshAdapter::load() { return true; }

void MantaMeshAdapter::free() {}

struct mantaParallelTrace {
  /**
   * Pointer to MantaMeshAdapter
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
  const size_t packetSize;

  const size_t begin, end;

  gvt::render::data::primitives::Mesh *mesh;

  /**
   * Construct a mantaParallelTrace struct with information needed for the
   * thread
   * to do its tracing
   */
  mantaParallelTrace(gvt::render::adapter::manta::data::MantaMeshAdapter *adapter,
                     gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                     const size_t workSize, glm::mat4 *m, glm::mat4 *minv, glm::mat3 *normi,
                     std::vector<gvt::render::data::scene::Light *> &lights, gvt::render::data::primitives::Mesh *mesh,
                     std::atomic<size_t> &counter, const size_t begin, const size_t end)
      : adapter(adapter), rayList(rayList), moved_rays(moved_rays), workSize(workSize), m(m), minv(minv), normi(normi),
        lights(lights), mesh(mesh), counter(counter), begin(begin), end(end), packetSize(Manta::RayPacket::MaxSize) {}

  /**
   * Convert a set of rays from a vector into a Manta ray packet.
   *
   * \param startIdx      starting point to read from in `rays`
   * \param rays          vector of rays to read from
   * \param mRays         reference of Manta ray packet to write to
   */
  void prepRayPacket(std::size_t startIdx, const gvt::render::actor::RayVector &rays, Manta::RayPacket *mRays) {
    // convert packetSize rays into manta's RayPacket class
    for (int i = mRays->begin(); i < mRays->end(); ++i) {
      const Ray &r = rays[startIdx + i];
      const auto origin = (*minv) * glm::vec4(r.origin, 1.f); // transform ray to local space
      const auto direction = (*minv) * glm::vec4(r.direction, 0.f);

      mRays->setRay(i, Manta::Vector(origin[0], origin[1], origin[2]),
                    Manta::Vector(direction[0], direction[1], direction[2]));
    }

    // reset ray packet (Manta sets hitMatl[i] to 0 and minT[i] to MAXT)
    mRays->resetHits();
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
                          gvt::render::data::primitives::Material *material, unsigned int *randSeed) {

    for (gvt::render::data::scene::Light *light : lights) {
      GVT_ASSERT(light, "generateShadowRays: light is null for some reason");
      // Try to ensure that the shadow ray is on the correct side of the
      // triangle.
      // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
      // Using about 8 * ULP(t).

      gvt::render::data::Color c;
      glm::vec3 lightPos;
      if (light->LightT == gvt::render::data::scene::Light::Area) {
        lightPos = ((gvt::render::data::scene::AreaLight *)light)->GetPosition(randSeed);
      } else {
        lightPos = light->position;
      }

      if (!gvt::render::data::primitives::Shade(material, r, normal, light, lightPos, c)) continue;

      const float multiplier = 1.0f - gvt::render::actor::Ray::RAY_EPSILON;
      const float t_shadow = multiplier * r.t;

      const glm::vec3 origin = r.origin + r.direction * t_shadow;
      const glm::vec3 dir = lightPos - origin;
      const float t_max = dir.length();

      // note: ray copy constructor is too heavy, so going to build it manually
      shadowRays.push_back(Ray(r.origin + r.direction * t_shadow, dir, r.w, Ray::SHADOW, r.depth));

      Ray &shadow_ray = shadowRays.back();
      shadow_ray.t = r.t;
      shadow_ray.id = r.id;
      shadow_ray.t_max = t_max;

      // gvt::render::data::Color c = adapter->getMesh()->mat->shade(shadow_ray,
      // normal, lights[lindex]);
      shadow_ray.color = glm::vec3(c[0], c[1], c[2]);
    }
  }

  glm::vec3 CosWeightedRandomHemisphereDirection2(glm::vec3 n, gvt::core::math::RandEngine &randEngine) {
    float Xi1 = 0;
    float Xi2 = 0;
    //	    if(randSeed == nullptr)
    //	    {
    //	      Xi1 = (float)rand() / (float)RAND_MAX;
    //	      Xi2 = (float)rand() / (float)RAND_MAX;
    //	    }
    //	    else
    //	    {
    Xi1 = randEngine.fastrand(0, 1);
    Xi2 = randEngine.fastrand(0, 1);
    //}

    float theta = std::acos(std::sqrt(1.0 - Xi1));
    float phi = 2.0 * 3.1415926535897932384626433832795 * Xi2;

    float xs = sinf(theta) * cosf(phi);
    float ys = cosf(theta);
    float zs = sinf(theta) * sinf(phi);

    glm::vec3 y(n);
    glm::vec3 h = y;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
      h[0] = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
      h[1] = 1.0;
    else
      h[2] = 1.0;

    glm::vec3 x = glm::cross(h, y);
    glm::vec3 z = glm::cross(x, y);

    glm::vec3 direction = x * xs + y * ys + z * zs;
    return glm::normalize(direction);
  }

  /**
   * Test occlusion for stored shadow rays.  Add missed rays
   * to the dispatch queue.
   */
  void traceShadowRays() {
    // int valid[Manta::RayPacket::MaxSize] = { 0 };
    Manta::RayPacketData rpData;
    Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0 /*rayBegin*/, Manta::RayPacket::MaxSize /*rayEnd*/,
                           0 /*depth*/, Manta::RayPacket::NormalizedDirections);

    const Manta::RenderContext &renderContext = *(adapter->getRenderContext());

    for (size_t idx = 0; idx < shadowRays.size(); idx += Manta::RayPacket::MaxSize) {
      const size_t localPacketSize =
          (idx + Manta::RayPacket::MaxSize > shadowRays.size()) ? (shadowRays.size() - idx) : Manta::RayPacket::MaxSize;

      Manta::RayPacket rayPacket(mRays, 0, localPacketSize);

      // populate rayPacket
      prepRayPacket(idx, shadowRays, &rayPacket);

      // do intersection test
      adapter->getAccelStruct()->intersect(renderContext, rayPacket);

      for (size_t pi = 0; pi < rayPacket.end(); ++pi) {
        if (!rayPacket.wasHit(pi)) {
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

    const Manta::RenderContext &renderContext = *(adapter->getRenderContext());

    localDispatch.reserve((end - begin) * 2);

    shadowRays.reserve(lights.size());

    GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: starting while loop");

    gvt::core::math::RandEngine randEngine;
    randEngine.SetSeed(begin);

    Manta::RayPacketData rpData;
    Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0 /*rayBegin*/, Manta::RayPacket::MaxSize /*rayEnd*/,
                           0 /*depth*/, Manta::RayPacket::NormalizedDirections /*flags*/);

    for (std::size_t localIdx = begin; localIdx < end; localIdx += Manta::RayPacket::MaxSize) {

      const std::size_t localPacketSize =
          (localIdx + Manta::RayPacket::MaxSize > end) ? (end - localIdx) : Manta::RayPacket::MaxSize;

      // create a subset of ray packet
      Manta::RayPacket rayPacket(mRays, 0, localPacketSize);

      // trace a packet of rays, then keep tracing the generated secondary
      // rays to completion
      // tracks to see if there are any valid rays left in the packet, if so,
      // keep tracing
      // NOTE: perf issue: this will cause smaller and smaller packets to be
      // traced at a time - need to track to see effects
      bool validRayLeft = true;

      while (validRayLeft) {
        validRayLeft = false;

        // copy rays to rayPacket
        prepRayPacket(localIdx, rayList, &rayPacket);

        // intersect ray packet with bvh and compute normals
        adapter->getAccelStruct()->intersect(renderContext, rayPacket);
        rayPacket.computeFFNormals<false>(renderContext);

        // trace rays
        for (size_t pi = 0; pi < localPacketSize; ++pi) {
          if (!rayPacket.rayIsMasked(pi)) {
            gvt::render::actor::Ray &r = rayList[localIdx + pi];

            if (rayPacket.wasHit(pi)) {

              // shadow ray hit something, so it should be dropped
              if (r.type == gvt::render::actor::Ray::SHADOW) {
                continue;
              }

              float t = rayPacket.getMinT(pi);
              r.t = t;

              Manta::Vector mantaNormal = rayPacket.getFFNormal(pi);
              glm::vec3 normal = glm::normalize((*normi) * glm::vec3(mantaNormal[0], mantaNormal[1], mantaNormal[2]));

              Material *mat = mesh->getMaterial();

              // reduce contribution of the color that the shadow rays get
              if (r.type == gvt::render::actor::Ray::SECONDARY) {
                t = (t > 1) ? 1.f / t : t;
                r.w = r.w * t;
              }

              generateShadowRays(r, normal, mat, randEngine.ReturnSeed());

              int ndepth = r.depth - 1;

              float p = 1.f - randEngine.fastrand(0, 1); //(float(rand()) / RAND_MAX);

              // replace current ray with generated secondary ray
              if (ndepth > 0 && r.w > p) {
                r.type = gvt::render::actor::Ray::SECONDARY;
                const float multiplier =
                    1.0f - 16.0f * std::numeric_limits<float>::epsilon(); // TODO: move out somewhere / make static
                const float t_secondary = multiplier * r.t;
                r.origin = r.origin + r.direction * t_secondary;


                r.direction = CosWeightedRandomHemisphereDirection2(normal, randEngine);

                r.w = r.w * glm::dot(r.direction, normal);
                r.depth = ndepth;
                validRayLeft = true; // we still have a valid ray in the packet to trace
              } else {
                // secondary ray is terminated, so disable its valid bit
                rayPacket.maskRay(pi);
              }

            } else {
              // ray is valid, but did not hit anything, so add to dispatch queue and disable it
              localDispatch.push_back(r);
              rayPacket.maskRay(pi);
            }
          }
        }
        // trace shadow rays generated by the packet
        traceShadowRays();
      }
    }


    // copy localDispatch rays to outgoing rays queue
    std::unique_lock<std::mutex> moved(adapter->_outqueue);
    moved_rays.insert(moved_rays.end(), localDispatch.begin(), localDispatch.end());
    moved.unlock();
  }
};

void MantaMeshAdapter::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                             glm::mat4 *m, glm::mat4 *minv, glm::mat3 *normi,
                             std::vector<gvt::render::data::scene::Light *> &lights, size_t _begin, size_t _end) {

  std::size_t begin = _begin;
  std::size_t end = _end;

  if (_end == 0) {
    begin = 0;
    end = rayList.size();
  }

  const size_t numThreads = std::thread::hardware_concurrency();
  const size_t workSize = std::max((size_t)4, (size_t)((end - begin) / (numThreads * 2))); // size of 'chunk'
                                                                                           // of rays to work
                                                                                           // on

  static tbb::auto_partitioner ap;
  tbb::parallel_for(tbb::blocked_range<size_t>(begin, end, workSize),
                    [&](tbb::blocked_range<size_t> chunk) {
                      mantaParallelTrace(this, rayList, moved_rays, chunk.end() - chunk.begin(), m, minv, normi, lights,
                                         mesh, counter, chunk.begin(), chunk.end())();
                    },
                    ap);

  GVT_DEBUG(DBG_ALWAYS, "MantaMeshAdapter: Forwarding rays: " << moved_rays.size());
}
