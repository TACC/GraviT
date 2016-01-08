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
// MantaDomain.C
//

#include <gvt/render/adapter/manta/data/domain/MantaDomain.h>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/manta/data/Transforms.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

// Manta includes
#include <Core/Math/CheapRNG.h>
#include <Core/Math/MT_RNG.h>
#include <Engine/Shadows/HardShadows.h>
#include <Engine/Shadows/NoShadows.h>
#include <Model/AmbientLights/AmbientOcclusionBackground.h>
#include <Model/AmbientLights/ArcAmbient.h>
#include <Model/AmbientLights/ConstantAmbient.h>
#include <Model/AmbientLights/EyeAmbient.h>
#include <Model/Primitives/Cube.h>
// end Manta includes

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>

using namespace gvt::render::actor;
using namespace gvt::render::adapter::manta::data;
using namespace gvt::render::adapter::manta::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;

static boost::atomic<size_t> counter(0);

MantaDomain::MantaDomain(GeometryDomain *domain) : GeometryDomain(*domain) {
  GVT_DEBUG(DBG_ALWAYS, "Converting domain");

  // Transform mesh
  meshManta = transform<Mesh *, Manta::Mesh *>(this->mesh);
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

MantaDomain::MantaDomain(std::string filename, gvt::core::math::AffineTransformMatrix<float> m)
    : gvt::render::data::domain::GeometryDomain(filename, m) {}

MantaDomain::MantaDomain(const MantaDomain &other) : gvt::render::data::domain::GeometryDomain(other) {}

MantaDomain::~MantaDomain() {
  // GeometryDomain::~GeometryDomain();
}

bool MantaDomain::load() {
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

void MantaDomain::free() {}

struct parallelTrace {
  gvt::render::adapter::manta::data::domain::MantaDomain *dom;
  gvt::render::actor::RayVector &rayList;
  gvt::render::actor::RayVector &moved_rays;
  const size_t workSize;

  boost::atomic<size_t> &counter;

  parallelTrace(gvt::render::adapter::manta::data::domain::MantaDomain *dom, gvt::render::actor::RayVector &rayList,
                gvt::render::actor::RayVector &moved_rays, const size_t workSize, boost::atomic<size_t> &counter)
      : dom(dom), rayList(rayList), moved_rays(moved_rays), workSize(workSize), counter(counter) {}

  void operator()() {
    const size_t maxPacketSize = 64;

    Manta::RenderContext &renderContext = *dom->getRenderContext();

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
      boost::unique_lock<boost::mutex> queue(dom->_inqueue);
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
          mRays.setRay(i, transform<Ray, Manta::Ray>(dom->toLocal(rayPacket[i])));
          // mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(rayPacket[i]));
        }

        mRays.resetHits();
        dom->getAccelStruct()->intersect(renderContext, mRays);
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

            gvt::core::math::Vector4f normal =
                dom->toWorld(gvt::render::adapter::manta::data::transform<Manta::Vector, gvt::core::math::Vector4f>(
                    mRays.getNormal(pindex)));

            if (rayPacket[pindex].type == gvt::render::actor::Ray::SECONDARY) {
              t = (t > 1) ? 1.f / t : t;
              rayPacket[pindex].w = rayPacket[pindex].w * t;
            }

            std::vector<gvt::render::data::scene::Light *> lights = dom->getLights();
            for (int lindex = 0; lindex < lights.size(); lindex++) {
              gvt::render::actor::Ray ray(rayPacket[pindex]);
              ray.domains.clear();
              ray.type = gvt::render::actor::Ray::SHADOW;
              ray.origin = ray.origin + ray.direction * ray.t;
              ray.setDirection(lights[lindex]->position - ray.origin);
              gvt::render::data::Color c = dom->getMesh()->shade(ray, normal, lights[lindex]);
              // ray.color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);
              ray.color = GVT_COLOR_ACCUM(1.f, 1.0, c[1], c[2], 1.f);
              localQueue.push_back(ray);
            }

            int ndepth = rayPacket[pindex].depth - 1;

            float p = 1.f - (float(rand()) / RAND_MAX);

            if (ndepth > 0 && rayPacket[pindex].w > p) {
              gvt::render::actor::Ray ray(rayPacket[pindex]);
              ray.domains.clear();
              ray.type = gvt::render::actor::Ray::SECONDARY;
              ray.origin = ray.origin + ray.direction * ray.t;
              ray.setDirection(
                  dom->getMesh()->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());
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

    boost::unique_lock<boost::mutex> moved(dom->_outqueue);
    moved_rays.insert(moved_rays.begin(), localDispatch.begin(), localDispatch.end());
    moved.unlock();
  }
};

void MantaDomain::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays) {
  GVT_DEBUG(DBG_ALWAYS, "trace<MantaDomain>: " << rayList.size());
  GVT_DEBUG(DBG_ALWAYS, "tracing geometry of domain " << domainID);
  size_t workload =
      std::max((size_t)1, (size_t)(rayList.size() / (gvt::core::schedule::asyncExec::instance()->numThreads * 4)));

  for (int rc = 0; rc < gvt::core::schedule::asyncExec::instance()->numThreads; ++rc) {
    gvt::core::schedule::asyncExec::instance()->run_task(parallelTrace(this, rayList, moved_rays, workload, counter));
  }
  gvt::core::schedule::asyncExec::instance()->sync();
//            parallelTrace(this, rayList, moved_rays, rayList.size(),counter)();

#ifdef NDEBUG
  std::cout << "Proccessed rays : " << counter << std::endl;
#else
  GVT_DEBUG(DBG_ALWAYS, "Proccessed rays : " << counter);
#endif
  GVT_DEBUG(DBG_ALWAYS, "Forwarding rays : " << moved_rays.size());
  rayList.clear();
}
