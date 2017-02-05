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

#include <cassert>
#include <gvt/render/tracer/RayTracer.h>
#if 0
#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/MantaMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/OptixMeshAdapter.h>
#endif

#if defined(GVT_RENDER_ADAPTER_OPTIX) && defined(GVT_RENDER_ADAPTER_EMBREE)
#include <gvt/render/adapter/heterogeneous/HeterogeneousMeshAdapter.h>
#endif
#endif

#include <boost/timer/timer.hpp>

namespace gvt {
namespace render {

RayTracer::RayTracer() : cntxt(gvt::render::RenderContext::instance()) {
  resetCamera();
  resetFilm();
  resetBVH();
};

RayTracer::~RayTracer(){};

void RayTracer::operator()() {
  cam->AllocateCameraRays();
  cam->generateRays();
  gvt::render::actor::RayVector moved_rays;
  for (auto d : meshRef) {
    std::cout << "Processing " << d.first << std::endl;
    calladapter(d.first, cam->rays, moved_rays);
  }
  processRays(moved_rays);
  img->composite();
};
bool RayTracer::MessageManager(std::shared_ptr<gvt::comm::Message> msg) { return true; };
bool RayTracer::isDone() { return false; };
bool RayTracer::hasWork() { return true; };

void RayTracer::processRays(gvt::render::actor::RayVector &rays, const int src, const int dst) {
  for (gvt::render::actor::Ray &r : rays) {
    img->localAdd(r.id, r.color * r.w, 1.0, r.t);
  }
}

void RayTracer::calladapter(const int instTarget, gvt::render::actor::RayVector &toprocess,
                            gvt::render::actor::RayVector &moved_rays) {
  std::shared_ptr<gvt::render::Adapter> adapter;

  gvt::render::data::primitives::Mesh *mesh = meshRef[instTarget];
  auto it = adapterCache.find(mesh);

  if (it != adapterCache.end()) {
    adapter = it->second;
  } else {
    adapter = 0;
  }

  if (!adapter) {
    switch (adapterType) {
#ifdef GVT_RENDER_ADAPTER_EMBREE
    case gvt::render::adapter::Embree:
      adapter = std::make_shared<gvt::render::adapter::embree::data::EmbreeMeshAdapter>(mesh);
      break;
#endif
#ifdef GVT_RENDER_ADAPTER_MANTA
    case gvt::render::adapter::Manta:
      adapter = new gvt::render::adapter::manta::data::MantaMeshAdapter(mesh);
      break;
#endif
#ifdef GVT_RENDER_ADAPTER_OPTIX
    case gvt::render::adapter::Optix:
      adapter = new gvt::render::adapter::optix::data::OptixMeshAdapter(mesh);
      break;
#endif

#if defined(GVT_RENDER_ADAPTER_OPTIX) && defined(GVT_RENDER_ADAPTER_EMBREE)
    case gvt::render::adapter::Heterogeneous:
      adapter = new gvt::render::adapter::heterogeneous::data::HeterogeneousMeshAdapter(mesh);
      break;
#endif
    default:
      GVT_ERR_MESSAGE("Image scheduler: unknown adapter type: " << adapterType);
    }
    adapterCache[mesh] = adapter;
  }
  GVT_ASSERT(adapter != nullptr, "image scheduler: adapter not set");
  {
    moved_rays.reserve(toprocess.size() * 10);
    adapter->trace(toprocess, moved_rays, instM[instTarget], instMinv[instTarget], instMinvN[instTarget], lights);
    toprocess.clear();
  }
}

float *RayTracer::getImageBuffer() { return img->composite(); };
void RayTracer::resetCamera() {
  assert(cntxt != nullptr);
  cam = std::make_shared<gvt::render::data::scene::gvtPerspectiveCamera>();
  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH camNode = root["Camera"];
  gvt::core::DBNodeH filmNode = root["Film"];
  glm::vec3 cameraposition = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  float fov = camNode["fov"].value().toFloat();
  glm::vec3 up = camNode["upVector"].value().tovec3();
  int rayMaxDepth = camNode["rayMaxDepth"].value().toInteger();
  int raySamples = camNode["raySamples"].value().toInteger();
  cam->lookAt(cameraposition, focus, up);
  cam->setMaxDepth(rayMaxDepth);
  cam->setSamples(raySamples);
  cam->setFOV(fov);
  cam->setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());
}

void RayTracer::resetFilm() {
  assert(cntxt != nullptr);
  img = std::make_shared<gvt::render::composite::IceTComposite>();
  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH filmNode = root["Film"];
  img = std::make_shared<gvt::render::composite::IceTComposite>(filmNode["width"].value().toInteger(),
                                                                filmNode["height"].value().toInteger());
}

void RayTracer::resetBVH() {
  assert(cntxt != nullptr);
  gvt::core::DBNodeH rootnode = cntxt->getRootNode();

  gvt::core::Vector<gvt::core::DBNodeH> instancenodes = rootnode["Instances"].getChildren();
  adapterType = rootnode["Schedule"]["adapter"].value().toInteger();
  int numInst = instancenodes.size();
  meshRef.clear();
  instM.clear();
  instMinv.clear();
  instMinvN.clear();
  for (auto &l : lights) {
    delete l;
  }

  lights.clear();
  bvh = std::make_shared<gvt::render::data::accel::BVH>(instancenodes);
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
}
}
