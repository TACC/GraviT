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
 * File:   Material.cpp
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:07 PM
 */

#include <cmath>
#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Material.h>
#ifdef GVT_RENDER_ADAPTER_EMBREE

#include <gvt/render/adapter/embree/EmbreeMaterial.h>
using namespace embree;

#endif
using namespace gvt::render::actor;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;



////////////////////////////////////////////////////////////////////////////////
//                          GVT Legacy Materials                              //
////////////////////////////////////////////////////////////////////////////////

glm::vec3 lambertShade(const gvt::render::data::primitives::Material *material, const gvt::render::actor::Ray &ray,
                       const glm::vec3 &N, const glm::vec3 &wi) {

  float NdotL = std::max(0.f, glm::dot(N, wi));
  glm::vec3 diffuse = material->kd * (NdotL * ray.w);

  return diffuse;
}

glm::vec3 phongShade(const gvt::render::data::primitives::Material *material, const gvt::render::actor::Ray &ray,
                     const glm::vec3 &N, const glm::vec3 &wi) {

  float NdotL = std::max(0.f, glm::dot(N, wi));
  glm::vec3 R = ((N * 2.f) * NdotL) - wi;
  float VdotR = std::max(0.f, glm::dot(R, (-ray.direction)));
  float power = VdotR * std::pow(VdotR, material->alpha);

  gvt::render::data::Color finalColor = material->kd * (NdotL * ray.w);
  finalColor += material->ks * (power * ray.w);
  return finalColor;
}

glm::vec3 blinnPhongShade(const gvt::render::data::primitives::Material *material, const gvt::render::actor::Ray &ray,
                          const glm::vec3 &N, const glm::vec3 &wi) {

  float NdotL = std::max(0.f, glm::dot(N, wi));

  glm::vec3 H = glm::normalize(wi - ray.direction);

  float NdotH = std::max(0.f, glm::dot(H, N));
  float power = NdotH * std::pow(NdotH, material->alpha);

  gvt::render::data::Color diffuse = material->kd * (NdotL * ray.w);
  gvt::render::data::Color specular = material->ks * (power * ray.w);

  gvt::render::data::Color finalColor = (diffuse + specular);
  return finalColor;
}


bool gvt::render::data::primitives::Shade(gvt::render::data::primitives::Material *material,
                                          const gvt::render::actor::Ray &ray, const glm::vec3 &surfaceNormal,
                                          const gvt::render::data::scene::Light *lightSource,
                                          const glm::vec3 lightPosSample, glm::vec3 &color) {

  glm::vec3 hitPoint = ray.origin + ray.direction * ray.t;
  glm::vec3 wi = glm::normalize(lightPosSample - hitPoint);
  float NdotL = std::max(0.f, glm::dot(surfaceNormal, wi));
  glm::vec3 Li = lightSource->contribution(hitPoint, lightPosSample);

  if (NdotL == 0.f || (Li[0] == 0.f && Li[1] == 0.f && Li[2] == 0.f)) return false;

  switch (material->type) {
  case LAMBERT:
    color = lambertShade(material, ray, surfaceNormal, wi);
    break;
  case PHONG:
    color = phongShade(material, ray, surfaceNormal, wi);
    break;
  case BLINN:
    color = blinnPhongShade(material, ray, surfaceNormal, wi);
    break;
#ifdef GVT_RENDER_ADAPTER_EMBREE
  case EMBREE_MATERIAL_METAL:
  case EMBREE_MATERIAL_VELVET:
  case EMBREE_MATERIAL_MATTE: {

    DifferentialGeometry dg;
    dg.Ns = Vec3fa(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z);
    Vec3fa ewi = Vec3fa(wi.x, wi.y, wi.z);
    Vec3fa wo = Vec3fa(-ray.direction.x, -ray.direction.y, -ray.direction.z);

    Vec3fa r = gvt::render::data::primitives::Material__eval(material, 0, 1, BRDF(), wo, dg, ewi);

    color = 2.f * glm::vec3(r.x, r.y, r.z) * ray.w;

  } break;
#endif
  default:
    printf("Material implementation missing for embree adpater\n");

    break;
  }

  color *= Li;

  color = glm::clamp(color, glm::vec3(0), glm::vec3(1));

  return true;
}
