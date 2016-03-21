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
#include <gvt/render/data/primitives/EmbreeMaterial.h>

using namespace gvt::render::actor;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;

using namespace embree;


////////////////////////////////////////////////////////////////////////////////
//                          GVT Legacy Materials                              //
////////////////////////////////////////////////////////////////////////////////

glm::vec3 lambertShade(const gvt::render::data::primitives::Material *material, const gvt::render::actor::Ray &ray,
                       const glm::vec3 &N, const gvt::render::data::scene::Light *lightSource,
                       const glm::vec3 lightPostion) {
  glm::vec3 hitPoint = ray.origin + ray.direction * ray.t;
  glm::vec3 L = glm::normalize(lightSource->position - hitPoint);
  float NdotL = std::max(0.f, std::abs(glm::dot(N, L)));
  glm::vec3  lightSourceContrib = lightSource->contribution(hitPoint);
  glm::vec3  diffuse = (lightSourceContrib * material->kd) * (NdotL * ray.w);
  return diffuse;
}

glm::vec3 phongShade(const gvt::render::data::primitives::Material *material, const Ray &ray, const glm::vec3 &N,
                     const Light *lightSource, const glm::vec3 lightPostion) {
  glm::vec3 hitPoint = ray.origin + (ray.direction * ray.t);
  glm::vec3 L = glm::normalize(lightPostion - hitPoint);

  float NdotL = std::max(0.f, glm::dot(N, L));
  glm::vec3 R = ((N * 2.f) * NdotL) - L;
  float VdotR = std::max(0.f, glm::dot(R, (-ray.direction)));
  float power = VdotR * std::pow(VdotR, material->alpha);

  glm::vec3 lightSourceContrib = lightSource->contribution(hitPoint); //  distance;

  gvt::render::data::Color finalColor = (lightSourceContrib * material->kd) * (NdotL * ray.w);
  finalColor += (lightSourceContrib * material->ks) * (power * ray.w);
  return finalColor;
}

glm::vec3 blinnPhongShade(const gvt::render::data::primitives::Material *material, const Ray &ray, const glm::vec3 &N,
                          const Light *lightSource, const glm::vec3 lightPostion) {
  glm::vec3 hitPoint = ray.origin + (ray.direction * ray.t);
  glm::vec3 L = glm::normalize(lightPostion - hitPoint);
  float NdotL = std::max(0.f, glm::dot(N, L));

  glm::vec3 H = glm::normalize(L - ray.direction);

  float NdotH = std::max(0.f, glm::dot(H, N));
  float power = NdotH * std::pow(NdotH, material->alpha);

  glm::vec3 lightSourceContrib = lightSource->contribution(hitPoint);

  gvt::render::data::Color diffuse = (lightSourceContrib * material->kd) * (NdotL * ray.w);
  gvt::render::data::Color specular = (lightSourceContrib * material->ks) * (power * ray.w);

  gvt::render::data::Color finalColor = (diffuse + specular);
  return finalColor;
}


////////////////////////////////////////////////////////////////////////////////
//                          Embree Materials                                  //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                          Matte Material                                    //
////////////////////////////////////////////////////////////////////////////////

#include "../embree/common/math/math.h"
#include "../embree/common/math/linearspace3.h"
#include "../embree/tutorials/pathtracer/shapesampler.h"
#include "../embree/tutorials/pathtracer/optics.h"

inline Vec3fa toVec3fa(glm::vec3& v){
	return Vec3fa(v.x,v.y,v.z);
}

struct Lambertian {
  Vec3fa R;
};

inline Vec3fa Lambertian__eval(const Lambertian *This, const Vec3fa &wo, const DifferentialGeometry &dg,
                               const Vec3fa &wi) {
  return This->R * (1.0f/(float)(float(pi))) * clamp(dot(wi, dg.Ns));
}

inline Vec3fa Lambertian__sample(const Lambertian *This, const Vec3fa &wo, const DifferentialGeometry &dg, Sample3f &wi,
                                 const Vec2f &s) {
  wi = cosineSampleHemisphere(s.x, s.y, dg.Ns);
  return Lambertian__eval(This, wo, dg, wi.v);
}

inline void Lambertian__Constructor(Lambertian *This, const Vec3fa &R) { This->R = R; }

inline Lambertian make_Lambertian(const Vec3fa &R) {
  Lambertian v;
  Lambertian__Constructor(&v, R);
  return v;
}

Vec3fa MatteMaterial__eval(Material *This, const BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
                           const Vec3fa &wi) {
  Lambertian lambertian = make_Lambertian(Vec3fa(toVec3fa(This->ks)));
  return Lambertian__eval(&lambertian, wo, dg, wi);
}

////////////////////////////////////////////////////////////////////////////////
//                          Minneart BRDF                                     //
////////////////////////////////////////////////////////////////////////////////

struct Minneart {
  /*! The reflectance parameter. The vale 0 means no reflection,
   *  and 1 means full reflection. */
  Vec3fa R;

  /*! The amount of backscattering. A value of 0 means lambertian
   *  diffuse, and inf means maximum backscattering. */
  float b;
};

inline Vec3fa Minneart__eval(const Minneart *This, const Vec3fa &wo, const DifferentialGeometry &dg, const Vec3fa &wi) {
  const float cosThetaI = clamp(dot(wi, dg.Ns));
  const float backScatter = powf(clamp(dot(wo, wi)), This->b);
  return (backScatter * cosThetaI * float(one_over_pi)) * This->R;
}

inline Vec3fa Minneart__sample(const Minneart *This, const Vec3fa &wo, const DifferentialGeometry &dg, Sample3f &wi,
                               const Vec2f &s) {
  wi = cosineSampleHemisphere(s.x, s.y, dg.Ns);
  return Minneart__eval(This, wo, dg, wi.v);
}

inline void Minneart__Constructor(Minneart *This, const Vec3fa &R, const float b) {
  This->R = R;
  This->b = b;
}

inline Minneart make_Minneart(const Vec3fa &R, const float f) {
  Minneart m;
  Minneart__Constructor(&m, R, f);
  return m;
}

////////////////////////////////////////////////////////////////////////////////
//                        Velvet Material                                     //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                            Velvet BRDF                                     //
////////////////////////////////////////////////////////////////////////////////

struct Velvety {
  BRDF base;

  /*! The reflectance parameter. The vale 0 means no reflection,
   *  and 1 means full reflection. */
  Vec3fa R;

  /*! The falloff of horizon scattering. 0 no falloff,
   *  and inf means maximum falloff. */
  float f;
};

inline Vec3fa Velvety__eval(const Velvety *This, const Vec3fa &wo, const DifferentialGeometry &dg, const Vec3fa &wi) {
  const float cosThetaO = clamp(dot(wo, dg.Ns));
  const float cosThetaI = clamp(dot(wi, dg.Ns));
  const float sinThetaO = sqrt(1.0f - cosThetaO * cosThetaO);
  const float horizonScatter = powf(sinThetaO, This->f);
  return (horizonScatter * cosThetaI * float(one_over_pi)) * This->R;
}

//inline Vec3fa Velvety__sample(const Velvety *This, const Vec3fa &wo, const DifferentialGeometry &dg, Sample3f &wi,
//                              const Vec2f &s) {
//  wi = cosineSampleHemisphere(s.x, s.y, dg.Ns);
//  return Velvety__eval(This, wo, dg, wi.v);
//}

inline void Velvety__Constructor(Velvety *This, const Vec3fa &R, const float f) {
  This->R = R;
  This->f = f;
}

inline Velvety make_Velvety(const Vec3fa &R, const float f) {
  Velvety m;
  Velvety__Constructor(&m, R, f);
  return m;
}

void VelvetMaterial__preprocess(Material *material, BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
                                const Medium &medium) {}

Vec3fa VelvetMaterial__eval(Material *This, const BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
                            const Vec3fa &wi) {
  Minneart minneart;
  Minneart__Constructor(&minneart, toVec3fa(This->ks), This->backScattering);
  Velvety velvety;
  Velvety__Constructor(&velvety, toVec3fa(This->horizonScatteringColor), This->horizonScatteringFallOff);
  return Minneart__eval(&minneart, wo, dg, wi) + Velvety__eval(&velvety, wo, dg, wi);
}

////////////////////////////////////////////////////////////////////////////////
//                        Metal Material                                      //
////////////////////////////////////////////////////////////////////////////////

Vec3fa MetalMaterial__eval(Material *This, const BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
                           const Vec3fa &wi) {
  const FresnelConductor fresnel = make_FresnelConductor(toVec3fa(This->eta), toVec3fa(This->k));
  const PowerCosineDistribution distribution = make_PowerCosineDistribution(rcp(This->roughness));

  const float cosThetaO = dot(wo, dg.Ns);
  const float cosThetaI = dot(wi, dg.Ns);
  if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) return Vec3fa(0.f);
  const Vec3fa wh = normalize(wi + wo);
  const float cosThetaH = dot(wh, dg.Ns);
  const float cosTheta = dot(wi, wh); // = dot(wo, wh);
  const Vec3fa F = eval(fresnel, cosTheta);
  const float D = eval(distribution, cosThetaH);
  const float G = min(1.0f, min(2.0f * cosThetaH * cosThetaO / cosTheta, 2.0f * cosThetaH * cosThetaI / cosTheta));
  return (toVec3fa(This->ks) * F) * D * G * rcp(4.0f * cosThetaO);
}




/*
 * Proxy call used by adapater/integrator
 * Will temporarly call the sub-proxy of each sub-set of materials
 * Interfaces to the different shading materials are significantly different
 * mainly due to light assessing
 */
glm::vec3 gvt::render::data::primitives::Shade(gvt::render::data::primitives::Material *material,
                                               const gvt::render::actor::Ray &ray, const glm::vec3 &sufaceNormal,
                                               const gvt::render::data::scene::Light *lightSource,
                                               const glm::vec3 lightPostion) {

  glm::vec3 color;


  switch (material->type) {
  case LAMBERT:
	  color = lambertShade(material, ray, sufaceNormal, lightSource, lightPostion);
    break;
  case PHONG:
	  color = phongShade(material, ray, sufaceNormal, lightSource, lightPostion);
    break;
  case BLINN:
	  color = blinnPhongShade(material, ray, sufaceNormal, lightSource, lightPostion);
    break;
   case EMBREE_MATERIAL_METAL:
  case EMBREE_MATERIAL_VELVET:
  case EMBREE_MATERIAL_MATTE:

    glm::vec3 hitPoint = ray.origin + ray.direction * ray.t;
    glm::vec3 L = glm::normalize(lightSource->position - hitPoint);
    glm::vec3 lightSourceContrib = lightSource->contribution(hitPoint);

    DifferentialGeometry dg;
    dg.Ns = Vec3fa(L.x, L.y, L.z);
    Vec3fa r = gvt::render::data::primitives::Material__eval(material, 0, 1, BRDF(), Vec3fa(), dg,
                                                             Vec3fa(sufaceNormal.x, sufaceNormal.y, sufaceNormal.z));

    color = lightSourceContrib * glm::vec3(r.x, r.y, r.z) * ray.w;
    break;
  default:
   	   printf("Material implementation missing for embree adpater\n");

     break;
   }


  return color;
}




inline Vec3fa gvt::render::data::primitives::Material__eval(Material *materials, int materialID, int numMaterials,
                                                            const BRDF &brdf, const Vec3fa &wo,
                                                            const DifferentialGeometry &dg, const Vec3fa &wi) {
  Vec3fa c = Vec3fa(0.0f);
  int id = materialID;
  {
    if (id >= 0 && id < numMaterials) // FIXME: workaround for ISPC bug, location reached with empty execution mask
    {
      Material *material = &materials[materialID];
      switch (material->type) {
      //      case MATERIAL_OBJ  : c = OBJMaterial__eval  ((OBJMaterial*)  material, brdf, wo, dg, wi); break;
      case EMBREE_MATERIAL_METAL:
        c = MetalMaterial__eval(material, brdf, wo, dg, wi);
        break;
      //      case MATERIAL_REFLECTIVE_METAL: c = ReflectiveMetalMaterial__eval((ReflectiveMetalMaterial*)material,
      //      brdf, wo, dg, wi); break;
      case EMBREE_MATERIAL_VELVET:
        c = VelvetMaterial__eval(material, brdf, wo, dg, wi);
        break;
      //      case MATERIAL_DIELECTRIC: c = DielectricMaterial__eval((DielectricMaterial*)material, brdf, wo, dg, wi);
      //      break;
      //      case MATERIAL_METALLIC_PAINT: c = MetallicPaintMaterial__eval((MetallicPaintMaterial*)material, brdf, wo,
      //      dg, wi); break;
      case EMBREE_MATERIAL_MATTE:
        c = MatteMaterial__eval(material, brdf, wo, dg, wi);
        break;
      //      case MATERIAL_MIRROR: c = MirrorMaterial__eval((MirrorMaterial*)material, brdf, wo, dg, wi); break;
      //      case MATERIAL_THIN_DIELECTRIC: c = ThinDielectricMaterial__eval((ThinDielectricMaterial*)material, brdf,
      //      wo, dg, wi); break;
      //      case MATERIAL_HAIR: c = HairMaterial__eval((HairMaterial*)material, brdf, wo, dg, wi); break;
      default:
   	   printf("Material implementation missing for embree adpater\n");

        c = Vec3fa(0.0f);
      }
    }
  }
  return c;
}
