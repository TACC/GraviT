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
 * File:   EmbreeMaterials.h
 * Author: Roberto Ribeiro
 *
 * Created on Mar 17, 2016, 3:07 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_EMBREEMATERIAL_H
#define GVT_RENDER_DATA_PRIMITIVES_EMBREEMATERIAL_H

#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/scene/Light.h>

#include <boost/container/vector.hpp>
#include <time.h>

#include <embree-shaders/common/math/vec2.h>
#include <embree-shaders/common/math/vec3.h>
#include <embree-shaders/common/math/vec3fa.h>

#include <gvt/render/data/primitives/Material.h>

#include <gvt/render/data/primitives/Shade.h>

using namespace embree;
using namespace gvt::render::actor;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;

namespace gvt {
namespace render {
namespace data {
namespace primitives {

////////////////////////////////////////////////////////////////////////////////
//                          Embree Materials
//			Shading kernels retrieved from Embree rendering samples
////////////////////////////////////////////////////////////////////////////////

struct DifferentialGeometry {
  int geomID;
  int primID;
  float u, v;
  embree::Vec3fa P;
  embree::Vec3fa Ng;
  embree::Vec3fa Ns;
  embree::Vec3fa Tx; // direction along hair
  embree::Vec3fa Ty;
  float tnear_eps;
};

struct Sample3f {
  Sample3f() {}

  Sample3f(const embree::Vec3fa &v, const float pdf) : v(v), pdf(pdf) {}

  embree::Vec3fa v;
  float pdf;
};

struct BRDF {
  float Ns;          /*< specular exponent */
  float Ni;          /*< optical density for the surface (index of refraction) */
  embree::Vec3fa Ka; /*< ambient reflectivity */
  embree::Vec3fa Kd; /*< diffuse reflectivity */
  embree::Vec3fa Ks; /*< specular reflectivity */
  embree::Vec3fa Kt; /*< transmission filter */
  float dummy[30];
};

struct Medium {
  embree::Vec3fa transmission; //!< Transmissivity of medium.
  float eta;                   //!< Refraction index of medium.
};

inline embree::Vec3fa Material__eval(Material *materials, int materialID, int numMaterials, const BRDF &brdf,
                                     const embree::Vec3fa &wo, const DifferentialGeometry &dg,
                                     const embree::Vec3fa &wi);
}
}
}
}


////////////////////////////////////////////////////////////////////////////////
//                          Embree Materials                                  //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                          Matte Material                                    //
////////////////////////////////////////////////////////////////////////////////

#include <embree-shaders/common/math/linearspace3.h>
#include <embree-shaders/common/math/math.h>
#include <embree-shaders/tutorials/pathtracer/shapesampler.h>

#include <embree-shaders/tutorials/pathtracer/optics.h>
inline Vec3fa toVec3fa(glm::vec3 &v) { return Vec3fa(v.x, v.y, v.z); }

struct Lambertian {
  Vec3fa R;
};

inline Vec3fa Lambertian__eval(const Lambertian *This, const Vec3fa &wo, const DifferentialGeometry &dg,
                               const Vec3fa &wi) {
  return This->R * /*(1.0f/(float)(float(pi))) */ clamp(dot(wi, dg.Ns));
}

inline void Lambertian__Constructor(Lambertian *This, const Vec3fa &R) { This->R = R; }

inline Lambertian make_Lambertian(const Vec3fa &R) {
  Lambertian v;
  Lambertian__Constructor(&v, R);
  return v;
}

inline Vec3fa MatteMaterial__eval(Material *This, const BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
                           const Vec3fa &wi) {
  Lambertian lambertian = make_Lambertian(Vec3fa(toVec3fa(This->kd)));
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
  const float sinThetaO = embree::sqrt(1.0f - cosThetaO * cosThetaO);
  const float horizonScatter = powf(sinThetaO, This->f);
  return (horizonScatter * cosThetaI * float(one_over_pi)) * This->R;
}

inline void Velvety__Constructor(Velvety *This, const Vec3fa &R, const float f) {
  This->R = R;
  This->f = f;
}

inline Velvety make_Velvety(const Vec3fa &R, const float f) {
  Velvety m;
  Velvety__Constructor(&m, R, f);
  return m;
}

inline void VelvetMaterial__preprocess(Material *material, BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
                                const Medium &medium) {}

inline Vec3fa VelvetMaterial__eval(Material *This, const BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
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

inline Vec3fa MetalMaterial__eval(Material *This, const BRDF &brdf, const Vec3fa &wo, const DifferentialGeometry &dg,
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
  Vec3fa c = (toVec3fa(This->ks) * F) * D * G * rcp(4.0f * cosThetaO);

  return c;
}


inline Vec3fa gvt::render::data::primitives::Material__eval(Material *materials, int materialID, int numMaterials,
                                                            const BRDF &brdf, const Vec3fa &wo,
                                                            const DifferentialGeometry &dg, const Vec3fa &wi) {
  Vec3fa c = Vec3fa(0.0f);
  int id = materialID;
  {
    if (id >= 0 && id < numMaterials) {
      Material *material = &materials[materialID];
      switch (material->type) {
      case EMBREE_MATERIAL_METAL:
        c = MetalMaterial__eval(material, brdf, wo, dg, wi);
        break;
      case EMBREE_MATERIAL_VELVET:
        c = VelvetMaterial__eval(material, brdf, wo, dg, wi);
        break;
      case EMBREE_MATERIAL_MATTE:
        c = MatteMaterial__eval(material, brdf, wo, dg, wi);
        break;
      default:
        printf("Material implementation missing for embree adpater\n");

        c = Vec3fa(0.0f);
      }
    }
  }
  return c;
}

#endif
