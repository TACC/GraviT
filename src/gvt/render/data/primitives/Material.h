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
 * File:   Material.h
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:07 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_MATERIAL_H
#define GVT_RENDER_DATA_PRIMITIVES_MATERIAL_H

#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/scene/Light.h>

#include <boost/container/vector.hpp>
#include <time.h>



#include "../embree/common/math/vec3.h"
#include "../embree/common/math/vec3fa.h"
#include "../embree/common/math/vec2.h"

//#include "../embree/tutorials/common/tutorial/tutorial_device.h"
//#include "../embree/tutorials/common/tutorial/scene_device.h"
//#include "../embree/tutorials/common/tutorial/random_sampler.h"
//#include "../embree/tutorials/pathtracer/shapesampler.h"

namespace gvt {
namespace render {
namespace data {
namespace primitives {

struct Material {
  Material(){

  }

  Material(Material* m){
    type=m->type;
    memcpy(buf, m->buf,m->size());
  }

  /*
   * This will receive a specialized material and copu the data of the material
   * to our base Material
   */
  template <typename T>
  Material(T om){

    Material * m = (Material*)&om;

    type=m->type;
    memcpy(buf, m->buf,m->size());
  }


  inline int size(){
    return 992;
  }

  int type;
  unsigned char buf[992]; //comply with Embree worst case

};


glm::vec3 Shade(
     gvt::render::data::primitives::Material* material,
                        const gvt::render::actor::Ray &ray,
                        const glm::vec3 &sufaceNormal,
                        const gvt::render::data::scene::Light *lightSource,
                        const glm::vec3 lightPostion);

typedef enum
{
  EMBREE_MATERIAL_OBJ,
  EMBREE_MATERIAL_THIN_DIELECTRIC,
  EMBREE_MATERIAL_METAL,
  EMBREE_MATERIAL_VELVET,
  EMBREE_MATERIAL_DIELECTRIC,
  EMBREE_MATERIAL_METALLIC_PAINT,
  EMBREE_MATERIAL_MATTE,
  EMBREE_MATERIAL_MIRROR,
  EMBREE_MATERIAL_REFLECTIVE_METAL,
  EMBREE_MATERIAL_HAIR,
  GVT_LAMBERT,
  GVT_PHONG,
  GVT_BLINN
} MATERIAL_TYPE;



////////////////////////////////////////////////////////////////////////////////
//                          GVT Materials                                     //
////////////////////////////////////////////////////////////////////////////////



struct Lambert {
  Lambert(glm::vec3 v){
    kd = v;
    type = GVT_LAMBERT;
  }

  MATERIAL_TYPE type;
  glm::vec3 kd;
};

struct Phong {
  Phong(const glm::vec3 &_kd, const glm::vec3 &_ks, const float &_alpha){
    kd = _kd;
    ks=_ks;
    alpha=_alpha;
    type = GVT_PHONG;
  }

  MATERIAL_TYPE type;
  glm::vec3 kd;
  glm::vec3 ks;
  float alpha;

};

struct Blinn {
  Blinn(const glm::vec3 &_kd, const glm::vec3 &_ks, const float &_alpha){
    kd = _kd;
    ks=_ks;
    alpha=_alpha;
    type = GVT_BLINN;
  }

  MATERIAL_TYPE type;
  glm::vec3 kd;
  glm::vec3 ks;
  float alpha;

};

glm::vec3 MaterialShade(
    const gvt::render::data::primitives::Material* material,
                        const gvt::render::actor::Ray &ray,
                        const glm::vec3 &sufaceNormal,
                        const gvt::render::data::scene::Light *lightSource,
                        const glm::vec3 lightPostion) ;

////////////////////////////////////////////////////////////////////////////////
//                          Embree Materials                                  //
////////////////////////////////////////////////////////////////////////////////

using namespace embree;



struct DifferentialGeometry
{
  int geomID;
  int primID;
  float u,v;
  Vec3fa P;
  Vec3fa Ng;
  Vec3fa Ns;
  Vec3fa Tx; //direction along hair
  Vec3fa Ty;
  float tnear_eps;
};

struct Sample3f
{
  Sample3f () {}

  Sample3f (const Vec3fa& v, const float pdf)
    : v(v), pdf(pdf) {}

  Vec3fa v;
  float pdf;
};

struct BRDF
{
  float Ns;               /*< specular exponent */
  float Ni;               /*< optical density for the surface (index of refraction) */
  Vec3fa Ka;              /*< ambient reflectivity */
  Vec3fa Kd;              /*< diffuse reflectivity */
  Vec3fa Ks;              /*< specular reflectivity */
  Vec3fa Kt;              /*< transmission filter */
  float dummy[30];
};

struct Medium
{
  Vec3fa transmission; //!< Transmissivity of medium.
  float eta;             //!< Refraction index of medium.
};

inline Vec3fa Material__eval(Material* materials,
                             int materialID,
                             int numMaterials,
                             const BRDF& brdf,
                             const Vec3fa& wo,
                             const DifferentialGeometry& dg,
                             const Vec3fa& wi);

struct MatteMaterial
{
public:

  MatteMaterial (glm::vec3 v)
  : ty(EMBREE_MATERIAL_MATTE), reflectance(Vec3fa(v[0],v[1],v[2])) {}

  MatteMaterial (const Vec3fa& reflectance)
  : ty(EMBREE_MATERIAL_MATTE), reflectance(reflectance) {}

public:
  int ty;
  int align[3];
  Vec3fa reflectance;
};

struct MetalMaterial
{
public:

  MetalMaterial (glm::vec3 v, glm::vec3 e, glm::vec3 kk, float r)
  : ty(EMBREE_MATERIAL_METAL), reflectance(Vec3fa(v[0],v[1],v[2])),
    eta(Vec3fa(e[0],e[1],e[2])),
    k(Vec3fa(kk[0],kk[1],kk[2])),
    roughness(r) {}

  MetalMaterial (const Vec3fa& reflectance, const Vec3fa& eta, const Vec3fa& k)
  : ty(EMBREE_MATERIAL_REFLECTIVE_METAL), reflectance(reflectance), eta(eta), k(k), roughness(0.0f) {}

  MetalMaterial (const Vec3fa& reflectance, const Vec3fa& eta, const Vec3fa& k, const float roughness)
  : ty(EMBREE_MATERIAL_METAL), reflectance(reflectance), eta(eta), k(k), roughness(roughness) {}

public:
  int ty;
  int align[3];

  Vec3fa reflectance;
  Vec3fa eta;
  Vec3fa k;
  float roughness;
};

struct VelvetMaterial
{

  VelvetMaterial (const glm::vec3 reflectance, const float backScattering, const glm::vec3 horizonScatteringColor, const float horizonScatteringFallOff)
  : ty(EMBREE_MATERIAL_VELVET), reflectance(reflectance[0], reflectance[1],reflectance[2]),
    backScattering(backScattering), horizonScatteringColor(horizonScatteringColor[0],horizonScatteringColor[1],horizonScatteringColor[2]),
    horizonScatteringFallOff(horizonScatteringFallOff) {}

  VelvetMaterial (const Vec3fa& reflectance, const float backScattering, const Vec3fa& horizonScatteringColor, const float horizonScatteringFallOff)
  : ty(EMBREE_MATERIAL_VELVET), reflectance(reflectance), backScattering(backScattering), horizonScatteringColor(horizonScatteringColor), horizonScatteringFallOff(horizonScatteringFallOff) {}

public:
  int ty;
  int align[3];

  Vec3fa reflectance;
  Vec3fa horizonScatteringColor;
  float backScattering;
  float horizonScatteringFallOff;
};

}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_MATERIAL_H */
