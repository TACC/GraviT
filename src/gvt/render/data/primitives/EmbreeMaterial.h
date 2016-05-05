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



#include <embree-shaders/common/math/vec3.h>
#include <embree-shaders/common/math/vec3fa.h>
#include <embree-shaders/common/math/vec2.h>

#include <gvt/render/data/primitives/Material.h>

namespace gvt {
namespace render {
namespace data {
namespace primitives {


/*
 * Material proxy call implemented per adpater
 * Interfaces to the different shading materials may be significantly different
 * mainly due to light assessing and vec formats
 */
bool Shade(gvt::render::data::primitives::Material* material,
                        const gvt::render::actor::Ray &ray,
                        const glm::vec3 &sufaceNormal,
                        const gvt::render::data::scene::Light *lightSource,
                        const glm::vec3 lightPosSample,
                        glm::vec3& color);


////////////////////////////////////////////////////////////////////////////////
//                          Embree Materials
//			Shading kernels retrieved from Embree rendering samples
////////////////////////////////////////////////////////////////////////////////

struct DifferentialGeometry
{
  int geomID;
  int primID;
  float u,v;
  embree::Vec3fa P;
  embree::Vec3fa Ng;
  embree::Vec3fa Ns;
  embree::Vec3fa Tx; //direction along hair
  embree::Vec3fa Ty;
  float tnear_eps;
};

struct Sample3f
{
  Sample3f () {}

  Sample3f (const embree::Vec3fa& v, const float pdf)
    : v(v), pdf(pdf) {}

  embree::Vec3fa v;
  float pdf;
};

struct BRDF
{
  float Ns;               /*< specular exponent */
  float Ni;               /*< optical density for the surface (index of refraction) */
  embree::Vec3fa Ka;              /*< ambient reflectivity */
  embree::Vec3fa Kd;              /*< diffuse reflectivity */
  embree::Vec3fa Ks;              /*< specular reflectivity */
  embree::Vec3fa Kt;              /*< transmission filter */
  float dummy[30];
};

struct Medium
{
  embree::Vec3fa transmission; //!< Transmissivity of medium.
  float eta;             //!< Refraction index of medium.
};

inline embree::Vec3fa Material__eval(Material* materials,
                             int materialID,
                             int numMaterials,
                             const BRDF& brdf,
                             const embree::Vec3fa& wo,
                             const DifferentialGeometry& dg,
                             const embree::Vec3fa& wi);


}
}
}
}

#endif
