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
 * File:   Light.h
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:18 PM
 */

#ifndef GVT_RENDER_DATA_SCENE_LIGHT_H
#define GVT_RENDER_DATA_SCENE_LIGHT_H

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/primitives/BBox.h>
#include <gvt/core/Math.h>

namespace gvt {
namespace render {
namespace data {
namespace scene {
/// base class for light sources
/** \sa AmbientLight, PointLight
*/
class Light {
public:

  enum LightType 
  { 
    Point, 
    Area 
  }; 

  Light(const gvt::core::math::Point4f position);
  Light(const Light &orig);
  virtual ~Light();

  virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray &ray) const;

  

  virtual gvt::render::data::primitives::Box3D getWorldBoundingBox() {
    gvt::render::data::primitives::Box3D bb(position, position);
    return bb;
  }
  LightType LightT;
  gvt::core::math::Point4f position;  
};
/// general lighting factor added to each successful ray intersection
class AmbientLight : public Light {
public:
  AmbientLight(const gvt::core::math::Vector4f color);
  AmbientLight(const AmbientLight &orig);
  virtual ~AmbientLight();

  virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray &ray) const;

  gvt::core::math::Vector4f color;
};
/// point light source
class PointLight : public Light {
public:
  PointLight(const gvt::core::math::Point4f position, const gvt::core::math::Vector4f color);
  PointLight(const PointLight &orig);
  virtual ~PointLight();

  virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray &ray) const;

  gvt::core::math::Vector4f color; 
};

class AreaLight : public Light {
public:
  AreaLight(const gvt::core::math::Point4f position, const gvt::core::math::Vector4f color
            ,gvt::core::math::Vector4f lightNormal, float lightHeight, float lightWidth);
  AreaLight(const AreaLight &orig);
  virtual ~AreaLight();

  virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray &ray) const;

  gvt::core::math::Point4f virtual GetPosition();

  gvt::core::math::Vector4f color;
  gvt::core::math::Vector4f LightNormal;
  float LightWidth;
  float LightHeight;
  inline int fastrand() 
  {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
  }

protected:
  unsigned int g_seed;
  inline void fast_srand(int seed) { g_seed = seed; }
  gvt::core::math::Vector4f u,v,w;
};
}
}
}
}

#endif /* GVT_RENDER_DATA_SCENE_LIGHT_H */
