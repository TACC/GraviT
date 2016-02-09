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
 * File:   Light.cpp
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:18 PM
 */

#include "gvt/render/data/scene/Light.h"
#include "gvt/render/data/DerivedTypes.h"

using namespace gvt::core::math;
using namespace gvt::render::actor;
using namespace gvt::render::data::scene;

Light::Light(const Point4f position) : position(position) {}

Light::Light(const Light &orig) : position(orig.position) {}

Light::~Light() {}

Vector4f Light::contribution(const Ray &ray) const { return Color(); }

PointLight::PointLight(const Point4f position, const Vector4f color) : Light(position), color(color) {}

PointLight::PointLight(const PointLight &orig) : Light(orig), color(orig.color) {}

PointLight::~PointLight() {}

Vector4f PointLight::contribution(const Ray &ray) const {
  float distance = 1.f / ((Vector4f)position - ray.origin).length();
  distance = (distance > 1.f) ? 1.f : distance;
  return color * (distance + 0.5f);
}

AmbientLight::AmbientLight(const Vector4f color) : Light(), color(color) {}

AmbientLight::AmbientLight(const AmbientLight &orig) : Light(orig), color(orig.color) {}

AmbientLight::~AmbientLight() {}

Vector4f AmbientLight::contribution(const Ray &ray) const { return color; }
