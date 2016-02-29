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

#include <mutex>
#include <thread>

std::mutex mcout;

#include "gvt/render/data/DerivedTypes.h"
#include "gvt/render/data/scene/Light.h"

using namespace gvt::render::actor;
using namespace gvt::render::data::scene;

Light::Light(const glm::vec3 position) : position(position) {}

Light::Light(const Light &orig) : position(orig.position) {}

Light::~Light() {}

glm::vec3 Light::contribution(const Ray &ray) const { return Color(); }

PointLight::PointLight(const glm::vec3 position, const glm::vec3 color) : Light(position), color(color) {}

PointLight::PointLight(const PointLight &orig) : Light(orig), color(orig.color) {}

PointLight::~PointLight() {}

glm::vec3 PointLight::contribution(const Ray &ray) const {
  float d = glm::length(position - (ray.origin + ray.direction * ray.t));
  float att = 1.f / d; // FIX THIS it should be squared
  if (att > 1.f) att = 1.f;
  return color * att; // * distance; // + 0.5f);
}

AmbientLight::AmbientLight(const glm::vec3 color) : Light(), color(color) {}

AmbientLight::AmbientLight(const AmbientLight &orig) : Light(orig), color(orig.color) {}

AmbientLight::~AmbientLight() {}

glm::vec3 AmbientLight::contribution(const Ray &ray) const { return color; }
