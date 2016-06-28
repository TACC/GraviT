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

glm::vec3 Light::contribution(const glm::vec3 &hitpoint, const glm::vec3 &samplePos) const { return Color(); }

PointLight::PointLight(const glm::vec3 position, const glm::vec3 color) : Light(position), color(color) {
  LightT = Point;
}

PointLight::PointLight(const PointLight &orig) : Light(orig), color(orig.color) { LightT = Point; }

PointLight::~PointLight() {}

glm::vec3 PointLight::contribution(const glm::vec3 &hitpoint, const glm::vec3 &samplePos) const {
  float distance = 1.f / glm::length(position - hitpoint);
  distance = (distance > 1.f) ? 1.f : distance;
  return color * distance;
}

AmbientLight::AmbientLight(const glm::vec3 color) : Light(), color(color) {}

AmbientLight::AmbientLight(const AmbientLight &orig) : Light(orig), color(orig.color) {}

AmbientLight::~AmbientLight() {}

glm::vec3 AmbientLight::contribution(const glm::vec3 &ray, const glm::vec3 &samplePos) const { return color; }

AreaLight::AreaLight(const glm::vec3 position, const glm::vec3 color, glm::vec3 lightNormal, float lightHeight,
                     float lightWidth)
    : Light(position), color(color), LightNormal(lightNormal), LightWidth(lightWidth), LightHeight(lightHeight) {
  LightT = Area;

  v = LightNormal;
  glm::vec3 up(0, 1, 0);

  // check if v is the same as up
  if (v == up) {
    // identiy
    u[0] = 1;
    u[1] = 0;
    u[2] = 0;
    w[0] = 0;
    w[1] = 0;
    w[2] = 1;
  } else {
    u[0] = up[1] * v[2] - v[1] * up[2];
    u[1] = up[2] * v[0] - v[2] * up[0];
    u[2] = up[0] * v[1] - v[0] * up[1];

    // right cross lightNormal
    w[0] = v[1] * u[2] - u[1] * v[2];
    w[1] = v[2] * u[0] - u[2] * v[0];
    w[2] = v[0] * u[1] - u[0] * v[1];
  }
}

AreaLight::AreaLight(const AreaLight &orig) : Light(orig) {
  u = orig.u;
  v = orig.v;
  w = orig.w;

  LightT = orig.LightT;
  color = orig.color;
  LightNormal = orig.LightNormal;
  LightWidth = orig.LightWidth;
  LightHeight = orig.LightHeight;
}

AreaLight::~AreaLight() {}

glm::vec3 AreaLight::GetPosition(unsigned int *seedVal) {
  // generate points on plane then transform
  float xLocation = (randEngine.fastrand(seedVal, 0, 1) - 0.5) * LightWidth;
  float yLocation = 0;
  float zLocation = (randEngine.fastrand(seedVal, 0, 1) - 0.5) * LightHeight;

  // x coord
  float xCoord = xLocation * u[0] + zLocation * w[0];
  float yCoord = xLocation * u[1] + zLocation * w[1];
  float zCoord = xLocation * u[2] + zLocation * w[2];

  return glm::vec3(position[0] + xCoord, position[1] + yCoord, position[2] + zCoord);
}

glm::vec3 AreaLight::contribution(const glm::vec3 &hitpoint, const glm::vec3 &samplePos) const {
  float distance = 1.f / glm::length(samplePos - hitpoint);
  distance = (distance > 1.f) ? 1.f : distance;
  return color * (distance);
}
