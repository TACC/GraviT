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
 * File:   Ray.cpp
 * Author: jbarbosa
 *
 * Created on March 28, 2014, 1:29 PM
 */

#include <glm/ext.hpp>

#include <gvt/render/actor/Ray.h>

#include <boost/foreach.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/pool/singleton_pool.hpp>

using namespace gvt::render::actor;

const float Ray::RAY_EPSILON = 1.e-6;

Ray::Ray(glm::vec3 origin, glm::vec3 direction, float contribution, RayType type, int depth)
    : type(type), w(contribution), depth(depth) {

  this->origin = origin;
  setDirection(direction);
  t = FLT_MAX;
  id = -1;
}

Ray::Ray(Ray &ray, glm::mat4 &m) {
  std::memcpy(data, ray.data, packedSize());
  origin = glm::vec3(m * glm::vec4(ray.origin, 1.f));
  setDirection(glm::vec3(m * glm::vec4(ray.direction, 0.f)));
}

Ray::Ray(const Ray &ray) { std::memcpy(data, ray.data, packedSize()); }

Ray::Ray(Ray &&ray) { std::memmove(data, ray.data, packedSize()); }

Ray::~Ray() {}

Ray::Ray(const unsigned char *buf) { std::memcpy(data, buf, packedSize()); }

int Ray::pack(unsigned char *buffer) {
  unsigned char *buf = buffer;
  std::memcpy(buf, data, packedSize());
  return packedSize();
}

void Ray::setDirection(glm::vec3 dir) {
  direction = glm::fastNormalize(dir);
  for (int i = 0; i < 3; i++) {
    if (direction[i] != 0)
      inverseDirection[i] = 1.0 / direction[i];
    else
      inverseDirection[i] = 0.;
  }
}

void Ray::setDirection(double *dir) { setDirection(glm::vec3(dir[0], dir[1], dir[2])); }

void Ray::setDirection(float *dir) { setDirection(glm::vec3(dir[0], dir[1], dir[2])); }
