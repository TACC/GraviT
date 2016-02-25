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

#include <gvt/render/actor/Ray.h>

#include <boost/foreach.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <boost/pool/pool_alloc.hpp>

using namespace gvt::core::math;
using namespace gvt::render::actor;

const float Ray::RAY_EPSILON = 1.e-6;

Ray::Ray(Point4f origin, Vector4f direction, float contribution, RayType type, int depth)
    : type(type), w(contribution), depth(depth) {

  this->origin = origin;
  this->direction = (direction).normalize();
  setDirection(direction);
  t = FLT_MAX;
  id = -1;
}

Ray::Ray(Ray &ray, AffineTransformMatrix<float> &m) {
  origin = m * ray.origin;
  direction = m * (ray.direction).normalize();
  setDirection(direction);
  t = ray.t;
  color = ray.color;
  domains = ray.domains;
  id = ray.id;
  type = ray.type;
  w = ray.w;
  depth = ray.depth;
}

Ray::Ray(const Ray &ray) {
  //  origin = ray.origin;
  //  direction = ray.direction;
  //  inverseDirection = ray.inverseDirection;
  //  t = ray.t;
  //  color = ray.color;
  //  domains = ray.domains;
  //  id = ray.id;
  //  w = ray.w;
  //  type = ray.type;
  //  depth = ray.depth;
  std::memcpy(data, ray.data, 16 * 4 + 8 * 4);
  domains = ray.domains;
}

Ray::Ray(Ray &&ray) {
  //    std::memmove(&origin,&ray.origin,sizeof(Point4f));
  //    std::memmove(&direction,&ray.direction,sizeof(Vector4f));
  //    std::memmove(&inverseDirection,&ray.inverseDirection,sizeof(Vector4f));
  //    std::memmove(&color,&ray.color,sizeof(Vector4f));
  std::memmove(data, ray.data, 16 * 4 + 8 * 4);
  std::swap(domains, ray.domains);
  //  t = ray.t;
  //  id = ray.id;
  //  w = ray.w;
  //  type = ray.type;
  //  depth = ray.depth;
}

Ray::~Ray() {}

Ray::Ray(const unsigned char *buf) {
  // GVT_DEBUG(DBG_ALWAYS, "in Ray::Ray(const unsigned char* buf)");
  origin = Vector4f((float *)buf);
  buf += origin.packedSize();
  direction = Vector4f((float *)buf);
  buf += direction.packedSize();
  id = *((int *)buf);
  buf += sizeof(int);
  depth = *((int *)buf);
  buf += sizeof(int);
  type = *((int *)buf);
  buf += sizeof(int);
  w = *((double *)buf);
  buf += sizeof(double);
  t = *((double *)buf);
  buf += sizeof(double);
  color = GVT_COLOR_ACCUM(buf);
  buf += color.packedSize();
  int domain_size = *((int *)buf);
  buf += sizeof(int);
  for (int i = 0; i < domain_size; ++i, buf += sizeof(isecDom)) {
    domains.push_back(isecDom(*(int *)buf, *(float *)((int *)buf + sizeof(int))));
  }
}

int Ray::packedSize() {
  int total_size = origin.packedSize() + direction.packedSize() + color.packedSize();
  total_size += 4 * sizeof(int) + 2 * sizeof(double);
  total_size += domains.size() * (sizeof(float) + sizeof(int));
  // total_size += sizeof (isecDom) * domains.size();
  return total_size;
}

int Ray::pack(unsigned char *buffer) {

  unsigned char *buf = buffer;
  unsigned char *buf0 = buffer;

  buf += origin.pack(buf);
  buf += direction.pack(buf);
  *((int *)buf) = id;
  buf += sizeof(int);
  *((int *)buf) = depth;
  buf += sizeof(int);
  *((int *)buf) = type;
  buf += sizeof(int);
  *((double *)buf) = w;
  buf += sizeof(double);
  *((double *)buf) = t;
  buf += sizeof(double);
  buf += color.pack(buf);
  *((int *)buf) = domains.size();
  buf += sizeof(int);

  for (auto &dom : domains) {
    *((int *)buf) = dom;
    buf += sizeof(int);
    *((float *)buf) = dom;
    buf += sizeof(float);
  }

  if (packedSize() != (buf - buf0)) {
    std::cout << " error in pack " << buf - buf0 << " " << packedSize() << std::endl;
    exit(0);
  }
  return packedSize();
}

void Ray::setDirection(Vector4f dir) {
  inverseDirection[3] = 0;
  dir[3] = 0;
  direction = dir.normalize();
  for (int i = 0; i < 3; i++) {
    if (direction[i] != 0)
      inverseDirection[i] = 1.0 / direction[i];
    else
      inverseDirection[i] = 0.;
  }
}

void Ray::setDirection(double *dir) { setDirection(Vector4f(dir[0], dir[1], dir[2], dir[3])); }

void Ray::setDirection(float *dir) { setDirection(Vector4f(dir[0], dir[1], dir[2], dir[3])); }
