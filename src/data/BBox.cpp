/* ========================================================================== //
 * This file is released as part of GraviT2 - scalable, platform independent  //
 * ray tracing tacc.github.io/GraviT2                                         //
 *                                                                            //
 * Copyright (c) 2013-2019 The University of Texas at Austin.                 //
 * All rights reserved.                                                       //
 *                                                                            //
 * Licensed under the Apache License, Version 2.0 (the "License");            //
 * you may not use this file except in compliance with the License.           //
 * A copy of the License is included with this software in the file LICENSE.  //
 * If your copy does not contain the License, you may obtain a copy of the    //
 * License at:                                                                //
 *                                                                            //
 *     https://www.apache.org/licenses/LICENSE-2.0                            //
 *                                                                            //
 * Unless required by applicable law or agreed to in writing, software        //
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT  //
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.           //
 * See the License for the specific language governing permissions and        //
 * limitations under the License.                                             //
 * ========================================================================== */
#include "BBox.h"
#include "Ray.h"
#include "vec.h"
#include <limits>

using namespace gvt2;

Box3D::Box3D() {
  for (int i = 0; i < 3; ++i) {
    bounds_min[i] = std::numeric_limits<float>::max();
    bounds_max[i] = -std::numeric_limits<float>::max();
  }
}

Box3D::Box3D(vec3f vmin, vec3f vmax) {
  for (int i = 0; i < 3; i++) {
    bounds_min[i] = fastmin(vmin[i], vmax[i]);
    bounds_max[i] = fastmax(vmin[i], vmax[i]);
  }
}

Box3D::Box3D(const Box3D &other) {
  for (int i = 0; i < 3; i++) {
    bounds_min[i] = fastmin(other.bounds_min[i], other.bounds_max[i]);
    bounds_max[i] = fastmax(other.bounds_min[i], other.bounds_max[i]);
  }
}

void Box3D::merge(const Box3D &other) {
  bounds_min[0] = fastmin(other.bounds_min[0], bounds_min[0]);
  bounds_min[1] = fastmin(other.bounds_min[1], bounds_min[1]);
  bounds_min[2] = fastmin(other.bounds_min[2], bounds_min[2]);

  bounds_max[0] = fastmax(other.bounds_max[0], bounds_max[0]);
  bounds_max[1] = fastmax(other.bounds_max[1], bounds_max[1]);
  bounds_max[2] = fastmax(other.bounds_max[2], bounds_max[2]);
}

void Box3D::expand(vec3f &v) {
  bounds_min[0] = fastmin(bounds_min[0], v[0]);
  bounds_min[1] = fastmin(bounds_min[1], v[1]);
  bounds_min[2] = fastmin(bounds_min[2], v[2]);

  bounds_max[0] = fastmax(bounds_max[0], v[0]);
  bounds_max[1] = fastmax(bounds_max[1], v[1]);
  bounds_max[2] = fastmax(bounds_max[2], v[2]);
}

bool Box3D::intersectDistance(const vec3f &origin, const vec3f &inv, float &tmin, float &tmax) const {

  float t1 = (bounds_min.x - origin.x) * inv.x;

  float t3 = (bounds_min.y - origin.y) * inv.y;
  float t5 = (bounds_min.z - origin.z) * inv.z;
  float t2 = (bounds_max.x - origin.x) * inv.x;
  float t4 = (bounds_max.y - origin.y) * inv.y;
  float t6 = (bounds_max.z - origin.z) * inv.z;

  tmin = fastmax(fastmax(fastmin(t1, t2), fastmin(t3, t4)), fastmin(t5, t6));
  tmax = fastmin(fastmin(fastmax(t1, t2), fastmax(t3, t4)), fastmax(t5, t6));
  if (tmax < 0 || tmin > tmax) return false;
  // t = (tmin > 0) ? tmin : tmax;
  return true; //(t > gvt::render::actor::Ray::RAY_EPSILON);
}

// returns dimension with maximum extent
int Box3D::wideRangingBoxDir() const {
  vec3f diag = bounds_max - bounds_min;
  if (diag.x > diag.y && diag.x > diag.z)
    return 0; // x-axis
  else if (diag.y > diag.z)
    return 1; // y-axis
  else
    return 2; // z-axis
}

vec3f Box3D::centroid() const { return (0.5f * bounds_min + 0.5f * bounds_max); }

float Box3D::surfaceArea() const {
  vec3f diag = bounds_max - bounds_min;
  return (2.f * (diag.x * diag.y + diag.y * diag.z + diag.z * diag.x));
}

Box3D Box3D::transform(glm::mat4 m) {
  Box3D newBox;
  const vec3 min = bounds_min;
  const vec3 max = bounds_max;

  vec3f v;
  v = vec3f(m * vec4f(min.x, min.y, min.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(min.x, min.y, max.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(min.x, max.y, min.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(min.x, max.y, max.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(max.x, min.y, min.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(max.x, min.y, max.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(max.x, max.y, min.z, 1.0f));
  newBox.expand(v);
  v = vec3f(m * vec4f(max.x, max.y, max.z, 1.0f));
  newBox.expand(v);

  return newBox;
}
