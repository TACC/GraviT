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

#include "gvt/render/data/primitives/BBox.h"

#include "gvt/render/actor/Ray.h"

#include <limits>

using namespace gvt::render::actor;
using namespace gvt::render::data::primitives;

template <typename T> inline T fastmin(const T &a, const T &b) { return (a < b) ? a : b; }

template <typename T> inline T fastmax(const T &a, const T &b) { return (a > b) ? a : b; }

Box3D::Box3D() {
  for (int i = 0; i < 3; ++i) {
    bounds[0][i] = std::numeric_limits<float>::max();
    bounds[1][i] = -std::numeric_limits<float>::max();
  }
}

Box3D::Box3D(glm::vec3 vmin, glm::vec3 vmax) {
  for (int i = 0; i < 3; i++) {
    bounds[0][i] = fastmin(vmin[i], vmax[i]);
    bounds[1][i] = fastmax(vmin[i], vmax[i]);
  }
}

Box3D::Box3D(const Box3D &other) {
  for (int i = 0; i < 3; i++) {
    bounds[0][i] = fastmin(other.bounds[0][i], other.bounds[1][i]);
    bounds[1][i] = fastmax(other.bounds[0][i], other.bounds[1][i]);
  }
}

void Box3D::merge(const Box3D &other) {
  bounds[0][0] = fastmin(other.bounds[0][0], bounds[0][0]);
  bounds[0][1] = fastmin(other.bounds[0][1], bounds[0][1]);
  bounds[0][2] = fastmin(other.bounds[0][2], bounds[0][2]);

  bounds[1][0] = fastmax(other.bounds[1][0], bounds[1][0]);
  bounds[1][1] = fastmax(other.bounds[1][1], bounds[1][1]);
  bounds[1][2] = fastmax(other.bounds[1][2], bounds[1][2]);
}

void Box3D::expand(glm::vec3 &v) {
  bounds[0][0] = fastmin(bounds[0][0], v[0]);
  bounds[0][1] = fastmin(bounds[0][1], v[1]);
  bounds[0][2] = fastmin(bounds[0][2], v[2]);

  bounds[1][0] = fastmax(bounds[1][0], v[0]);
  bounds[1][1] = fastmax(bounds[1][1], v[1]);
  bounds[1][2] = fastmax(bounds[1][2], v[2]);
}

bool Box3D::intersectDistance(const glm::vec3 &origin, const glm::vec3 &inv, float &t) const {

  /*float t1 = (bounds[0].x - origin.x) * inv.x;
  float t3 = (bounds[0].y - origin.y) * inv.y;
  float t5 = (bounds[0].z - origin.z) * inv.z;
  float t2 = (bounds[1].x - origin.x) * inv.x;
  float t4 = (bounds[1].y - origin.y) * inv.y;
  float t6 = (bounds[1].z - origin.z) * inv.z;
*/

  glm::vec3 l = (bounds[0] - origin) * inv;
  glm::vec3 u = (bounds[1] - origin) * inv;
  glm::vec3 m = glm::min(l, u);
  float tmin = fastmax(fastmax(m.x, m.y), m.z);
  if (tmin < FLT_EPSILON) return false;

  glm::vec3 M = glm::max(l, u);
  float tmax = fastmin(fastmin(M.x, M.y), M.z);

  if (tmax < 0 || tmin > tmax) return false;
  t = (tmin > 0) ? tmin : -1;

  return (t > FLT_EPSILON);
}

bool Box3D::intersectDistance(const glm::vec3 &origin, const glm::vec3 &inv, float &tmin, float &tmax) const {

  float t1 = (bounds[0].x - origin.x) * inv.x;

  float t3 = (bounds[0].y - origin.y) * inv.y;
  float t5 = (bounds[0].z - origin.z) * inv.z;
  float t2 = (bounds[1].x - origin.x) * inv.x;
  float t4 = (bounds[1].y - origin.y) * inv.y;
  float t6 = (bounds[1].z - origin.z) * inv.z;

  tmin = fastmax(fastmax(fastmin(t1, t2), fastmin(t3, t4)), fastmin(t5, t6));
  tmax = fastmin(fastmin(fastmax(t1, t2), fastmax(t3, t4)), fastmax(t5, t6));
  if (tmax < 0 || tmin > tmax) return false;
  // t = (tmin > 0) ? tmin : tmax;
  return true; //(t > FLT_EPSILON);
}

// returns dimension with maximum extent
int Box3D::wideRangingBoxDir() const {
  glm::vec3 diag = bounds[1] - bounds[0];
  if (diag.x > diag.y && diag.x > diag.z)
    return 0; // x-axis
  else if (diag.y > diag.z)
    return 1; // y-axis
  else
    return 2; // z-axis
}

glm::vec3 Box3D::centroid() const { return (0.5f * bounds[0] + 0.5f * bounds[1]); }

float Box3D::surfaceArea() const {
  glm::vec3 diag = bounds[1] - bounds[0];
  return (2.f * (diag.x * diag.y + diag.y * diag.z + diag.z * diag.x));
}
