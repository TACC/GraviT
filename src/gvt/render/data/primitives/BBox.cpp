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

int inline GetIntersection(float fDst1, float fDst2, glm::vec3 P1, glm::vec3 P2, glm::vec3 &Hit) {
  if ((fDst1 * fDst2) >= 0.0f) return 0;
  if (fDst1 == fDst2) return 0;
  Hit = P1 + (P2 - P1) * (-fDst1 / (fDst2 - fDst1));
  return 1;
}

int inline InBox(glm::vec3 Hit, glm::vec3 B1, glm::vec3 B2, const int Axis) {
  if (Axis == 1 && Hit.z > B1.z && Hit.z < B2.z && Hit.y > B1.y && Hit.y < B2.y) return 1;
  if (Axis == 2 && Hit.z > B1.z && Hit.z < B2.z && Hit.x > B1.x && Hit.x < B2.x) return 1;
  if (Axis == 3 && Hit.x > B1.x && Hit.x < B2.x && Hit.y > B1.y && Hit.y < B2.y) return 1;
  return 0;
}

template <typename T> inline T fastmin(const T &a, const T &b) { return (a < b) ? a : b; }

template <typename T> inline T fastmax(const T &a, const T &b) { return (a > b) ? a : b; }

// returns true if line (L1, L2) intersects with the box (B1, B2)
// returns intersection point in Hit
int inline CheckLineBox(glm::vec3 B1, glm::vec3 B2, glm::vec3 L1, glm::vec3 L2, glm::vec3 &Hit) {
  if (L2.x < B1.x && L1.x < B1.x) return false;
  if (L2.x > B2.x && L1.x > B2.x) return false;
  if (L2.y < B1.y && L1.y < B1.y) return false;
  if (L2.y > B2.y && L1.y > B2.y) return false;
  if (L2.z < B1.z && L1.z < B1.z) return false;
  if (L2.z > B2.z && L1.z > B2.z) return false;
  if (L1.x > B1.x && L1.x < B2.x && L1.y > B1.y && L1.y < B2.y && L1.z > B1.z && L1.z < B2.z) {
    Hit = L1;
    return true;
  }
  if ((GetIntersection(L1.x - B1.x, L2.x - B1.x, L1, L2, Hit) && InBox(Hit, B1, B2, 1)) ||
      (GetIntersection(L1.y - B1.y, L2.y - B1.y, L1, L2, Hit) && InBox(Hit, B1, B2, 2)) ||
      (GetIntersection(L1.z - B1.z, L2.z - B1.z, L1, L2, Hit) && InBox(Hit, B1, B2, 3)) ||
      (GetIntersection(L1.x - B2.x, L2.x - B2.x, L1, L2, Hit) && InBox(Hit, B1, B2, 1)) ||
      (GetIntersection(L1.y - B2.y, L2.y - B2.y, L1, L2, Hit) && InBox(Hit, B1, B2, 2)) ||
      (GetIntersection(L1.z - B2.z, L2.z - B2.z, L1, L2, Hit) && InBox(Hit, B1, B2, 3)))
    return true;

  return false;
}

Box3D::Box3D() {
  for (int i = 0; i < 3; ++i) {
    bounds[0][i] = std::numeric_limits<float>::max();
    bounds[1][i] = -std::numeric_limits<float>::max();
  }
}

glm::vec3 Box3D::getHitpoint(const Ray &ray) const {
  glm::vec3 hit;
  CheckLineBox(bounds[0], bounds[1], ray.origin, (glm::vec3)((glm::vec3)ray.origin + ray.direction * 1.e6f), hit);
  return hit;
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

bool Box3D::intersect(const Ray &r) const {
  float t;
  return intersectDistance(r, t);
}

bool Box3D::inBox(const Ray &r) const { return inBox(r.origin); }

bool Box3D::inBox(const glm::vec3 &origin) const {
  bool TT[3];

  TT[0] = ((bounds[0].x - origin.x) <= FLT_EPSILON && (bounds[1].x - origin.x) >= -FLT_EPSILON);
  if (!TT[0]) return false;
  TT[1] = ((bounds[0].y - origin.y) <= FLT_EPSILON && (bounds[1].y - origin.y) >= -FLT_EPSILON);
  if (!TT[0]) return false;
  TT[2] = ((bounds[0].z - origin.z) <= FLT_EPSILON && (bounds[1].z - origin.z) >= -FLT_EPSILON);
  if (!TT[0]) return false;
  return (TT[0] && TT[1] && TT[2]);
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

bool Box3D::intersectDistance(const Ray &ray, float &t) const {

  /*float t1 = (bounds[0].x - ray.origin.x) * ray.inverseDirection.x;
  float t3 = (bounds[0].y - ray.origin.y) * ray.inverseDirection.y;
  float t5 = (bounds[0].z - ray.origin.z) * ray.inverseDirection.z;
  float t2 = (bounds[1].x - ray.origin.x) * ray.inverseDirection.x;
  float t4 = (bounds[1].y - ray.origin.y) * ray.inverseDirection.y;
  float t6 = (bounds[1].z - ray.origin.z) * ray.inverseDirection.z;
*/

  glm::vec3 l = (bounds[0] - ray.origin) * ray.inverseDirection;
  glm::vec3 u = (bounds[1] - ray.origin) * ray.inverseDirection;
  glm::vec3 m = glm::min(l,u);
  glm::vec3 M = glm::max(l,u);

  float tmin = fastmax(fastmax(m.x,m.y),m.z);
  float tmax = fastmin(fastmin(M.x,M.y),M.z);

  if (tmax < 0 || tmin > tmax) return false;

  t = (tmin > 0) ? tmin : -1;

  return (t > FLT_EPSILON);
}

bool Box3D::intersectDistance(const Ray &ray, float &tmin, float &tmax) const {

  float t1 = (bounds[0].x - ray.origin.x) * ray.inverseDirection.x;

  float t3 = (bounds[0].y - ray.origin.y) * ray.inverseDirection.y;
  float t5 = (bounds[0].z - ray.origin.z) * ray.inverseDirection.z;
  float t2 = (bounds[1].x - ray.origin.x) * ray.inverseDirection.x;
  float t4 = (bounds[1].y - ray.origin.y) * ray.inverseDirection.y;
  float t6 = (bounds[1].z - ray.origin.z) * ray.inverseDirection.z;

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
