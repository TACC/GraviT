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
 * File:   BBox.h
 * Author: jbarbosa
 *
 * Created on February 27, 2014, 2:30 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_BBOX_H
#define GVT_RENDER_DATA_PRIMITIVES_BBOX_H

#include <fstream>
#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>

namespace gvt {
namespace render {
namespace data {
namespace primitives {

template <typename T> inline T fastmin(const T &a, const T &b) { return (a < b) ? a : b; }

template <typename T> inline T fastmax(const T &a, const T &b) { return (a > b) ? a : b; }

/// bounding box for data and acceleration structures
class Box3D {
public:
  glm::vec3 bounds_min;
  float aa0;
  glm::vec3 bounds_max;
  float aa1;

  Box3D();
  Box3D(glm::vec3 vmin, glm::vec3 vmax);

  Box3D(const Box3D &other);
  inline bool intersectDistance(const glm::vec3 &origin, const glm::vec3 &inv, float &t) const {
    glm::vec3 l = (bounds_min - origin) * inv;
    glm::vec3 u = (bounds_max - origin) * inv;
    glm::vec3 m = glm::min(l, u);
    float tmin = fastmax(fastmax(m.x, m.y), m.z);

    if (tmin < gvt::render::actor::Ray::RAY_EPSILON) return false;

    glm::vec3 M = glm::max(l, u);

    float tmax = fastmin(fastmin(M.x, M.y), M.z);

    if (tmax < 0 || tmin > tmax) return false;
    t = (tmin > 0) ? tmin : -1;

    return (t > gvt::render::actor::Ray::RAY_EPSILON);
  };
  bool intersectDistance(const glm::vec3 &origin, const glm::vec3 &inv, float &tmin, float &tmax) const;
  void merge(const Box3D &other);
  void expand(glm::vec3 &v);
  int wideRangingBoxDir() const;
  glm::vec3 centroid() const;
  float surfaceArea() const;

  Box3D transform(glm::mat4 m);

  friend std::ostream &operator<<(std::ostream &os, const Box3D &bbox) {
    os << bbox.bounds_min << " x ";
    os << bbox.bounds_max;
    return os;
  }

  template <typename cast> operator cast() {
    GVT_ASSERT(false, "Cast operator not available from gvt::render::data::primitives::BBox");
  }
};
}
}
}
}
#endif /* GVT_RENDER_DATA_PRIMITIVES_BBOX_H */
