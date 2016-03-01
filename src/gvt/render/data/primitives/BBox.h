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

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <fstream>

namespace gvt {
namespace render {
namespace data {
namespace primitives {
/// bounding box for data and acceleration structures
class Box3D {
public:
  glm::vec3 bounds[2];

  Box3D();
  Box3D(glm::vec3 vmin, glm::vec3 vmax);

  Box3D(const Box3D &other);
  bool intersect(const gvt::render::actor::Ray &r) const;
  bool intersect(const gvt::render::actor::Ray &r, float &tmin, float &tmax) const;
  bool inBox(const gvt::render::actor::Ray &r) const;
  bool inBox(const glm::vec3 &r) const;
  glm::vec3 getHitpoint(const gvt::render::actor::Ray &r) const;
  bool intersectDistance(const gvt::render::actor::Ray &r, float &t) const;
  bool intersectDistance(const gvt::render::actor::Ray &r, float &tmin, float &tmax) const;
  void merge(const Box3D &other);
  void expand(glm::vec3 &v);
  int wideRangingBoxDir() const;
  glm::vec3 centroid() const;
  float surfaceArea() const;

  friend std::ostream &operator<<(std::ostream &os, const Box3D &bbox) {
    os << bbox.bounds[0] << " x ";
    os << bbox.bounds[1];
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
