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
 * File:   Ray.h
 * Author: jbarbosa
 *
 * Created on March 28, 2014, 1:29 PM
 */

#ifndef GVT_RENDER_ACTOR_RAY_H
#define GVT_RENDER_ACTOR_RAY_H
#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/render/data/scene/ColorAccumulator.h>

#include <limits>
#include <boost/aligned_storage.hpp>
#include <boost/container/set.hpp>
#include <boost/container/vector.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/smart_ptr.hpp>

#include <vector>

namespace gvt {
namespace render {
namespace actor {
/// container for intersection point information
typedef struct intersection {
  int domain; /// domain in which the intersection occurred
  float d;    /// distance to the intersection point

  intersection(int dom) : domain(dom), d(FLT_MAX) {}
  intersection(int dom, float dist) : domain(dom), d(dist) {}

  /// return the id of the intersected domain
  operator int() { return domain; }
  /// return the distance to the intersection point
  operator float() { return d; }
  friend inline bool operator==(const intersection &lhs, const intersection &rhs) {
    return (lhs.d == rhs.d) && (lhs.d == rhs.d);
  }
  friend inline bool operator<(const intersection &lhs, const intersection &rhs) {
    return (lhs.d < rhs.d) || ((lhs.d == rhs.d) && (lhs.domain < rhs.domain));
  }

} isecDom;

typedef std::vector<isecDom> isecDomList;

class Ray {
public:
  /// ray type
  /** ray type enumeration
   - PRIMARY - a camera or eye ray
   - SHADOW - a ray that tests visibility from a light source to an intersection point
   - SECONDARY - all other rays
   */
  // clang-format off
  enum RayType {
    PRIMARY,
    SHADOW,
    SECONDARY
  };

  const static float RAY_EPSILON;
  // clang-format on

  Ray(glm::vec4 origin = glm::vec4(0, 0, 0, 1),
      glm::vec4 direction = glm::vec4(0, 0, 0, 0), float contribution = 1.f,
      RayType type = PRIMARY, int depth = 10);
  Ray(Ray &ray, glm::mat4 &m);
  Ray(const Ray &orig);
  Ray(Ray &&ray);
  Ray(const unsigned char *buf);

  Ray operator=(const Ray &r) { return std::move(Ray(r)); }

  virtual ~Ray();

  void setDirection(glm::vec4 dir);
  void setDirection(double *dir);
  void setDirection(float *dir);

  /// returns size in bytes for the ray information to be sent via MPI
  int packedSize();

  /// packs the ray information onto the given buffer and returns the number of bytes packed
  int pack(unsigned char *buffer);

  friend std::ostream &operator<<(std::ostream &stream, Ray const &ray) {
    stream << ray.origin << "-->" << ray.direction << " [" << ray.type << "]";
    return stream;
  }

  union {
    struct {
      mutable glm::vec4 origin;
      mutable glm::vec4 direction;
      mutable glm::vec4 inverseDirection;
      mutable GVT_COLOR_ACCUM color;
      int id;    ///<! index into framebuffer
      int depth; ///<! sample rate
      float w;   ///<! weight of image contribution
      mutable float t;
      mutable float t_min;
      mutable float t_max;
      int type;
    };
    unsigned char data[16 * 4 + 7 * 4];
  };
  isecDomList domains;

protected:
};

// NOTE: removing boost pool allocator greatly improves timings
typedef std::vector<Ray> RayVector;
}
}
}

#endif /* GVT_RENDER_ACTOR_RAY_H */
