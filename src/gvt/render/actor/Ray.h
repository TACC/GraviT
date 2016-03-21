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

#include <boost/aligned_storage.hpp>
#include <boost/container/set.hpp>
#include <boost/container/vector.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/smart_ptr.hpp>
#include <limits>

#include <iomanip>
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
  inline Ray() /*: t_min(gvt::render::actor::Ray::RAY_EPSILON), t_max(FLT_MAX), t(FLT_MAX), id(-1), depth(8), w(0.f), type(PRIMARY) */ {
  }
  inline Ray(glm::vec3 _origin, glm::vec3 _direction, float contribution = 1.f, RayType type = PRIMARY, int depth = 10)
      : origin(_origin), t_min(gvt::render::actor::Ray::RAY_EPSILON), direction(glm::normalize(_direction)),
        t_max(FLT_MAX), t(FLT_MAX), id(-1), w(contribution), type(type) {}

  inline Ray(const Ray &r) { std::memcpy(data, r.data, packedSize()); }

  inline Ray(Ray &&r) { std::memmove(data, r.data, packedSize()); }

  inline Ray(const unsigned char *buf) { std::memcpy(data, buf, packedSize()); }

  inline Ray &operator=(const Ray &r) {
    std::memcpy(data, r.data, packedSize());
    return *this;
  }

  inline Ray &operator=(Ray &&r) {
    std::memmove(data, r.data, packedSize());
    return *this;
  }
  ~Ray(){};

  /// returns size in bytes for the ray information to be sent via MPI
  size_t packedSize() const { return sizeof(Ray); }

  /// packs the ray information onto the given buffer and returns the number of bytes packed
  int pack(unsigned char *buffer) {
    unsigned char *buf = buffer;
    std::memcpy(buf, data, packedSize());
    return packedSize();
  }

  friend std::ostream &operator<<(std::ostream &stream, Ray const &ray) {
    stream << std::setprecision(4) << std::fixed << std::scientific;
    stream << "Ray[" << ray.id << "][" << ((ray.type == PRIMARY) ? "P" : (ray.type == SHADOW) ? "SH" : "S");
    stream << "]" << ray.origin << " " << ray.direction << " " << ray.color;
    stream << " t[" << ray.t_min << ", " << ray.t << ", " << ray.t_max << "]";
    return stream;
  }

  union {
    struct {
      glm::vec3 origin;
      float t_min;
      glm::vec3 direction;
      float t_max;
      glm::vec3 color;
      float t;
      int id;    ///<! index into framebuffer
      int depth; ///<! sample rate
      float w;   ///<! weight of image contribution
      int type;
    };
    unsigned char data[64] GVT_ALIGN(16);
  };

protected:
};

// NOTE: removing boost pool allocator greatly improves timings
typedef std::vector<Ray> RayVector;
}
}
}

#endif /* GVT_RENDER_ACTOR_RAY_H */
