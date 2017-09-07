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
#ifndef GVT_RAY_PACKET_H
#define GVT_RAY_PACKET_H

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/primitives/BBox.h>

#include <climits>

namespace gvt {
namespace render {
namespace actor {

/**
 * Min of two T values
 * @method fastmin
 * @param  a
 * @param  b
 * @return
 */
template <typename T> inline T fastmin(const T &a, const T &b) { return (a < b) ? a : b; }
/**
 * Max of two T values
 * @method fastmax
 * @param  a
 * @param  b
 * @return
 */
template <typename T> inline T fastmax(const T &a, const T &b) { return (a > b) ? a : b; }

/**
 * \brief Ray packet intersion with a AABB box.
 *
 * @tparam simd_width Architecture vector unit width defined at compile time @see GVT_SIMD_WIDTH
 *
 */
template <size_t simd_width> struct RayPacketIntersection {
  float ox[simd_width]; /**< Origin x component for all rays in packet */
  float oy[simd_width]; /**< Origin y component for all rays in packet */
  float oz[simd_width]; /**< Origin z component for all rays in packet */
  float dx[simd_width]; /**< Direction x component for all rays in packet */
  float dy[simd_width]; /**< Direction y component for all rays in packet */
  float dz[simd_width]; /**< Direction z component for all rays in packet */
  float t[simd_width];  /**< Intersection distance */
  int mask[simd_width]; /**< Ray packet mask 0 | disable ray, 1 | Ray enable */
  bool gotcha = false;
  int igotcha;

  /**
   * Creates ray packet of the first simd_width elements from list of rays starting at ray_begin.
   * @method RayPacketIntersection
   * @param  ray_begin             Ray start iterator
   * @param  ray_end               Ray list end iterator
   */
  inline RayPacketIntersection(const RayVector::iterator &ray_begin, const RayVector::iterator &ray_end) {
    size_t i;
    RayVector::iterator rayit = ray_begin;
    for (i = 0; rayit != ray_end && i < simd_width; ++i, ++rayit) {
      Ray &ray = (*rayit);
      if(ray.id == 130816) {
        gotcha = true;
        igotcha = i;
      }
      ox[i] = ray.origin[0];
      oy[i] = ray.origin[1];
      oz[i] = ray.origin[2];
      dx[i] = 1.f / ray.direction[0];
      dy[i] = 1.f / ray.direction[1];
      dz[i] = 1.f / ray.direction[2];
      t[i] = (ray.type != Ray::PRIMARY_VOLUME) ? ray.t_max : -ray.t;
      mask[i] = 1;
    }
    for (; i < simd_width; ++i) {
      t[i] = -1;
      mask[i] = -1;
    }
  }

  /**
   * Computed the intersection of all rays in the packet with a AABB.
   * @method intersect
   * @param  bb        AABB
   * @param  hit       Array returns if the corresponding index ray hits the AABB (1) or not (-1)
   * @param  update    Should update t or not.
   * @return           At least one ray hits the AABB
   */
  inline bool intersect(const gvt::render::data::primitives::Box3D &bb, int hit[], bool update = false) {
    float lx[simd_width];
    float ly[simd_width];
    float lz[simd_width];
    float ux[simd_width];
    float uy[simd_width];
    float uz[simd_width];

    float minx[simd_width];
    float miny[simd_width];
    float minz[simd_width];

    float maxx[simd_width];
    float maxy[simd_width];
    float maxz[simd_width];

    float tnear[simd_width];
    float tfar[simd_width];

    const float blx = bb.bounds_min[0], bly = bb.bounds_min[1], blz = bb.bounds_min[2];
    const float bux = bb.bounds_max[0], buy = bb.bounds_max[1], buz = bb.bounds_max[2];
#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) lx[i] = (blx - ox[i]) * dx[i];
#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) ly[i] = (bly - oy[i]) * dy[i];
#ifndef __clang__
#pragma simd
#endif

    for (size_t i = 0; i < simd_width; ++i) lz[i] = (blz - oz[i]) * dz[i];
#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) ux[i] = (bux - ox[i]) * dx[i];
#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) uy[i] = (buy - oy[i]) * dy[i];
#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) uz[i] = (buz - oz[i]) * dz[i];

#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) {
      minx[i] = fastmin(lx[i], ux[i]);
      maxx[i] = fastmax(lx[i], ux[i]);
    }

#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) {
      miny[i] = fastmin(ly[i], uy[i]);
      maxy[i] = fastmax(ly[i], uy[i]);
    }

#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) {
      minz[i] = fastmin(lz[i], uz[i]);
      maxz[i] = fastmax(lz[i], uz[i]);
    }

#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) {
      tnear[i] = fastmax(fastmax(minx[i], miny[i]), minz[i]);
      tfar[i] = fastmin(fastmin(maxx[i], maxy[i]), maxz[i]);
    }

#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) {
        if(t[i] > 0.) hit[i] = (tfar[i] > tnear[i] && (!update || tnear[i] > gvt::render::actor::Ray::RAY_EPSILON) && t[i] > tnear[i])
                   ? 1
                   : -1;
        else 
      hit[i] = (tfar[i] > tnear[i] && (!update || tnear[i] > gvt::render::actor::Ray::RAY_EPSILON) && -t[i] < tnear[i])
                   ? 1
                   : -1;
    }
    if(gotcha  ) { std::cout << " RayPacketIntersect: " ;
        std::cout << hit[igotcha] << " " << tnear[igotcha] << " " << tfar[igotcha];
        std::cout << " " << t[igotcha] << " : " << minx[igotcha] << " ";
        std::cout << miny[igotcha] << " " << minz[igotcha] << " : ";
        std::cout << maxx[igotcha]<< " " << maxy[igotcha] << " " << maxz[igotcha] << std::endl;
    }
#ifndef __clang__
#pragma simd
#endif
    for (size_t i = 0; i < simd_width; ++i) {
      if (hit[i] == 1 && update) t[i] = tnear[i];
    }

    for (size_t i = 0; i < simd_width; ++i)
      if (hit[i] == 1) {
          gotcha = false;
          return true;
      }

    gotcha = false;
    return false;
  }
};
}
}
}

#endif
