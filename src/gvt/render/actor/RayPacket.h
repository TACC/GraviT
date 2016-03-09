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

template <typename T> inline T fastmin(const T &a, const T &b) { return (a < b) ? a : b; }

template <typename T> inline T fastmax(const T &a, const T &b) { return (a > b) ? a : b; }

template <size_t simd_width> struct RayPacketIntersection {
  union {
    struct {
      float ox[simd_width];
      float oy[simd_width];
      float oz[simd_width];
      float dx[simd_width];
      float dy[simd_width];
      float dz[simd_width];
      float t[simd_width];
      int mask[simd_width];
    };
    unsigned char packet[];
  };

  inline RayPacketIntersection(const RayVector::iterator &ray_begin, const RayVector::iterator &ray_end) {
    size_t i;
    RayVector::iterator rayit = ray_begin;
    for (i = 0; rayit != ray_end && i < simd_width; ++i, ++rayit) {
      Ray &ray = (*rayit);
      ox[i] = ray.origin[0];
      oy[i] = ray.origin[1];
      oz[i] = ray.origin[2];
      dx[i] = 1.f / ray.direction[0];
      dy[i] = 1.f / ray.direction[1];
      dz[i] = 1.f / ray.direction[2];
      t[i] = FLT_MAX;
      mask[i] = 1;
    }
    for (; i < simd_width; ++i) {
      t[i] = -1.f;
      mask[i] = -1;
    }
  }

  inline RayPacketIntersection(const RayPacketIntersection &other) {
    std::memcpy(packet, other.packet, sizeof(RayPacketIntersection));
  }

  inline float *min(const float a[simd_width], const float b[simd_width]) {
    float m[simd_width];
    for (int i = 0; i < simd_width; i++) m[i] = fastmin(a[i], b[i]);
    return std::move(m);
  }

  inline float *max(const float a[simd_width], const float b[simd_width]) {
    float m[simd_width];
#pragma unroll
    for (int i = 0; i < simd_width; i++) m[i] = fastmax(a[i], b[i]);
    return std::move(m);
  }

  inline RayPacketIntersection(RayPacketIntersection &&other) { std::swap(packet, other.packet); }

  inline bool intersect(const gvt::render::data::primitives::Box3D &bb, int hit[]) {
    // float lx[simd_width];
    // float ly[simd_width];
    // float lz[simd_width];
    // float ux[simd_width];
    // float uy[simd_width];
    // float uz[simd_width];

    float data[simd_width * 8];
    float *lx = &data[simd_width * 0];
    float *ly = &data[simd_width * 1];
    float *lz = &data[simd_width * 2];
    float *ux = &data[simd_width * 3];
    float *uy = &data[simd_width * 4];
    float *uz = &data[simd_width * 5];
    float *tnear = &data[simd_width * 6];
    float *tfar = &data[simd_width * 7];

#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) hit[i] = -1;
#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) lx[i] = (bb.bounds_min[0] - ox[i]) * dx[i];
#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) ly[i] = (bb.bounds_min[1] - oy[i]) * dy[i];
#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) lz[i] = (bb.bounds_min[2] - oz[i]) * dz[i];
#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) ux[i] = (bb.bounds_max[0] - ox[i]) * dx[i];
#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) uy[i] = (bb.bounds_max[1] - oy[i]) * dy[i];
#pragma unroll
    for (size_t i = 0; i < simd_width; ++i) uz[i] = (bb.bounds_max[2] - oz[i]) * dz[i];

    tnear = max(min(lx, ux), max(min(ly, uy), min(lz, uz)));
    tfar = min(max(lx, ux), min(max(ly, uy), max(lz, uz)));
#pragma unroll

    for (size_t i = 0; i < simd_width; ++i) {
      if (tfar[i] > tnear[i] && tnear[i] > FLT_EPSILON) {
        t[i] = lx[i];
        hit[i] = 1;
      }
    }

    for (size_t i = 0; i < simd_width; ++i)
      if (hit[i]) return true;

    return false;
  }
};
}
}
}

#endif
