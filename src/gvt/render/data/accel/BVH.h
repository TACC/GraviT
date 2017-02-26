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
//
// BVH.h
//

#ifndef GVT_RENDER_DATA_ACCEL_BVH_H
#define GVT_RENDER_DATA_ACCEL_BVH_H

#include <stack>
#include <mutex>

#include <gvt/core/Math.h>
#include <gvt/render/actor/RayPacket.h>
#include <gvt/render/data/accel/AbstractAccel.h>
#include <gvt/render/data/primitives/BBox.h>

namespace gvt {
namespace render {
namespace data {
namespace accel {
/// bounding volume hierarchy (BVH) acceleration structure
/** bounding volume hierarchy (BVH) acceleration structure
used to organize high-level data domains within GraviT. GraviT
intersects rays against the BVH to determine traversal order through
the data domains and the work scheduler uses this information as
part of its evaluation process.
*/
class BVH : public AbstractAccel {
public:
  BVH(gvt::core::Vector<gvt::core::DBNodeH> &instanceSet);
  ~BVH();

  struct hit {
    int next = -1;
    float t = FLT_MAX;
  };

  template <size_t simd_width>
  gvt::core::Vector<hit> intersect(const gvt::render::actor::RayVector::iterator &ray_begin,
                                   const gvt::render::actor::RayVector::iterator &ray_end,
                                   const int from) {

    gvt::core::Vector<hit> ret((ray_end - ray_begin));
    size_t offset = 0;
#ifndef GVT_BRUTEFORCE
    Node *stack[instanceSet.size() * 2];
    Node **stackptr = stack;
#endif

    gvt::render::actor::RayVector::iterator chead = ray_begin;
    for (; offset < ret.size(); offset += simd_width, chead += simd_width) {
      gvt::render::actor::RayPacketIntersection<simd_width> rp(chead, ray_end);

#ifdef GVT_BRUTEFORCE
      for (int i = 0; i < instanceSet.size(); i++) {
        if (from == instanceSetID[i]) continue;
        int hit[simd_width];
        const primitives::Box3D &ibbox = *instanceSetBB[i];
        rp.intersect(ibbox, hit, true);
        {
          for (int o = 0; o < simd_width; ++o) {
            if (hit[o] == 1 && rp.mask[o] == 1) {
              ret[offset + o].next = instanceSetID[i];
              ret[offset + o].t = rp.t[o];
            }
          }
        }
      }
#else

      *(stackptr++) = nullptr;
      Node *cur = root;
      int hit[simd_width];
      while (cur) {

        const gvt::render::data::primitives::Box3D &bb = cur->bbox;
        if (!rp.intersect(bb, hit)) {
          cur = *(--stackptr);
          continue;
        }

        if (cur->numInstances > 0) { // leaf node
          int start = cur->instanceSetIdx;
          int end = start + cur->numInstances;
          for (int i = start; i < end; ++i) {
            if (from == instanceSetID[i]) continue;
            const primitives::Box3D &ibbox = *instanceSetBB[i];
            int hit[simd_width];
            if (rp.intersect(ibbox, hit, true)) {
              for (int o = 0; o < simd_width; ++o) {
                if (hit[o] == 1 && rp.mask[o] == 1 && ret[offset + o].t > rp.t[o]) {
                  ret[offset + o].next = instanceSetID[i];
                  ret[offset + o].t = rp.t[o];
                }
              }
            }
          }

          cur = *(--stackptr);

        } else {
          *(stackptr++) = cur->rightChild;
          cur = cur->leftChild;
        }
      }
#endif
    }
    return ret;
  }

private:
  struct Node {
    Node() : leftChild(NULL), rightChild(NULL), numInstances(0) {}
    Node *leftChild;  /// null for leaf nodes
    Node *rightChild; /// null for leaf nodes
    gvt::render::data::primitives::Box3D bbox;
    int instanceSetIdx; /// base, valid when numInstances>0
    int numInstances;   /// 0 means an internal node
  };

  struct CentroidLessThan {
    CentroidLessThan(float splitPoint, int splitAxis) : splitPoint(splitPoint), splitAxis(splitAxis) {}
    bool operator()(const gvt::core::DBNodeH inst) const {
      gvt::core::DBNodeH i2 = inst;
      glm::vec3 centroid = i2["centroid"].value().tovec3();
      return (centroid[splitAxis] < splitPoint);
    }

    float splitPoint;
    int splitAxis;
  };

private:
  Node *build(gvt::core::Vector<gvt::core::DBNodeH> &sortedDomainSet, int start, int end, int level);

  float findSplitPoint(int splitAxis, int start, int end);

  gvt::core::Vector<gvt::render::data::primitives::Box3D *> instanceSetBB;
  gvt::core::Vector<int> instanceSetID;

private:
  gvt::core::Vector<Node *> nodes;
  Node *root;
  static std::mutex c_out;
};
}
}
}
}

#endif // GVT_RENDER_DATA_ACCEL_BVH_H
