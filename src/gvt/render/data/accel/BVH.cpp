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
// BVH.cpp
//

#include <gvt/render/data/accel/BVH.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

#include <boost/range/algorithm.hpp>

using namespace gvt::render::data::accel;
using namespace gvt::render::data::primitives;

#define TRAVERSAL_COST 0.5 // TODO: best value?
#define LEAF_SIZE 1        // TODO: best value?

// #define DEBUG_ACCEL

BVH::BVH(gvt::core::Vector<gvt::core::DBNodeH> &instanceSet) : AbstractAccel(instanceSet), root(NULL) {
  gvt::core::Vector<gvt::core::DBNodeH> sortedInstanceSet;
  root = build(sortedInstanceSet, 0, instanceSet.size(), 0);

#ifdef DEBUG_ACCEL
  assert(this->instanceSet.size() == sortedInstanceSet.size());
#endif

  // this->instanceSet.swap(sortedInstanceSet);
  std::swap(this->instanceSet, sortedInstanceSet);

  // std::vector<gvt::render::data::primitives::Box3D*> instanceSetBB;
  // std::vector<int> instanceSetID;

  for (auto &node : this->instanceSet) {
    instanceSetBB.push_back((Box3D *)node["bbox"].value().toULongLong());
    instanceSetID.push_back(node["id"].value().toInteger());
  }

  std::cout << "BVH has " << instanceSet.size() << " instances " << root->bbox << std::endl;
}

BVH::~BVH() {
  // TODO: better way to manage memory allocation?
  for (int i = 0; i < nodes.size(); ++i) {
    delete nodes[i];
    nodes[i] = NULL;
  }
}

void BVH::intersect(const gvt::render::actor::Ray &ray, gvt::render::actor::isecDomList &isect) {

  const glm::vec3 &origin = ray.origin;
  const glm::vec3 &inv = 1.f / ray.direction;

  if (root) {
    // ClosestHit hit;
    trace(origin, inv, root, /*hit,*/ isect, 0);
  }
}

BVH::Node *BVH::build(gvt::core::Vector<gvt::core::DBNodeH> &sortedInstanceSet, int start, int end, int level) {
  Node *node = new Node();

  // TODO: better way to manange memory allocation?
  nodes.push_back(node);

  // evaluate bounds
  Box3D bbox;
  for (int i = start; i < end; ++i) {
    Box3D *tmpbb = (Box3D *)instanceSet[i]["bbox"].value().toULongLong();
    bbox.merge(*tmpbb);
  }

  int instanceCount = end - start;

  // base case
  if (instanceCount <= LEAF_SIZE) {
#ifdef DEBUG_ACCEL
    std::cout << "creating leaf node.."
              << "[LVL:" << level << "][offset: " << sortedInstanceSet.size() << "][#domains:" << instanceCount
              << "]\n";
#endif
    // create leaf node
    node->bbox = bbox;
    node->instanceSetIdx = sortedInstanceSet.size();
    node->numInstances = instanceCount;
    for (int i = start; i < end; ++i) {
      sortedInstanceSet.push_back(instanceSet[i]);
    }
    return node;
  }

  // choose partition axis based on largest variation of centroids
  int splitAxis = bbox.wideRangingBoxDir();

  // choose split point based on SAH
  float splitPoint = findSplitPoint(splitAxis, start, end);

#ifdef DEBUG_ACCEL
#ifdef DEBUG_ACCEL_DOMAIN_SET
  for (int i = start; i < end; ++i) {
    glm::vec3 centroid = instanceSet[i]->worldCentroid();
    bool lessThan = (centroid[splitAxis] < splitPoint);
    std::cout << "[Lvl" << level << "][SP:" << splitPoint << "][" << i << "][id:" << instanceSet[i]->getDomainID()
              << "][centroid: " << centroid[splitAxis] << "][isLess: " << lessThan << "]\t";
  }
  std::cout << "\n";
#else
  for (int i = start; i < end; ++i) {
    glm::vec3 centroid = instanceSet[i]["centroid"].value().tovec3();
    bool lessThan = (centroid[splitAxis] < splitPoint);
    int id = instanceSet[i]["id"].value().toInteger();
    std::cout << "[Lvl" << level << "][SP:" << splitPoint << "][" << i << "][id:" << id
              << "][centroid: " << centroid[splitAxis] << "][isLess: " << lessThan << "]\t";
  }
  std::cout << "\n";
#endif
#endif

  // partition domains into two subsets
  gvt::core::DBNodeH *instanceBound =
      std::partition(&instanceSet[start], &instanceSet[end - 1] + 1, CentroidLessThan(splitPoint, splitAxis));
  int splitIdx = instanceBound - &instanceSet[0];

  if (splitIdx == start || splitIdx == end) {
#ifdef DEBUG_ACCEL
    std::cout << "creating leaf node.."
              << "[LVL:" << level << "][offset: " << sortedInstanceSet.size() << "][#domains:" << instanceCount
              << "]\n";
#endif
    // create leaf node
    node->bbox = bbox;
    node->instanceSetIdx = sortedInstanceSet.size();
    node->numInstances = instanceCount;
    for (int i = start; i < end; ++i) {
      sortedInstanceSet.push_back(instanceSet[i]);
    }
    return node;
  }

  // recursively build internal nodes
  int nextLevel = level + 1;
  Node *nodeL = build(sortedInstanceSet, start, splitIdx, nextLevel);
  Node *nodeR = build(sortedInstanceSet, splitIdx, end, nextLevel);

  node->leftChild = nodeL;
  node->rightChild = nodeR;
  node->bbox = bbox;
  node->numInstances = 0;

  return node;
}

float BVH::findSplitPoint(int splitAxis, int start, int end) {
  // choose split point based on SAH
  // SAH cost = c_t + (p_l * c_l) + (p_r * c_r)
  // for now, do exhaustive searches on both edges of all bounding boxes
  float minCost = std::numeric_limits<float>::max();
  float splitPoint;

  for (int i = start; i < end; ++i) {

    Box3D &refBbox = *(Box3D *)instanceSet[i]["bbox"].value().toULongLong();

    for (int e = 0; e < 2; ++e) {

      float edge;
      if (e == 0) edge = refBbox.bounds_min[splitAxis];
      if (e == 1) edge = refBbox.bounds_max[splitAxis];

      Box3D leftBox, rightBox;
      int leftCount = 0;

      for (int j = start; j < end; ++j) {
        Box3D &bbox = *(Box3D *)instanceSet[j]["bbox"].value().toULongLong();
        if (bbox.centroid()[splitAxis] < edge) {
          ++leftCount;
          leftBox.merge(bbox);
        } else {
          rightBox.merge(bbox);
        }
      }
      // compute SAH
      int rightCount = end - start - leftCount;
      float cost = TRAVERSAL_COST + (leftBox.surfaceArea() * leftCount) + (rightBox.surfaceArea() * rightCount);

      if (cost < minCost) {
        minCost = cost;
        splitPoint = edge;
      }
    }
  }
  return splitPoint;
}

void BVH::trace(const glm::vec3 &origin, const glm::vec3 &inv, const Node *node, /*ClosestHit &hit,*/
                gvt::render::actor::isecDomList &isect, int level) {

  float t = std::numeric_limits<float>::max();

  if (!(node->bbox.intersectDistance(origin, inv, t) && (t > gvt::render::actor::Ray::RAY_EPSILON))) {
    return;
  }

  // if (t > hit.distance) {
  //   return;
  // }

  int instanceCount = node->numInstances;

  if (instanceCount > 0) { // leaf node
#ifdef DEBUG_ACCEL
    assert(!node->leftChild && !node->rightChild);
#endif
    int start = node->instanceSetIdx;
    int end = start + instanceCount;

    for (int i = start; i < end; ++i) {
      Box3D *ibbox = instanceSetBB[i];
      if (ibbox->intersectDistance(origin, inv, t) && (t > gvt::render::actor::Ray::RAY_EPSILON)) {
        int id = instanceSetID[i]; // gvt::core::variant_toInteger(instanceSet[i]["id"].value());
        isect.push_back(gvt::render::actor::isecDom(id, t));
      }
    }
  } else {
#ifdef DEBUG_ACCEL
    assert(node->leftChild && node->rightChild);
#endif
    int nextLevel = level + 1;
    trace(origin, inv, node->leftChild, /*hit,*/ isect, nextLevel);
    trace(origin, inv, node->rightChild, /*hit,*/ isect, nextLevel);
  }
}

// int BVH::intersect(const gvt::render::actor::Ray &ray, int from, float &t) {
//   if (root) {
//     int rid = trace(ray, root, from, t);
//     return rid;
//   }
//   return -1;
// }
//
// int BVH::trace(const gvt::render::actor::Ray &ray, const Node *node, int cid, float &t) {
//
//   float tlocal = std::numeric_limits<float>::max();
//
//   if (!(node->bbox.intersectDistance(ray, tlocal) && (tlocal > gvt::render::actor::Ray::RAY_EPSILON) && (tlocal <
//   t))) {
//     return -1;
//   }
//
//   if (node->numInstances > 0) { // leaf node
//     int start = node->instanceSetIdx;
//     int end = start + node->numInstances;
//     float best = t;
//     int rid = -1;
//     for (int i = start; i < end; ++i) {
//       if (cid == instanceSetID[i]) continue;
//       Box3D *ibbox = instanceSetBB[i];
//       float tlocal;
//       if (ibbox->intersectDistance(ray, tlocal) && (tlocal < best)) {
//         best = tlocal;
//         rid = instanceSetID[i];
//       }
//     }
//     if (rid != -1) t = best;
//     return rid;
//   } else {
//     int rid = trace(ray, node->leftChild, cid, t);
//     int lid = trace(ray, node->rightChild, cid, t);
//     if (lid != -1) return lid;
//     return rid;
//   }
// }
