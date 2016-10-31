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
#ifndef _MANTA_DYNBVH_H_
#define _MANTA_DYNBVH_H_

#include <Core/Geometry/BBox.h>
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/AlignedAllocator.h>
#include <Core/Util/SpinLock.h>
#include <Interface/AccelerationStructure.h>
#include <Interface/RayPacket.h>
#include <Model/Groups/Group.h>
#include <Model/Groups/Mesh.h>
#include <Model/Groups/TreeTraversalProbability.h>
#include <stdio.h>
namespace Manta {
class Task;
class TaskList;

class MANTA_ALIGN(MAXCACHELINESIZE) DynBVH : public AccelerationStructure,
                                             public AlignedAllocator<DynBVH, MAXCACHELINESIZE> {
public:
  struct IAData {
    Real min_rcp[3];
    Real max_rcp[3];
    Real min_org[3];
    Real max_org[3];
    Real min_org_rcp[3];
    Real max_org_rcp[3];
  };

  struct BVHNode {
    BBox bounds;
    int child;          // my child
    unsigned char axis; // 0 = x, 1 = y, 2 = z.  axis==3 means node is uninitialized.
    unsigned isLeftCheaper : 1;
    unsigned isLargeSubtree : 1; // If it has lots of subnodes, it's
                                 // handled by serial part of BVH update
    short children;              // num children

    // 24 bytes + 4 bytes + 1 + 1 + 2 = 32 bytes

    inline void makeLeaf(int first_child, short num_children) {
      child = first_child;
      children = num_children;
      axis = 0;
    }

    inline void makeInternal(int first_child, unsigned char _axis) {
      child = first_child;
      children = 0;
      axis = _axis;
    }

    inline bool isLeaf() const { return children != 0; }

    inline bool isUninitialized() const { return axis == 3; }

    inline void setUninitialized() { axis = 3; }

    void readwrite(ArchiveElement *archive);
  };

  struct BVHBuildRecord {
    int objectBegin;
    int objectEnd;
  };

public:
  // NOTE(boulos): Because intersect is const, lazy build requires
  // that almost everything in here become mutable and that almost
  // all functions become const. How sad.
  mutable vector<BVHNode> nodes;
  mutable vector<int> object_ids;
  mutable vector<BVHBuildRecord> build_records;

  vector<unsigned int> subtreeList; // for load balancing parallel update
  unsigned int largeSubtreeSize;
  Mutex subtreeListMutex;
  bool subtreeListFilled;

  mutable AtomicCounter num_nodes;
  Group *currGroup;
  Mesh *mesh; // NULL if there is no mesh.
  bool group_changed;
  Barrier barrier;
  SpinLock taskpool_mutex;

  static const unsigned int kNumLazyBuildMutexes = 256;
  mutable AtomicCounter nextFree;

  // clang-format off
  // Align each vector (align the vector itself, not the data in it).
  template <class T, size_t Alignment> struct MANTA_ALIGN(MAXCACHELINESIZE) SpacedVectors {
    vector<T> v;
  };
  // clang-format on
  mutable MANTA_ALIGN(MAXCACHELINESIZE) SpacedVectors<int, MAXCACHELINESIZE> nodesBeingBuilt[kNumLazyBuildMutexes];

  mutable Mutex lazybuild_mutex[kNumLazyBuildMutexes];
  mutable ConditionVariable lazybuild_cond[kNumLazyBuildMutexes];

  mutable vector<BBox> obj_bounds;
  mutable vector<Vector> obj_centroids;

  char *TaskListMemory;
  size_t CurTaskList;

  char *TaskMemory;
  size_t CurTask;

  char *TwoArgCallbackMemory;
  size_t CurTwoArgCallback;

  char *FourArgCallbackMemory;
  size_t CurFourArgCallback;

  char *FiveArgCallbackMemory;
  size_t CurFiveArgCallback;

  bool print_info;

  TreeTraversalProb ttp;

public:
  DynBVH(bool print = true)
      : subtreeListMutex("subtreeList"), subtreeListFilled(false), num_nodes("DynBVH Num Nodes", 0), currGroup(NULL),
        mesh(NULL), group_changed(false), barrier("DynBVH barrier"), nextFree("DynBVH Next Free", 0), TaskListMemory(0),
        CurTaskList(0), TaskMemory(0), CurTask(0), TwoArgCallbackMemory(0), CurTwoArgCallback(0),
        FourArgCallbackMemory(0), CurFourArgCallback(0), FiveArgCallbackMemory(0), CurFiveArgCallback(0),
        print_info(print) {}
  virtual ~DynBVH();

  void setGroup(Group *new_group);
  Group *getGroup() const;
  void groupDirty(); // tells the acceleration structure that the group has changed

  virtual void addToUpdateGraph(ObjectUpdateGraph *graph, ObjectUpdateGraphNode *parent);

  void allocate() const;

  void printNode(int nodeID, int depth) const;

  void beginParallelPreprocess(UpdateContext context);
  void parallelPreprocess(Task *task, int objectBegin, int objectEnd, UpdateContext context);
  void finishParallelPreprocess(TaskList *tasklist, UpdateContext context);

  void beginParallelBuild(UpdateContext context);

  void parallelTopDownBuild(Task *task, int node_id, int objectBegin, int objectEnd, UpdateContext context);
  void parallelBuildReduction(TaskList *list, int node_id, Task *task);

  void parallelApproximateSAH(Task *task, int nodeID, int objectBegin, int objectEnd, UpdateContext context);
  void parallelComputeBounds(Task *task, int nodeID, int objectBegin, int objectEnd, UpdateContext context);
  void parallelComputeBoundsReduction(TaskList *tasklist, Task *task, int nodeID, int objectBegin, int objectEnd,
                                      UpdateContext context);
  void parallelComputeBins(Task *task, int nodeID, int objectBegin, int objectEnd, UpdateContext context);
  void parallelComputeBinsReduction(TaskList *tasklist, Task *task, int nodeID, int objectBegin, int objectEnd,
                                    UpdateContext context);

  int partitionObjects(int first, int last, int axis, float position) const;
  void splitBuild(Task *task, int nodeID, int objectBegin, int objectEnd, UpdateContext context);

  void preprocess(const PreprocessContext &);
  void intersect(const RenderContext &context, RayPacket &rays) const;

  Interpolable::InterpErr parallelInterpolate(const std::vector<keyframe_t> &keyframes, int proc, int numProc) {
    return currGroup->parallelInterpolate(keyframes, proc, numProc);
  }

  // return the first index (between [rays.begin(),rays.end()]) which hits the box
  static int firstIntersects(const BBox &box, const RayPacket &rays, const IAData &ia_data);
  // return the last index which hits the box
  static int lastIntersects(const BBox &box, const RayPacket &rays);

protected:
  void intersectNode(int nodeID, const RenderContext &context, RayPacket &rays, const IAData &ia_data) const;

public:
  void computeBounds(const PreprocessContext &context, BBox &bbox) const { currGroup->computeBounds(context, bbox); }

  void update(int proc = 0, int numProcs = 1);
  void rebuild(int proc = 0, int numProcs = 1);

  void build(int nodeID, int objectBegin, int objectEnd, const bool useApproximateBuild = false,
             const bool nodeAlreadyExists = false);

  void lazyBuild(const RenderContext &context, const int nodeID) const;

  void parallelUpdateBounds(const PreprocessContext &context, int proc, int numProcs);

  template <bool bottomSubtreesDone> void updateBounds(int ID = 0);
  void seedSubtrees(const int nodeID);
  unsigned int computeSubTreeSizes(size_t nodeID);

  void readwrite(ArchiveElement *archive);

  virtual bool buildFromFile(const string &file);
  virtual bool saveToFile(const string &file);

protected:
  void computeTraversalCost();
  VectorT<float, 2> computeSubTreeTraversalCost(unsigned int nodeID);

  struct PartitionData {
    BBox bounds_l, bounds_r;
    int axis;
    int split;
  };

  inline PartitionData partition2Objs(int nodeID, int objBegin) const;

  PartitionData partitionSAH(int nodeID, int objBegin, int objEnd) const;
  PartitionData partitionApproxSAH(int nodeID, int objBegin, int objEnd) const;

  struct BVHCostEval {
    BBox bounds_r;
    float position;
    float cost;
    int num_left;
    int num_right;
    int event;
    int axis;
  };

  struct BVHSAHEvent {
    float position;
    int obj_id;
    float left_area;
    float right_area;
    int num_left;
    int num_right;
    float cost;
  };

  struct CompareBVHSAHEvent {
    bool operator()(const BVHSAHEvent &x, const BVHSAHEvent &y) {
      // add obj_id sorting in here automatically?
      return x.position < y.position;
    }
  };

  bool buildEvents(int parentID, int first, int last, int axis, BVHCostEval &eval) const;

#ifdef MANTA_SSE
  // DynBVH ports
  void templatedTraverse(const RenderContext &context, RayPacket &packet) const;

  int firstActivePort(RayPacket &packet, int firstActive, const BBox &box) const;
  int lastActivePort(RayPacket &packet, int firstActive, const BBox &box) const;

  int firstActiveSameSignFrustumPort(RayPacket &packet, const int firstActive, const BBox &bounds, const int signs[3],
                                     const __m128 sc_min_org[3], const __m128 sc_max_org[3], const __m128 sc_min_rcp[3],
                                     const __m128 sc_max_rcp[3], const __m128 &max_t) const;

  int lastThatIntersectsSameSignPort(RayPacket &packet, const int firstActive, const BBox &box,
                                     const int signs[3]) const;
#endif

// #define COLLECT_STATS
#ifdef COLLECT_STATS
  struct Stats {
    Stats() { reset(); }

    void reset() {
      nIntersects = 0;
      nTraversals = 0;
      nTotalRays = 0;
      nTotalRaysInPacket = 0;
      nTotalPackets = 0;
      nLeavesVisited = 0;
    }

    long nIntersects;
    long nTraversals;
    long nTotalRays;
    long nTotalRaysInPacket;
    long nTotalPackets;
    long nLeavesVisited;

    // hopefully big enough to keep any false sharing among different
    // processors from occuring.
    char emptySpace[128];
  };

  mutable vector<Stats> stats; // one per thread

  Stats accumulateStats() const {
    Stats finalStats;
    for (size_t i = 0; i < stats.size(); ++i) {
      finalStats.nIntersects += stats[i].nIntersects;
      finalStats.nTraversals += stats[i].nTraversals;
      finalStats.nTotalRays += stats[i].nTotalRays;
      finalStats.nTotalRaysInPacket += stats[i].nTotalRaysInPacket;
      finalStats.nTotalPackets += stats[i].nTotalPackets;
      finalStats.nLeavesVisited += stats[i].nLeavesVisited;
    }
    return finalStats;
  }

  void updateStats(int proc, int numProcs) {
    if (proc == 0) {
      Stats finalStats = accumulateStats();

      printf("intersections per ray:        %f\n", (float)finalStats.nIntersects / finalStats.nTotalRays);
      printf("node traversals per ray:      %f\n", (float)finalStats.nTraversals / finalStats.nTotalRays);
      printf("leaves visited per ray:       %f\n", (float)finalStats.nLeavesVisited / finalStats.nTotalRays);
      printf("average ray packet size:      %f\n", (float)finalStats.nTotalRaysInPacket / finalStats.nTotalPackets);
      printf("number of packets:            %ld\n", finalStats.nTotalPackets);
      printf("number of rays:               %ld\n", finalStats.nTotalRays);
      printf("\n");

      stats.resize(numProcs);
    }

    barrier.wait(numProcs);

    stats[proc].reset();
  }
#endif

#define TREE_ROT 1
#if TREE_ROT
protected:
  // TODO: instead of storing costs and subtree_size for all nodes (so
  // wasteful!), try to compute these on the fly during the rotation step.
  mutable vector<Real> costs;
  mutable vector<unsigned int> subtree_size; // number of nodes making up this subtree

  void rotateNode(int const nodeID);
  void rotateTree(int const nodeID);
#endif
  template <bool saveCost> Real computeCost(int const nodeID);
};
};

#endif
