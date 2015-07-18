//
// BVH.cpp
//

#include <gvt/render/data/accel/BVH.h>

#include <limits>
#include <algorithm>
#include <cassert>
#include <iostream>

#include <boost/range/algorithm.hpp>

using namespace gvt::render::data::accel;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::domain;
using namespace gvt::core::math;

#define TRAVERSAL_COST 0.5 // TODO: best value?
#define LEAF_SIZE 1 // TODO: best value?

// #define DEBUG_ACCEL

BVH::BVH(std::vector<gvt::render::data::domain::AbstractDomain*>& domainSet)
: AbstractAccel(domainSet), root(NULL)
{
    std::vector<gvt::render::data::domain::AbstractDomain*> sortedDomainSet;
    root = build(sortedDomainSet, 0, domainSet.size(), 0);

#ifdef DEBUG_ACCEL
    assert(this->domainSet.size() == sortedDomainSet.size());
#endif

    this->domainSet.swap(sortedDomainSet);
}

BVH::~BVH()
{
    //TODO: better way to manage memory allocation?
    for (int i=0; i<nodes.size(); ++i) {
        delete nodes[i];
        nodes[i] = NULL;   
    }
}

void BVH::intersect(const gvt::render::actor::Ray& ray, gvt::render::actor::isecDomList& isect)
{
    if (root) {
        ClosestHit hit;
        trace(ray, root, hit, isect, 0);
    }
}

BVH::Node* BVH::build(std::vector<gvt::render::data::domain::AbstractDomain*>& sortedDomainSet,
                      int start, int end, int level)
{
    Node* node = new Node();

    // TODO: better way to manange memory allocation?
    nodes.push_back(node);

    // evaluate bounds
    Box3D bbox;
    for (int i=start; i<end; ++i) {
        bbox.merge(domainSet[i]->getWorldBoundingBox());
    }

    int domainCount = end - start;

    // base case
	if (domainCount <= LEAF_SIZE) {
#ifdef DEBUG_ACCEL
        std::cout<<"creating leaf node.."<<"[LVL:"<<level<<"][offset: "<<sortedDomainSet.size()<<"][#domains:"<<domainCount<<"]\n";
#endif
		// create leaf node
        node->bbox = bbox;
		node->domainSetIdx = sortedDomainSet.size();
		node->numDomains = domainCount;
        for (int i=start; i<end; ++i) {
            sortedDomainSet.push_back(domainSet[i]);
        }
		return node;
	}

    // choose partition axis based on largest variation of centroids
    int splitAxis = bbox.wideRangingBoxDir();

    // choose split point based on SAH
    float splitPoint = findSplitPoint(splitAxis, start, end);

#ifdef DEBUG_ACCEL
    for (int i=start; i<end; ++i) {
        gvt::core::math::Point4f centroid = domainSet[i]->worldCentroid();
        bool lessThan = (centroid[splitAxis] < splitPoint);
        std::cout<<"[Lvl"<<level<<"][SP:"<<splitPoint<<"]["<<i<<"][id:"<<domainSet[i]->getDomainID()<<"][centroid: "<<centroid[splitAxis]<<"][isLess: "<<lessThan<<"]\t";
    }
    std::cout<<"\n";
#endif

    // partition domains into two subsets
    AbstractDomain** domainBound = std::partition(&domainSet[start], &domainSet[end-1]+1,
    	                                         CentroidLessThan(splitPoint, splitAxis));
    int splitIdx = domainBound - &domainSet[0];

    if (splitIdx == start || splitIdx == end)
    {
#ifdef DEBUG_ACCEL
        std::cout<<"creating leaf node.."<<"[LVL:"<<level<<"][offset: "<<sortedDomainSet.size()<<"][#domains:"<<domainCount<<"]\n";
#endif
        // create leaf node
        node->bbox = bbox;
        node->domainSetIdx = sortedDomainSet.size();
        node->numDomains = domainCount;
        for (int i=start; i<end; ++i) {
            sortedDomainSet.push_back(domainSet[i]);
        }
        return node;
    }

// #ifdef DEBUG_ACCEL
//     std::cout<<"start:"<<start<<"\tend:"<<end<<"\tsplitIdx: "<<splitIdx<<"\t level: "<<level<<"\n";
//     if (level == 1) return node;
// #endif

    // recursively build internal nodes
    int nextLevel = level + 1;
    Node* nodeL = build(sortedDomainSet, start, splitIdx, nextLevel);
    Node* nodeR = build(sortedDomainSet, splitIdx, end, nextLevel);

    node->leftChild = nodeL;
    node->rightChild = nodeR;
    node->bbox = bbox;
    node->numDomains = 0;

    return node;
}

float BVH::findSplitPoint(int splitAxis, int start, int end)
{
    // choose split point based on SAH
    // SAH cost = c_t + (p_l * c_l) + (p_r * c_r)
    // for now, do exhaustive searches on both edges of all bounding boxes
    float minCost = std::numeric_limits<float>::max();
    float splitPoint;

    for (int i=start; i<end; ++i) {

    	Box3D refBbox = domainSet[i]->getWorldBoundingBox();

    	for (int e=0; e<2; ++e) {

            float edge = refBbox.bounds[e][splitAxis];

            Box3D leftBox, rightBox;
            int leftCount = 0;

            for (int j=start; j<end; ++j) {
                Box3D bbox = domainSet[j]->getWorldBoundingBox();
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
    return 0.0f;
}

void BVH::trace(const gvt::render::actor::Ray& ray, const Node* node, ClosestHit& hit, gvt::render::actor::isecDomList& isect, int level)
{

	float t = std::numeric_limits<float>::max();

    if (!(node->bbox.intersectDistance(ray, t) && (t > gvt::render::actor::Ray::RAY_EPSILON))) {
      return;
    }

    if (t > hit.distance) {
        return;
    }

    int domainCount = node->numDomains;

    if (domainCount > 0) { // leaf node
#ifdef DEBUG_ACCEL
        assert(!node->leftChild && !node->rightChild);
#endif
  	    int start = node->domainSetIdx;
  	    int end = start + domainCount;

        for (int i=start; i<end; ++i) {
            if (domainSet[i]->getWorldBoundingBox().intersectDistance(ray, t) && (t > gvt::render::actor::Ray::RAY_EPSILON)) {
                isect.push_back(gvt::render::actor::isecDom(domainSet[i]->getDomainID(), t));
            }
        }
    } else {
#ifdef DEBUG_ACCEL
    	assert(node->leftChild && node->rightChild);
#endif
        int nextLevel = level + 1;
        trace(ray, node->leftChild, hit, isect, nextLevel);
        trace(ray, node->rightChild, hit, isect, nextLevel);
    }
}