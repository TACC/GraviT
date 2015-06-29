//
// BVH.h
//

#ifndef GVT_RENDER_DATA_ACCEL_BVH_H
#define GVT_RENDER_DATA_ACCEL_BVH_H

#include <gvt/render/data/accel/AbstractAccel.h>
#include <gvt/render/data/primitives/BBox.h>
#include <gvt/core/Math.h>

namespace gvt {
    namespace render {
        namespace data {
            namespace accel {
                class BVH : public AbstractAccel
                {
                public:
                    BVH(std::vector<gvt::render::data::domain::AbstractDomain*>& domainSet);
                    ~BVH();

                    virtual void intersect(const gvt::render::actor::Ray& ray, gvt::render::actor::isecDomList& isect);
                private:
                    struct Node
                    {
                    	Node() : leftChild(NULL), rightChild(NULL), numDomains(0) {}
                        Node* leftChild; // null for leaf nodes
                        Node* rightChild; // null for leaf nodes
                        gvt::render::data::primitives::Box3D bbox;
                        int domainSetIdx; // base, valid when numDomains>0
                        int numDomains; // 0 means an internal node
                    };

                    struct CentroidLessThan
                    {
                        CentroidLessThan(float splitPoint, int splitAxis)
                        : splitPoint(splitPoint), splitAxis(splitAxis)
                        {
                        }
                        bool operator()(const gvt::render::data::domain::AbstractDomain* domain) const {
                            gvt::core::math::Point4f centroid = domain->worldCentroid();
                            return (centroid[splitAxis] < splitPoint);
                        }
                        float splitPoint;
                        int splitAxis;
                    };

                private:
                    Node* build(std::vector<gvt::render::data::domain::AbstractDomain*>& sortedDomainSet,
                                int start, int end, int level);

                    float findSplitPoint(int splitAxis, int start, int end);

                    void trace(const gvt::render::actor::Ray& ray, const Node* node, ClosestHit& hit, gvt::render::actor::isecDomList& isect, int level);

                private:
                	std::vector<Node*> nodes;
                	Node* root;
                };
            }
        }
    }
}

#endif // GVT_RENDER_DATA_ACCEL_BVH_H
