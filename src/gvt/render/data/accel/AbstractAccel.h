//
// AbstractAccel.h
//

#ifndef GVT_RENDER_DATA_ACCEL_ABSTRACT_ACCEL_H
#define GVT_RENDER_DATA_ACCEL_ABSTRACT_ACCEL_H

#include <gvt/render/data/primitives/BBox.h>
#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/actor/Ray.h>

#include <vector>
#include <limits>

namespace gvt {
    namespace render {
        namespace data {
            namespace accel {

                struct ClosestHit
                {
                    ClosestHit() : domain(NULL), distance(std::numeric_limits<float>::max()) {}
                    gvt::render::data::domain::AbstractDomain* domain;
                    float distance;
                };

                class AbstractAccel
                {
            	public:
            	    AbstractAccel(std::vector<gvt::render::data::domain::AbstractDomain*>& domainSet)
                    : domainSet(domainSet) {}
                    virtual void intersect(const gvt::render::actor::Ray& ray, gvt::render::actor::isecDomList& isect) = 0;
            	protected:
                    std::vector<gvt::render::data::domain::AbstractDomain*> domainSet;
                };
            }
        }
    }
}

#endif // GVT_RENDER_DATA_ACCEL_ABSTRACT_ACCEL_H