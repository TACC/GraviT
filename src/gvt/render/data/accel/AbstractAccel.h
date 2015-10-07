//
// AbstractAccel.h
//

#ifndef GVT_RENDER_DATA_ACCEL_ABSTRACT_ACCEL_H
#define GVT_RENDER_DATA_ACCEL_ABSTRACT_ACCEL_H

#include <gvt/core/CoreContext.h>

#include <gvt/render/data/primitives/BBox.h>
#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/actor/Ray.h>


#include <vector>
#include <limits>

namespace gvt {
    namespace render {
        namespace data {
            namespace accel {
                /// struct for closest intersection between ray and acceleration structure
                struct ClosestHit
                {
                    ClosestHit() : domain(NULL), distance(std::numeric_limits<float>::max()) {}
                    gvt::render::data::domain::AbstractDomain* domain;
                    gvt::core::DBNodeH instance;
                    float distance;
                };

                /// abstract base class for acceleration structures
                class AbstractAccel
                {
            	public:
                    AbstractAccel(gvt::core::Vector<gvt::core::DBNodeH>& instanceSet)
                        : instanceSet(instanceSet) {}
                    virtual void intersect(const gvt::render::actor::Ray& ray, gvt::render::actor::isecDomList& isect) = 0;
            	protected:
                    gvt::core::Vector<gvt::core::DBNodeH> instanceSet;
                };
            }
        }
    }
}

#endif // GVT_RENDER_DATA_ACCEL_ABSTRACT_ACCEL_H
