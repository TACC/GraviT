/*
 * File:   gvt_optix.h
 * Author: jbarbosa
 *
 * Created on April 22, 2014, 12:47 PM
 */

#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H

#include <gvt/core/data/Transform.h>

#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/optix/data/Formats.h>
#include <gvt/render/data/Primitives.h>

#include <vector>

namespace gvt {
	namespace render {
		namespace adapter {
			namespace optix {
				namespace data {

                    GVT_TRANSFORM_TEMPLATE // see gvt/core/data/Transform.h

					std::vector<OptixRay> convertRaysToOptix(const gvt::render::actor::RayVector& rays);
    
					template<>
					struct transform_impl<gvt::render::actor::RayVector, std::vector<OptixRay> > {
					    inline static std::vector<OptixRay> transform(const gvt::render::actor::RayVector& rays) {
					        return convertRaysToOptix(rays);
					    }
					};
				}
			}
		}
	}
}

#endif /* GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H */

