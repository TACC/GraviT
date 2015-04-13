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

					std::vector<OptixRayFormat> convertRaysToOptix(const RayVector& rays);
    
					template<>
					struct transform_impl<RayVector, std::vector<OptixRayFormat> > {
					    inline static std::vector<OptixRayFormat> transform(const RayVector& rays) {
					        return convertRaysToOptix(rays);
					    }
					};
				}
			}
		}
	}
}

#endif /* GVT_RENDER_ADAPTER_OPTIX_DATA_TRANSFORMS_H */

