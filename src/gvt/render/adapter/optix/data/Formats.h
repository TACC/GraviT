/* 
 * File:   optixdata.h
 * Author: jbarbosa
 *
 * Created on January 11, 2015, 10:24 PM
 */

#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H
#define	GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H


namespace gvt {
	namespace render {
		namespace adapter {
			namespace optix {
				namespace data {

					struct OptixRay {
					  float origin[3];
					  float t_min;
					  float direction[3];
					  float t_max;
					};

					struct OptixHit {
					  float t;
					  int triangle_id;
					  float u;
					  float v;
					};
				}
			}
		}
	}
}

#endif	/* GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H */

