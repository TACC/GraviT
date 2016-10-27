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
/*
 * File:   Light.cuh
 * Author: Roberto Ribeiro
 *
 * Created on February 4, 2016, 11:00 PM
 */

#include <vector_functions.h>
#include <stdio.h>

#include "Ray.cuh"



#ifndef GVT_RENDER_DATA_SCENE_LIGHT_CUH
#define GVT_RENDER_DATA_SCENE_LIGHT_CUH

namespace gvt {
namespace render {
namespace data {
namespace cuda_primitives {

typedef enum {BASE_LIGHT, AMBIENT, POINT, AREA} LIGH_TYPE;

class BaseLight {
public:

           __device__ cuda_vec contribution(const cuda_vec &hit,const cuda_vec &samplePos) const;


  cuda_vec position;


};
/// general lighting factor added to each successful ray intersection
class AmbientLight : public BaseLight {
public:

           __device__ cuda_vec contribution(const cuda_vec &hit,const cuda_vec &samplePos) const;

  cuda_vec color;
};
/// point light source
class PointLight : public BaseLight {
public:

           __device__  cuda_vec contribution(const cuda_vec &hit,const cuda_vec &samplePos) const;

  cuda_vec color;
};

class AreaLight : public BaseLight {
public:


  __device__ cuda_vec contribution(const cuda_vec &hitpoint, const cuda_vec &samplePos) const;

  __device__ cuda_vec  GetPosition();

  cuda_vec color;
  cuda_vec LightNormal;
  float LightWidth;
  float LightHeight;

  cuda_vec u, v, w;

};


typedef struct {
	LIGH_TYPE type;
	union {
		BaseLight light;
		AmbientLight ambient;
		PointLight point;
                AreaLight area;

	};

           __device__ cuda_vec contribution(const cuda_vec &hit,const cuda_vec &samplePos) const {
		cuda_vec r;
		switch (type) {
		case BASE_LIGHT:
                        r = light.contribution(hit,samplePos);
			break;
		case AMBIENT:
                        r = ambient.contribution(hit,samplePos);
			break;
		case POINT:
                        r = point.contribution(hit,samplePos);
			break;
                case AREA:
                        r = area.contribution(hit,samplePos);
                            break;
		default:
			break;
		}

		return r;
	}

} Light;

}
}
}
}

#endif /* GVT_RENDER_DATA_SCENE_LIGHT_H */
