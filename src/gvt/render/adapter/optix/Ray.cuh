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
 * File:   Ray.cuh
 * Author: Roberto Ribeiro
 *
 * Created on February 4, 2016, 19:00 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_RAY_CUH
#define GVT_RENDER_DATA_PRIMITIVES_RAY_CUH

#include <vector_functions.h>
#include <stdio.h>
#include <float.h>

namespace gvt {
namespace render {
namespace data {
namespace cuda_primitives {

#define cuda_vec float3
typedef cuda_vec Color;

#define make_cuda_vec make_float3

struct OptixRay {
	float origin[3];
	float t_min;
	float direction[3];
	float t_max;

	__device__
	void print() {
		printf("optix gpu ray  o: %f %f %f, d: %f %f %f \n", origin[0],
				origin[1], origin[2], direction[0], direction[1], direction[2]);
	}
};

/// OptiX hit format
struct OptixHit {
	float t;
	int triangle_id;
	float u;
	float v;

	__device__
	void print() {
		printf("gpu hit  t: %f , triID: %d \n", t, triangle_id);
	}
};

class Ray {
public:

	typedef enum {
		PRIMARY, SHADOW, SECONDARY
	} RayType;


	struct {

		cuda_vec origin;
		float t_min;
		cuda_vec direction;
		float t_max;
		cuda_vec color;
		float t;
		int id;    ///<! index into framebuffer
		int depth; ///<! sample rate
		float w;   ///<! weight of image contribution
		int type;

	};

	__device__
	void print() {
		printf("cuda gpu ray  o: %f %f %f, d: %f %f %f \n", origin.x, origin.y,
				origin.z, direction.x, direction.y, direction.z);
	}

	__device__
	void setDirection(cuda_vec dir);

};

}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_RAY_CUH */
