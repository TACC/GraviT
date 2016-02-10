/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
   at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use
   this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software
   distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
//
// OptixMeshAdapter.cu
//

#include <float.h>
#include "Mesh.cuh"
#include "Ray.cuh"
#include "Light.cuh"
#include "Material.cuh"

#include "OptixMeshAdapter.cuh"
#include "cutil_math.h"

__device__ curandState *globalState;

using namespace gvt;
using namespace render;
using namespace data;
using namespace cuda_primitives;

__device__ Ray &push_back(Ray *array, uint &size) {
  register uint a = atomicAdd((uint *)&size, 1);

  return array[a];
}

__device__ int getGlobalIdx_2D_2D() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
  int id = getGlobalIdx_2D_2D();
  curand_init(seed, id, 0, &state[id]);

  if (id == 0)
    globalState = state;
}

__device__ float cudaRand() {

  float RANDOM;

  int ind = getGlobalIdx_2D_2D();
  curandState localState = globalState[ind];
  RANDOM = curand_uniform(&localState);
  globalState[ind] = localState;

  return RANDOM;
}

curandState *set_random_states(dim3 numBlocks, dim3 threadsPerBlock) {

  int N = numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y;
  curandState *devStates;
  cudaMalloc(&devStates, N * sizeof(curandState));

  // setup seeds
  setup_kernel<<<numBlocks, threadsPerBlock>>>(devStates, time(NULL));

  printf("Seeds set for %d\n threads", N);
  return devStates;
}

__device__ void generateShadowRays(const Ray &r, const float4 &normal,
                                   int primID, Mesh *mesh,
                                   CudaShade cudaShade) {

  for (int l = 0; l < cudaShade.nLights; l++) {
    Light *light = &(cudaShade.lights[l]);

    // Try to ensure that the shadow ray is on the correct side of the
    // triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).
    const float multiplier = 1.0f - 16.0f * FLT_EPSILON;
    const float t_shadow = multiplier * r.t;

    const float4 origin = r.origin + r.direction * t_shadow;
    const float4 dir = light->light.position - origin;
    const float t_max = length(dir);

    // note: ray copy constructor is too heavy, so going to build it manually

    // shadowRays.push_back(Ray(r.origin + r.direction * t_shadow, dir, r.w,
    // Ray::SHADOW, r.depth));

    Ray &shadow_ray = push_back(cudaShade.shadowRays, cudaShade.shadowRayCount);

    shadow_ray.origin = r.origin + r.direction * t_shadow;
    shadow_ray.direction = dir;
    shadow_ray.w = r.w;
    shadow_ray.type = Ray::SHADOW;
    shadow_ray.depth = r.depth;
    shadow_ray.t = r.t;
    shadow_ray.id = r.id;
    shadow_ray.t_max = t_max;

    // FIXME: remove dependency on mesh->shadeFace
    Color c = cudaShade.mesh.mat->shade(/*primID,*/ shadow_ray, normal, light);
    // gvt::render::data::Color c = adapter->getMesh()->mat->shade(shadow_ray,
    // normal, lights[lindex]);

    shadow_ray.color.t = 1.0f;
    shadow_ray.color.rgba[0] = c.x;
    shadow_ray.color.rgba[1] = c.y;
    shadow_ray.color.rgba[2] = c.z;
    shadow_ray.color.rgba[3] = 1.0f;
  }
}

__global__ void
kernel(gvt::render::data::cuda_primitives::CudaShade cudaShade) {

  printf("GPU mat %f \n", cudaShade.mesh.mat->phong.alpha);
  printf("GPU ray %f \n", cudaShade.rays[3].origin.x);
  printf("GPU light %f \n", cudaShade.lights[0].light.position.x);
}

extern "C" void trace(gvt::render::data::cuda_primitives::CudaShade cudaShade) {

  kernel<<<1, 1, 0>>>(cudaShade);
}
