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
#include "cuda.h"
#include "OptixMeshAdapter.cuh"
#include "cutil_math.h"

__device__ curandState *globalState;

using namespace gvt;
using namespace render;
using namespace data;
using namespace cuda_primitives;

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

curandState *set_random_states(int rayCount) {


	 dim3 threadsPerBlock = dim3(16, 16);
	  dim3 numBlocks = dim3((rayCount / (threadsPerBlock.x * threadsPerBlock.y)) + 1, 1);

  int N = numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y;
  curandState *devStates;
  cudaMalloc(&devStates, N * sizeof(curandState));

  // setup seeds
  setup_kernel<<<numBlocks, threadsPerBlock>>>(devStates, time(NULL));
  gpuErrchk(cudaGetLastError());

  return devStates;
}

__global__ void cudaKernelPrepOptixRays(OptixRay* optixrays, bool* valid,
                   const int localPacketSize,
                   Ray* rays,
                   CudaGvtContext* cudaGvtCtx, bool ignoreValid) {

	int i = getGlobalIdx_2D_2D();
	if (i >= localPacketSize) return;

    if (ignoreValid || valid[i]) {
       Ray &r = rays[ i];

      r.origin.w=1;
      float4 origin = (*(cudaGvtCtx->minv)) * r.origin; // transform ray to local space
      float4 direction = (*(cudaGvtCtx->minv)) * r.direction;

      OptixRay optix_ray;
      optix_ray.origin[0] = origin.x;
      optix_ray.origin[1] = origin.y;
      optix_ray.origin[2] = origin.z;
      optix_ray.t_min = 0;
      optix_ray.direction[0] = direction.x;
      optix_ray.direction[1] = direction.y;
      optix_ray.direction[2] = direction.z;
      optix_ray.t_max = FLT_MAX;
      optixrays[i] = optix_ray;


    }
}

void cudaPrepOptixRays(OptixRay* optixrays, bool* valid,
                  const int localPacketSize, Ray* rays,
                 CudaGvtContext* cudaGvtCtx,bool ignoreValid, cudaStream_t& stream) {

		dim3 blockDIM = dim3(16, 16);
		dim3 gridDIM = dim3((localPacketSize / (blockDIM.x * blockDIM.y)) + 1, 1);


		cudaKernelPrepOptixRays<<<gridDIM,blockDIM , 0, stream>>>(
				optixrays,valid,localPacketSize,rays,
				cudaGvtCtx->toGPU(),ignoreValid);

		  gpuErrchk(cudaGetLastError());


}

__global__ void cudaKernelFilterShadow( CudaGvtContext* cudaGvtCtx) {

	int tID = getGlobalIdx_2D_2D();
	if (tID >= cudaGvtCtx->shadowRayCount) return;

	 if (cudaGvtCtx->traceHits[tID].triangle_id < 0) {
	          // ray is valid, but did not hit anything, so add to dispatch queue
	    	  int a = atomicAdd((int *)&(cudaGvtCtx->dispatchCount), 1);
	    	  cudaGvtCtx->dispatch[a] = cudaGvtCtx->shadowRays[tID];
	        }
}

void cudaProcessShadows(CudaGvtContext* cudaGvtCtx) {

		dim3 blockDIM = dim3(16, 16);
		dim3 gridDIM = dim3((cudaGvtCtx->shadowRayCount / (blockDIM.x * blockDIM.y)) + 1, 1);

		cudaKernelFilterShadow<<<gridDIM,blockDIM , 0, cudaGvtCtx->stream>>>(cudaGvtCtx->toGPU());
		gpuErrchk(cudaGetLastError());

		cudaGvtCtx->toHost();
}



__device__ void generateShadowRays(const Ray &r, const float4 &normal,
                                   int primID, CudaGvtContext* cudaGvtCtx) {

  for (int l = 0; l < cudaGvtCtx->nLights; l++) {


    Light *light = &(cudaGvtCtx->lights[l]);

    // Try to ensure that the shadow ray is on the correct side of the
    // triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).
    const float multiplier = 1.0f - 16.0f * FLT_EPSILON;
    const float t_shadow = multiplier * r.t;

    float4 origin = r.origin + r.direction * t_shadow;
    origin.w=1.0f;
    const float4 dir = light->light.position - origin;
    const float t_max = length(dir);

    Ray shadow_ray;

    shadow_ray.origin = origin;
    shadow_ray.setDirection(dir);
    shadow_ray.w = r.w;
    shadow_ray.type = Ray::SHADOW;
    shadow_ray.depth = r.depth;
    shadow_ray.t = r.t;
    shadow_ray.id = r.id;
    shadow_ray.t_max = t_max;

    Color c = cudaGvtCtx->mesh.mat->shade(/*primID,*/ shadow_ray, normal, light);


    shadow_ray.color.x = c.x;
    shadow_ray.color.y = c.y;
    shadow_ray.color.z = c.z;
    shadow_ray.color.w = 1.0f;

    int a = atomicAdd((int *)&(cudaGvtCtx->shadowRayCount), 1);
    cudaGvtCtx->shadowRays[a] = shadow_ray;


  }
}



__global__ void kernel(gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx) {

	int tID = getGlobalIdx_2D_2D();

	if (tID >= cudaGvtCtx->rayCount) return;

    if (cudaGvtCtx->valid[tID]) {
      Ray &r = cudaGvtCtx->rays[tID];
      if (cudaGvtCtx->traceHits[tID].triangle_id >= 0) {

        // ray has hit something
        // shadow ray hit something, so it should be dropped
        if (r.type == Ray::SHADOW) {
        	return;
        }

        float t = cudaGvtCtx->traceHits[tID].t;
        r.t = t;

        float4 manualNormal;
        {
          const int triangle_id = cudaGvtCtx->traceHits[tID].triangle_id;
#ifndef FLAT_SHADING
          const float u = cudaGvtCtx->traceHits[tID].u;
          const float v = cudaGvtCtx->traceHits[tID].v;
          const int3 &normals =
        		  cudaGvtCtx->mesh.faces_to_normals[triangle_id]; // FIXME: need to
                                                   // figure out
                                                   // to store
          // `faces_to_normals`
          // list
          const float4 &a =   cudaGvtCtx->mesh.normals[normals.x];
          const float4 &b =   cudaGvtCtx->mesh.normals[normals.y];
          const float4 &c =   cudaGvtCtx->mesh.normals[normals.z];
          manualNormal = a * u + b * v + c * (1.0f - u - v);

          manualNormal =make_float4(
              (*(cudaGvtCtx->normi)) * make_float3(manualNormal.x,
            		  manualNormal.y,manualNormal.z));

          manualNormal=normalize(manualNormal);

#else
          int I = mesh->faces[triangle_id].get<0>();
          int J = mesh->faces[triangle_id].get<1>();
          int K = mesh->faces[triangle_id].get<2>();

          Vector4f a = mesh->vertices[I];
          Vector4f b = mesh->vertices[J];
          Vector4f c = mesh->vertices[K];
          Vector4f u = b - a;
          Vector4f v = c - a;
          Vector4f normal;
          normal.n[0] = u.n[1] * v.n[2] - u.n[2] * v.n[1];
          normal.n[1] = u.n[2] * v.n[0] - u.n[0] * v.n[2];
          normal.n[2] = u.n[0] * v.n[1] - u.n[1] * v.n[0];
          normal.n[3] = 0.0f;
          manualNormal = normal.normalize();
#endif
        }
        const float4 &normal = manualNormal;


        // reduce contribution of the color that the shadow rays get
        if (r.type == Ray::SECONDARY) {
          t = (t > 1) ? 1.f / t : t;
          r.w = r.w * t;
        }

       generateShadowRays(r, normal, cudaGvtCtx->
        		traceHits[tID].triangle_id, cudaGvtCtx);

        int ndepth = r.depth - 1;
        float p = 1.f - cudaRand();
        // replace current ray with generated secondary ray
        if (ndepth > 0 && r.w > p) {
          r.type = Ray::SECONDARY;
          const float multiplier =
              1.0f -
              16.0f *
                 FLT_EPSILON;

          const float t_secondary = multiplier * r.t;
          r.origin = r.origin + r.direction * t_secondary;
          r.origin.w=1.0f;

         float4 dir = normalize(cudaGvtCtx->mesh.mat->material.
                  		  CosWeightedRandomHemisphereDirection2(normal));

          r.setDirection(dir);

          r.w = r.w * (r.direction * normal);
          r.depth = ndepth;
          if (!cudaGvtCtx->validRayLeft)
        	  cudaGvtCtx->validRayLeft = true;

        } else {
        	cudaGvtCtx->valid[tID] = false;
        }
      } else {
        // ray is valid, but did not hit anything, so add to dispatch
    	  int a = atomicAdd((int *)&(cudaGvtCtx->dispatchCount), 1);
    	  cudaGvtCtx->dispatch[a] = r;

    	cudaGvtCtx->valid[tID] = false;

      }
    }

}

void shade(
		gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx) {

	int N= cudaGvtCtx->rayCount;

	dim3 blockDIM = dim3(16, 16);
	dim3 gridDIM = dim3((N / (blockDIM.x * blockDIM.y)) + 1, 1);

	kernel<<<gridDIM,blockDIM , 0, cudaGvtCtx->stream >>>(cudaGvtCtx->toGPU());
	  gpuErrchk(cudaGetLastError());

	cudaGvtCtx->toHost();


}
