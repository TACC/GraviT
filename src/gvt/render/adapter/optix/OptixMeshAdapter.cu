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
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
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
	dim3 numBlocks = dim3(
			(rayCount / (threadsPerBlock.x * threadsPerBlock.y)) + 1, 1);

	int N = numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y;
	curandState *devStates;
	cudaMalloc(&devStates, N * sizeof(curandState));

	// setup seeds
	setup_kernel<<<numBlocks, threadsPerBlock>>>(devStates, time(NULL));
	gpuErrchk(cudaGetLastError());

	return devStates;
}

__global__ void cudaKernelPrepOptixRays(OptixRay* optixrays, bool* valid,
		const int localPacketSize, Ray* rays, CudaGvtContext* cudaGvtCtx,
		bool ignoreValid) {

	int i = getGlobalIdx_2D_2D();
	if (i >= localPacketSize)
		return;

	if (ignoreValid || valid[i]) {
		Ray &r = rays[i];

		//r.origin.w=1;
		cuda_vec origin = make_float3(
				(*(cudaGvtCtx->minv)) * make_float4(r.origin, 1.0f)); // transform ray to local space
		cuda_vec direction = make_float3(
				(*(cudaGvtCtx->minv)) * make_float4(r.direction, 0.0f));

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
		const int localPacketSize, Ray* rays, CudaGvtContext* cudaGvtCtx,
		bool ignoreValid, cudaStream_t& stream) {

	dim3 blockDIM = dim3(16, 16);
	dim3 gridDIM = dim3((localPacketSize / (blockDIM.x * blockDIM.y)) + 1, 1);

	cudaKernelPrepOptixRays<<<gridDIM, blockDIM, 0, stream>>>(optixrays, valid,
			localPacketSize, rays, cudaGvtCtx->toGPU(), ignoreValid);

	gpuErrchk(cudaGetLastError());

}

__global__ void cudaKernelFilterShadow(CudaGvtContext* cudaGvtCtx) {

	int tID = getGlobalIdx_2D_2D();
	if (tID >= cudaGvtCtx->shadowRayCount)
		return;

	if (cudaGvtCtx->traceHits[tID].triangle_id < 0) {
		// ray is valid, but did not hit anything, so add to dispatch queue
		int a = atomicAdd((int *) &(cudaGvtCtx->dispatchCount), 1);
		cudaGvtCtx->dispatch[a] = cudaGvtCtx->shadowRays[tID];
	}
}

void cudaProcessShadows(CudaGvtContext* cudaGvtCtx) {

	dim3 blockDIM = dim3(16, 16);
	dim3 gridDIM = dim3(
			(cudaGvtCtx->shadowRayCount / (blockDIM.x * blockDIM.y)) + 1, 1);

	cudaKernelFilterShadow<<<gridDIM, blockDIM, 0, cudaGvtCtx->stream>>>(
			cudaGvtCtx->toGPU());
	gpuErrchk(cudaGetLastError());

	cudaGvtCtx->toHost();
}

__device__ cuda_vec CosWeightedRandomHemisphereDirection2(cuda_vec n) {

	float Xi1 = cudaRand();
	float Xi2 = cudaRand();

	float theta = acos(sqrt(1.0 - Xi1));
	float phi = 2.0 * 3.1415926535897932384626433832795 * Xi2;

	float xs = sinf(theta) * cosf(phi);
	float ys = cosf(theta);
	float zs = sinf(theta) * sinf(phi);

	float3 y = make_float3(n);
	float3 h = y;
	if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
		h.x = 1.0;
	else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
		h.y = 1.0;
	else
		h.z = 1.0;

	float3 x = cross(h, y); //(h ^ y);
	float3 z = cross(x, y);

	cuda_vec direction = make_cuda_vec(x * xs + y * ys + z * zs);
	return normalize(direction);
}

__device__ void generateShadowRays(const Ray &r, const cuda_vec &normal,
		int primID, CudaGvtContext* cudaGvtCtx) {

	for (int l = 0; l < cudaGvtCtx->nLights; l++) {

		Light *light = &(cudaGvtCtx->lights[l]);

		// Try to ensure that the shadow ray is on the correct side of the
		// triangle.
		// Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
		// Using about 8 * ULP(t).

		cuda_vec lightPos;
		if (light->type == AREA) {
			lightPos = ((AreaLight *) light)->GetPosition();
		} else {
			lightPos = light->light.position;
		}

		cuda_vec c;
		if (!gvt::render::data::cuda_primitives::Shade(cudaGvtCtx->mesh.mat, r,
				normal, light, lightPos, c))
			continue;

		const float multiplier = 1.0f - 16.0f * FLT_EPSILON;
		const float t_shadow = multiplier * r.t;

		cuda_vec origin = r.origin + r.direction * t_shadow;
		//origin.w=1.0f;
		const cuda_vec dir = lightPos - origin;
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
		shadow_ray.color.x = c.x;
		shadow_ray.color.y = c.y;
		shadow_ray.color.z = c.z;

		int a = atomicAdd((int *) &(cudaGvtCtx->shadowRayCount), 1);
		cudaGvtCtx->shadowRays[a] = shadow_ray;

	}
}

__global__ void kernel(
		gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx) {

	int tID = getGlobalIdx_2D_2D();

	if (tID >= cudaGvtCtx->rayCount)
		return;

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

			const int triangle_id = cudaGvtCtx->traceHits[tID].triangle_id;

			cuda_vec manualNormal;
			cuda_vec normalflat;

			{
				int I = cudaGvtCtx->mesh.faces[triangle_id].x;
				int J = cudaGvtCtx->mesh.faces[triangle_id].y;
				int K = cudaGvtCtx->mesh.faces[triangle_id].z;

				cuda_vec a = cudaGvtCtx->mesh.vertices[I];
				cuda_vec b = cudaGvtCtx->mesh.vertices[J];
				cuda_vec c = cudaGvtCtx->mesh.vertices[K];
				cuda_vec u = b - a;
				cuda_vec v = c - a;
				cuda_vec normal;
				normal.x = u.y * v.z - u.z * v.y;
				normal.y = u.z * v.x - u.x * v.z;
				normal.z = u.x * v.y - u.y * v.x;
				//normal.w = 0.0f;
				normalflat = normalize(normal);

				normalflat = normalize(
						make_cuda_vec(
								(*(cudaGvtCtx->normi))
										* make_float3(normalflat.x,
												normalflat.y, normalflat.z)));

			}
			{

#ifndef FLAT_SHADING
				{
					const float u = cudaGvtCtx->traceHits[tID].u;
					const float v = cudaGvtCtx->traceHits[tID].v;
					const int3 &normals =
							cudaGvtCtx->mesh.faces_to_normals[triangle_id]; // FIXME: need to
					// figure out
					// to store
					// `faces_to_normals`
					// list
					const cuda_vec &a = cudaGvtCtx->mesh.normals[normals.x];
					const cuda_vec &b = cudaGvtCtx->mesh.normals[normals.y];
					const cuda_vec &c = cudaGvtCtx->mesh.normals[normals.z];
					manualNormal = a * u + b * v + c * (1.0f - u - v);

					manualNormal = make_cuda_vec(
							(*(cudaGvtCtx->normi))
									* make_float3(manualNormal.x,
											manualNormal.y, manualNormal.z));

					manualNormal = normalize(manualNormal);
				}
#else

				manualNormal = normalflat;

#endif
			}

			//backface check, requires flat normal
			if ((-(r.direction) * normalflat) <= 0.f) {
				manualNormal = -manualNormal;
			}

			const cuda_vec &normal = manualNormal;

			// reduce contribution of the color that the shadow rays get
			if (r.type == Ray::SECONDARY) {
				t = (t > 1) ? 1.f / t : t;
				r.w = r.w * t;
			}

			generateShadowRays(r, normal,
					cudaGvtCtx->traceHits[tID].triangle_id, cudaGvtCtx);

			int ndepth = r.depth - 1;
			float p = 1.f - cudaRand();
			// replace current ray with generated secondary ray
			if (ndepth > 0 && r.w > p) {
				r.type = Ray::SECONDARY;
				const float multiplier = 1.0f - 16.0f *
				FLT_EPSILON;

				const float t_secondary = multiplier * r.t;
				r.origin = r.origin + r.direction * t_secondary;
				//r.origin.w=1.0f;

				cuda_vec dir = normalize(
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
			int a = atomicAdd((int *) &(cudaGvtCtx->dispatchCount), 1);
			cudaGvtCtx->dispatch[a] = r;

			cudaGvtCtx->valid[tID] = false;

		}
	}

}

void shade(gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx) {

	int N = cudaGvtCtx->rayCount;

	dim3 blockDIM = dim3(16, 16);
	dim3 gridDIM = dim3((N / (blockDIM.x * blockDIM.y)) + 1, 1);

	kernel<<<gridDIM, blockDIM, 0, cudaGvtCtx->stream>>>(cudaGvtCtx->toGPU());
	gpuErrchk(cudaGetLastError());

	cudaGvtCtx->toHost();

}
