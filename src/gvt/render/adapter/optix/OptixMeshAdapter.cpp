/* =======================================================================================
 This file is released as part of GraviT - scalable, platform independent ray
 tracing
 tacc.github.io/GraviT

 Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at
 Austin
 All rights reserved.

 Licensed under the BSD 3-Clause License, (the "License"); you may not use this
 file
 except in compliance with the License.
 A copy of the License is included with this software in the file LICENSE.
 If your copy does not contain the License, you may obtain a copy of the License
 at:

 http://opensource.org/licenses/BSD-3-Clause

 Unless required by applicable law or agreed to in writing, software distributed
 under
 the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 CONDITIONS OF ANY
 KIND, either express or implied.
 See the License for the specific language governing permissions and limitations
 under
 limitations under the License.

 GraviT is funded in part by the US National Science Foundation under awards
 ACI-1339863,
 ACI-1339881 and ACI-1339840
 =======================================================================================
 */
//
// OptixMeshAdapter.cpp
//
#include "gvt/render/adapter/optix/data/OptixMeshAdapter.h"
#include "gvt/core/CoreContext.h"

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Mesh.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

#include <atomic>

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>
#include <tbb/task_group.h>
#include <tbb/parallel_for.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>
#include <optix.h>
#include <optix_cuda_interop.h>
#include "optix_prime/optix_prime.h"

#include "OptixMeshAdapter.cuh"


#include <float.h>

#include <gvt/render/data/primitives/Material.h>


__inline__ void cudaRayToGvtRay(
		const gvt::render::data::cuda_primitives::Ray& cudaRay,
		gvt::render::actor::Ray& gvtRay) {

	memcpy(&(gvtRay.origin[0]), &(cudaRay.origin.x), sizeof(glm::vec3));
	memcpy(&(gvtRay.direction[0]), &(cudaRay.direction.x), sizeof(glm::vec3));
//	memcpy(&(gvtRay.inverseDirection[0]), &(cudaRay.inverseDirection.x),
//			sizeof(float) * 4);
	memcpy(glm::value_ptr(gvtRay.color), &(cudaRay.color.x),
			sizeof(glm::vec3));
	gvtRay.id = cudaRay.id;
	gvtRay.depth = cudaRay.depth;
	gvtRay.w = cudaRay.w;
	gvtRay.t = cudaRay.t;
	gvtRay.t_min = cudaRay.t_min;
	gvtRay.t_max = cudaRay.t_max;
	gvtRay.type = cudaRay.type;

}

__inline__ void gvtRayToCudaRay(const gvt::render::actor::Ray& gvtRay,
		gvt::render::data::cuda_primitives::Ray& cudaRay) {

	memcpy(&(cudaRay.origin.x), &(gvtRay.origin[0]), sizeof(glm::vec3));
	cudaRay.origin.w=1.0f;
	memcpy(&(cudaRay.direction.x), &(gvtRay.direction[0]),sizeof(glm::vec3));
//	memcpy(&(cudaRay.inverseDirection.x), &(gvtRay.inverseDirection[0]),
//			sizeof(float) * 4);
	memcpy(&(cudaRay.color.x), glm::value_ptr(gvtRay.color),
			sizeof(float4));
	cudaRay.id = gvtRay.id;
	cudaRay.depth = gvtRay.depth;
	cudaRay.w = gvtRay.w;
	cudaRay.t = gvtRay.t;
	cudaRay.t_min = gvtRay.t_min;
	cudaRay.t_max = gvtRay.t_max;
	cudaRay.type = gvtRay.type;

}

int3*
cudaCreateFacesToNormals(
		std::vector<gvt::render::data::primitives::Mesh::FaceToNormals>& gvt_face_to_normals) {

	int3* faces_to_normalsBuff;

	gpuErrchk(
			cudaMalloc((void ** ) &faces_to_normalsBuff,
					sizeof(int3) * gvt_face_to_normals.size()));

	std::vector<int3> faces_to_normals;
	for (int i = 0; i < gvt_face_to_normals.size(); i++) {

		const boost::tuple<int, int, int> &f = gvt_face_to_normals[i];

		int3 v = make_int3(f.get<0>(), f.get<1>(), f.get<2>());

		faces_to_normals.push_back(v);
	}

	gpuErrchk(
			cudaMemcpy(faces_to_normalsBuff, &faces_to_normals[0],
					sizeof(int3) * faces_to_normals.size(),
					cudaMemcpyHostToDevice));

	return faces_to_normalsBuff;

}

float4*
cudaCreateNormals(std::vector<glm::vec3>& gvt_normals) {

	float4* normalsBuff;

	gpuErrchk(
			cudaMalloc((void ** ) &normalsBuff,
					sizeof(float4) * gvt_normals.size()));

	std::vector<float4> normals;
	for (int i = 0; i < gvt_normals.size(); i++) {

		float4 v = make_float4(gvt_normals[i].x, gvt_normals[i].y,
				gvt_normals[i].z, 0.f);
		normals.push_back(v);
	}

	gpuErrchk(
			cudaMemcpy(normalsBuff, &normals[0],
					sizeof(float4) * gvt_normals.size(),
					cudaMemcpyHostToDevice));

	return normalsBuff;

}

int3*
cudaCreateFaces(
		std::vector<gvt::render::data::primitives::Mesh::Face>& gvt_faces) {

	int3* facesBuff;

	gpuErrchk(
			cudaMalloc((void ** ) &facesBuff, sizeof(int3) * gvt_faces.size()));

	std::vector<int3> faces;
	for (int i = 0; i < gvt_faces.size(); i++) {

		const boost::tuple<int, int, int> &f = gvt_faces[i];

		int3 v = make_int3(f.get<0>(), f.get<1>(), f.get<2>());
		faces.push_back(v);
	}

	gpuErrchk(
			cudaMemcpy(facesBuff, &faces[0], sizeof(int3) * gvt_faces.size(),
					cudaMemcpyHostToDevice));

	return facesBuff;

}

float4*
cudaCreateVertices(std::vector<glm::vec3>& gvt_verts) {

	float4* buff;

	gpuErrchk(
			cudaMalloc((void ** ) &buff,
					sizeof(float4) * gvt_verts.size()));

	std::vector<float4> verts;
	for (int i = 0; i < gvt_verts.size(); i++) {

		float4 v = make_float4(gvt_verts[i].x, gvt_verts[i].y,
				gvt_verts[i].z, 0.f);
		verts.push_back(v);
	}

	gpuErrchk(
			cudaMemcpy(buff, &verts[0],
					sizeof(float4) * gvt_verts.size(),
					cudaMemcpyHostToDevice));

	return buff;

}
gvt::render::data::primitives::Material *
cudaCreateMaterial(gvt::render::data::primitives::Material *gvtMat) {

	gvt::render::data::primitives::Material *cudaMat_ptr;

	gpuErrchk(
			cudaMalloc((void ** ) &cudaMat_ptr,
					sizeof(gvt::render::data::primitives::Material)));

	gpuErrchk(
			cudaMemcpy(cudaMat_ptr, gvtMat,
					sizeof(gvt::render::data::primitives::Material),
					cudaMemcpyHostToDevice));

	return cudaMat_ptr;
}

void cudaSetRays(gvt::render::actor::RayVector::iterator gvtRayVector,
		int localRayCount,
		gvt::render::data::cuda_primitives::Ray *cudaRays_devPtr, int localIdx,
		cudaStream_t& stream,
		gvt::render::data::cuda_primitives::Ray* cudaRays) {

	const int offset_rays =
			localRayCount > std::thread::hardware_concurrency() ?
					localRayCount / std::thread::hardware_concurrency() : 100;


	static tbb::auto_partitioner ap;
	tbb::parallel_for(tbb::blocked_range<int>(0, localRayCount, 128),
			[&](tbb::blocked_range<int> chunk) {
				for (int jj = chunk.begin(); jj < chunk.end(); jj++) {
						gvtRayToCudaRay(gvtRayVector[jj], cudaRays[jj]);

				}}, ap);


	gpuErrchk(
			cudaMemcpyAsync(cudaRays_devPtr, &cudaRays[0],
					sizeof(gvt::render::data::cuda_primitives::Ray)
							* localRayCount, cudaMemcpyHostToDevice, stream));


}

void cudaGetRays(size_t& localDispatchSize,
		gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx,
		gvt::render::data::cuda_primitives::Ray* disp_tmp,
		gvt::render::actor::RayVector& localDispatch,
		gvt::render::actor::RayVector::iterator &rayList) {


	gpuErrchk(
			cudaMemcpyAsync(&disp_tmp[0], cudaGvtCtx.dispatch,
					sizeof(gvt::render::data::cuda_primitives::Ray)
							* cudaGvtCtx.dispatchCount, cudaMemcpyDeviceToHost,
					cudaGvtCtx.stream));

	gpuErrchk(cudaStreamSynchronize(cudaGvtCtx.stream));


	static tbb::auto_partitioner ap;
	tbb::parallel_for(tbb::blocked_range<int>(0, cudaGvtCtx.dispatchCount, 128),
			[&](tbb::blocked_range<int> chunk) {
				for (int jj = chunk.begin(); jj < chunk.end(); jj++) {
					if (jj < cudaGvtCtx.dispatchCount) {
						gvt::render::actor::Ray& gvtRay =
						localDispatch[localDispatchSize + jj];
						const gvt::render::data::cuda_primitives::Ray& cudaRay =
						disp_tmp[jj];

						cudaRayToGvtRay(cudaRay, gvtRay);

						//gvtRay.setDirection(gvtRay.direction);
					}
				}
			}, ap);


	localDispatchSize += cudaGvtCtx.dispatchCount;
	cudaGvtCtx.dispatchCount = 0;


}

void cudaGetLights(std::vector<gvt::render::data::scene::Light *> gvtLights,
		gvt::render::data::cuda_primitives::Light * cudaLights_devPtr) {

	gvt::render::data::cuda_primitives::Light *cudaLights =
			new gvt::render::data::cuda_primitives::Light[gvtLights.size()];

	for (int i = 0; i < gvtLights.size(); i++) {

		if (dynamic_cast<gvt::render::data::scene::AmbientLight *>(gvtLights[i])
				!=
				NULL) {

			gvt::render::data::scene::AmbientLight *l =
					dynamic_cast<gvt::render::data::scene::AmbientLight *>(gvtLights[i]);

			memcpy(&(cudaLights[i].ambient.position.x), &(l->position.x),
					sizeof(float4));
			memcpy(&(cudaLights[i].ambient.color.x), &(l->color.x),
					sizeof(float4));

			cudaLights[i].type =
					gvt::render::data::cuda_primitives::LIGH_TYPE::AMBIENT;

		} else if (dynamic_cast<gvt::render::data::scene::AreaLight *>(gvtLights[i])
				!= NULL) {

			gvt::render::data::scene::AreaLight *l =
					dynamic_cast<gvt::render::data::scene::AreaLight *>(gvtLights[i]);

			memcpy(&(cudaLights[i].area.position.x), &(l->position.x),
					sizeof(float4));

			memcpy(&(cudaLights[i].area.color.x), &(l->color.x),
					sizeof(float4));

			memcpy(&(cudaLights[i].area.u.x), &(l->u.x),
					sizeof(float4));

			memcpy(&(cudaLights[i].area.v.x), &(l->v.x),
					sizeof(float4));

			memcpy(&(cudaLights[i].area.w.x), &(l->w.x),
					sizeof(float4));

			memcpy(&(cudaLights[i].area.LightNormal.x), &(l->LightNormal.x),
					sizeof(float4));

			cudaLights[i].area.LightHeight = l->LightHeight;

			cudaLights[i].area.LightWidth = l->LightWidth;


			cudaLights[i].type =
					gvt::render::data::cuda_primitives::LIGH_TYPE::AREA;

		  } else if (dynamic_cast<gvt::render::data::scene::PointLight *>(gvtLights[i])
				  != NULL) {

                          gvt::render::data::scene::PointLight *l =
                                          dynamic_cast<gvt::render::data::scene::PointLight *>(gvtLights[i]);

                          memcpy(&(cudaLights[i].point.position.x), &(l->position.x),
                                          sizeof(float4));
                          memcpy(&(cudaLights[i].point.color.x), &(l->color.x),
                                          sizeof(float4));

                          cudaLights[i].type =
                                          gvt::render::data::cuda_primitives::LIGH_TYPE::POINT;


		} else {
			std::cout << "Unknown light" << std::endl;
			return;
		}
	}

	gpuErrchk(
			cudaMemcpy(cudaLights_devPtr, cudaLights,
					sizeof(gvt::render::data::cuda_primitives::Light)
							* gvtLights.size(), cudaMemcpyHostToDevice));

	delete[] cudaLights;

}

gvt::render::data::cuda_primitives::Mesh cudaInstanceMesh(
		gvt::render::data::primitives::Mesh* mesh) {

	gvt::render::data::cuda_primitives::Mesh cudaMesh;

	cudaMesh.faces_to_normals = cudaCreateFacesToNormals(
			mesh->faces_to_normals);
	cudaMesh.normals = cudaCreateNormals(mesh->normals);
	cudaMesh.faces = cudaCreateFaces(mesh->faces);
	cudaMesh.mat = cudaCreateMaterial(mesh->mat);
	cudaMesh.vertices = cudaCreateVertices(mesh->vertices);


	return cudaMesh;
}

void gvt::render::data::cuda_primitives::CudaGvtContext::initCudaBuffers(
		int packetSize) {

	gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

	auto lightNodes = root["Lights"].getChildren();
	std::vector<gvt::render::data::scene::Light *> lights;
	lights.reserve(2);
	for (auto lightNode : lightNodes) {
		auto color = lightNode["color"].value().tovec3();

		if (lightNode.name() == std::string("PointLight")) {
			auto pos = lightNode["position"].value().tovec3();
			lights.push_back(
					new gvt::render::data::scene::PointLight(pos, color));
		} else if (lightNode.name() == std::string("AmbientLight")) {
			lights.push_back(new gvt::render::data::scene::AmbientLight(color));
		}
		 else if (lightNode.name() == std::string("AreaLight")) {
			 auto pos = lightNode["position"].value().tovec3();
			         auto normal = lightNode["normal"].value().tovec3();
			         auto width = lightNode["width"].value().toFloat();
			         auto height = lightNode["height"].value().toFloat();
			         lights.push_back(new gvt::render::data::scene::AreaLight(pos, color, normal, width, height));
		 }
	}

	gvt::render::data::cuda_primitives::Light *cudaLights_devPtr;
	gpuErrchk(
			cudaMalloc((void ** )&cudaLights_devPtr,
					sizeof(gvt::render::data::cuda_primitives::Light)
							* lights.size()));

	cudaGetLights(lights, cudaLights_devPtr);

	gvt::render::data::cuda_primitives::Matrix3f *normiBuff;
	gpuErrchk(
			cudaMalloc((void ** ) &normiBuff,
					sizeof(gvt::render::data::cuda_primitives::Matrix3f)));

	gvt::render::data::cuda_primitives::Matrix4f *minvBuff;
	gpuErrchk(
			cudaMalloc((void ** ) &minvBuff,
					sizeof(gvt::render::data::cuda_primitives::Matrix4f)));

	gvt::render::data::cuda_primitives::OptixRay *cudaOptixRayBuff;
	gpuErrchk(
			cudaMalloc((void ** ) &cudaOptixRayBuff,
					sizeof(gvt::render::data::cuda_primitives::OptixRay)
							* packetSize * lights.size()));

	gvt::render::data::cuda_primitives::OptixHit *cudaHitsBuff;
	gpuErrchk(
			cudaMalloc((void ** ) &cudaHitsBuff,
					sizeof(gvt::render::data::cuda_primitives::OptixHit)
							* packetSize * lights.size()));

	gvt::render::data::cuda_primitives::Ray *shadowRaysBuff;
	gpuErrchk(
			cudaMalloc((void ** ) &shadowRaysBuff,
					sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize
							* lights.size()));

	bool *validBuff;
	gpuErrchk(cudaMalloc((void **) &validBuff, sizeof(bool) * packetSize));

	gvt::render::data::cuda_primitives::Ray *cudaRays_devPtr;
	gpuErrchk(
			cudaMalloc((void ** ) &cudaRays_devPtr,
					sizeof(gvt::render::data::cuda_primitives::Ray)
							* packetSize));

	// Size is rayCount but depends on instance, so majoring to double packetsize
	gvt::render::data::cuda_primitives::Ray * dispatchBuff;
	gpuErrchk(
			cudaMalloc((void ** ) &dispatchBuff,
					sizeof(gvt::render::data::cuda_primitives::Ray) *
					/*(end-begin)*/packetSize * 2));

	set_random_states(packetSize);

	gpuErrchk(
			cudaMalloc(&devicePtr,
					sizeof(gvt::render::data::cuda_primitives::CudaGvtContext)));

	gpuErrchk(cudaStreamCreate(&stream));

	rays = cudaRays_devPtr;
	traceRays = cudaOptixRayBuff;
	traceHits = cudaHitsBuff;
	this->lights = cudaLights_devPtr;
	shadowRays = shadowRaysBuff;
	nLights = lights.size();
	valid = validBuff;
	dispatch = dispatchBuff;
	normi = normiBuff;
	minv = minvBuff;

}

// TODO: add logic for other packet sizes
#define GVT_OPTIX_PACKET_SIZE 4096

using namespace gvt::render::actor;
using namespace gvt::render::adapter::optix::data;
using namespace gvt::render::data::primitives;

static std::atomic<size_t> counter(0);

// bool OptixMeshAdapter::init = false;

struct OptixRay {
	float origin[3];
	float t_min;
	float direction[3];
	float t_max;
	friend std::ostream &operator<<(std::ostream &os, const OptixRay &r) {
		return (os << "ray  o: " << r.origin[0] << ", " << r.origin[1] << ", "
				<< r.origin[2] << " d: " << r.direction[0] << ", "
				<< r.direction[1] << ", " << r.direction[2]);
	}
};

/// OptiX hit format
struct OptixHit {
	float t;
	int triangle_id;
	float u;
	float v;
	friend std::ostream &operator<<(std::ostream &os, const OptixHit &oh) {
		return (os << "hit  t: " << oh.t << " triID: " << oh.triangle_id);
	}
};

OptixContext *OptixContext::_singleton;

OptixMeshAdapter::OptixMeshAdapter(gvt::render::data::primitives::Mesh *m) : Adapter(m),
		packetSize(GVT_OPTIX_PACKET_SIZE), optix_context_(
				OptixContext::singleton()->context()) {


	GVT_ASSERT(optix_context_.isValid(), "Optix Context is not valid");
	GVT_ASSERT(mesh, "OptixMeshAdapter: mesh pointer in the database is null");

	int numVerts = mesh->vertices.size();
	int numTris = mesh->faces.size();

	// Create Optix Prime Context

	// Use all CUDA devices, if multiple are present ignore the GPU driving the
	// display
	{
		std::vector<unsigned> activeDevices;
		int devCount = 0;
		cudaDeviceProp prop;
		gpuErrchk(cudaGetDeviceCount(&devCount));
		GVT_ASSERT(devCount,
				"You choose optix render, but no cuda capable devices are present");

		for (int i = 0; i < devCount; i++) {
			gpuErrchk(cudaGetDeviceProperties(&prop, i));
//			if (prop.kernelExecTimeoutEnabled == 0)
//				activeDevices.push_back(i);
			// Oversubcribe the GPU
			packetSize = prop.multiProcessorCount
					* prop.maxThreadsPerMultiProcessor;

		}
		if (!activeDevices.size()) {
			activeDevices.push_back(0);
		}
		optix_context_->setCudaDeviceNumbers(activeDevices);
		gpuErrchk(cudaGetLastError());
	}

	cudaSetDevice(0);

	OptixContext::singleton()->initCuda(packetSize);
	cudaMesh = cudaInstanceMesh(mesh);

	cudaMallocHost(&(disp_Buff[0]),
			sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize * 2);
	cudaMallocHost(&(cudaRaysBuff[0]),
			sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize);
	cudaMallocHost(&(disp_Buff[1]),
			sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize * 2);
	cudaMallocHost(&(cudaRaysBuff[1]),
			sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize);

	cudaMallocHost(&(m_pinned), sizeof(glm::mat4));

	cudaMallocHost(&(minv_pinned), sizeof(glm::mat4));

	cudaMallocHost(&(normi_pinned), sizeof(glm::mat3));

	// Setup the buffer to hold our vertices.
	//
	std::vector<float> vertices;
	std::vector<int> faces;

	vertices.resize(numVerts * 3);
	faces.resize(numTris * 3);

	static tbb::auto_partitioner ap;
	tbb::parallel_for(tbb::blocked_range<int>(0, numVerts, 128),
			[&](tbb::blocked_range<int> chunk) {
				for (int jj = chunk.begin(); jj < chunk.end(); jj++) {
					vertices[jj * 3 + 0] = mesh->vertices[jj][0];
					vertices[jj * 3 + 1] = mesh->vertices[jj][1];
					vertices[jj * 3 + 2] = mesh->vertices[jj][2];
				}
			},ap);

	tbb::parallel_for(tbb::blocked_range<int>(0, numTris, 128),
			[&](tbb::blocked_range<int> chunk) {
				for (int jj = chunk.begin(); jj < chunk.end(); jj++) {
					gvt::render::data::primitives::Mesh::Face f = mesh->faces[jj];
					faces[jj * 3 + 0] = f.get<0>();
					faces[jj * 3 + 1] = f.get<1>();
					faces[jj * 3 + 2] = f.get<2>();
				}
			},ap);

	// Create and setup vertex buffer
	::optix::prime::BufferDesc vertices_desc;
	vertices_desc = optix_context_->createBufferDesc(
			RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST,
			&vertices[0]);

	GVT_ASSERT(vertices_desc.isValid(), "Vertices are not valid");
	vertices_desc->setRange(0, vertices.size() / 3);
	vertices_desc->setStride(sizeof(float) * 3);

	// Create and setup triangle buffer
	::optix::prime::BufferDesc indices_desc;
	indices_desc = optix_context_->createBufferDesc(
			RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, &faces[0]);

	GVT_ASSERT(indices_desc.isValid(), "Indices are not valid");
	indices_desc->setRange(0, faces.size() / 3);
	indices_desc->setStride(sizeof(int) * 3);

	// Create an Optix model.
	optix_model_ = optix_context_->createModel();
	GVT_ASSERT(optix_model_.isValid(), "Model is not valid");
	optix_model_->setTriangles(indices_desc, vertices_desc);
	optix_model_->update(RTP_MODEL_HINT_ASYNC);
	optix_model_->finish();
}

OptixMeshAdapter::~OptixMeshAdapter() {

	cudaFreeHost(disp_Buff);
	cudaFreeHost(cudaRaysBuff);
	cudaFreeHost(m_pinned);
	cudaFreeHost(minv_pinned);
	cudaFreeHost(normi_pinned);

}

struct OptixParallelTrace {

	gvt::render::adapter::optix::data::OptixMeshAdapter *adapter;

	/**
	 * Shared ray list used in the current trace() call
	 */
	gvt::render::actor::RayVector::iterator &rayList;

	/**
	 * Shared outgoing ray list used in the current trace() call
	 */
	gvt::render::actor::RayVector &moved_rays;

	/**
	 *
	 * Stored transformation matrix in the current instance
	 */
	const glm::mat4 *m;

	/**
	 * Stored inverse transformation matrix in the current instance
	 */
	const glm::mat4 *minv;

	/**
	 * Stored upper33 inverse matrix in the current instance
	 */
	const glm::mat3 *normi;


	/**
	 * Thread local outgoing ray queue
	 */
	gvt::render::actor::RayVector localDispatch;

	gvt::render::data::cuda_primitives::Ray* disp_Buff;

	gvt::render::data::cuda_primitives::Ray* cudaRaysBuff;

	/**
	 * Size of Optix-CUDA packet
	 */
	size_t packetSize; // TODO: later make this configurable

	const size_t begin, end;

	/**
	 * Construct a OptixParallelTrace struct with information needed for the
	 * thread
	 * to do its tracing
	 */
	OptixParallelTrace(
			gvt::render::adapter::optix::data::OptixMeshAdapter *adapter,
			gvt::render::actor::RayVector::iterator &rayList,
			gvt::render::actor::RayVector &moved_rays,
			 glm::mat4 *m, glm::mat4 *minv,
			glm::mat3 *normi,
			//std::vector<gvt::render::data::scene::Light *> &lights,
			std::atomic<size_t> &counter, const size_t begin, const size_t end,
			gvt::render::data::cuda_primitives::Ray* disp_Buff,
			gvt::render::data::cuda_primitives::Ray* cudaRaysBuff) :
			adapter(adapter), rayList(rayList), moved_rays(moved_rays),  m(m), minv(minv), normi(normi),
			packetSize(adapter->getPacketSize()), begin(begin), end(end), disp_Buff(
					disp_Buff), cudaRaysBuff(cudaRaysBuff) {
	}

	/**
	 * Test occlusion for stored shadow rays.  Add missed rays
	 * to the dispatch queue.
	 */
	void traceShadowRays(
			gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx) {

		::optix::prime::Model model = adapter->getScene();

		RTPquery query;
		rtpQueryCreate(model->getRTPmodel(), RTP_QUERY_TYPE_CLOSEST, &query);

		cudaPrepOptixRays(cudaGvtCtx.traceRays, NULL, cudaGvtCtx.shadowRayCount,
				cudaGvtCtx.shadowRays, &cudaGvtCtx, true, cudaGvtCtx.stream);

		RTPbufferdesc desc;
		rtpBufferDescCreate(
				OptixContext::singleton()->context()->getRTPcontext(),
				RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
				RTP_BUFFER_TYPE_CUDA_LINEAR, cudaGvtCtx.traceRays, &desc);

		rtpBufferDescSetRange(desc, 0, cudaGvtCtx.shadowRayCount);
		rtpQuerySetRays(query, desc);

		RTPbufferdesc desc2;
		rtpBufferDescCreate(
				OptixContext::singleton()->context()->getRTPcontext(),
				RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR,
				cudaGvtCtx.traceHits, &desc2);

		rtpBufferDescSetRange(desc2, 0, cudaGvtCtx.shadowRayCount);
		rtpQuerySetHits(query, desc2);

		rtpQuerySetCudaStream(query, cudaGvtCtx.stream);

		rtpQueryExecute(query, RTP_QUERY_HINT_ASYNC);
		//rtpQueryFinish(query);

		cudaProcessShadows(&cudaGvtCtx);

	}

	void traceRays(
			gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx) {

		gpuErrchk(
				cudaMemsetAsync(cudaGvtCtx.traceRays, 0,
						sizeof(gvt::render::data::cuda_primitives::OptixRay)
								* cudaGvtCtx.rayCount, cudaGvtCtx.stream));

		gpuErrchk(
				cudaMemsetAsync(cudaGvtCtx.traceHits, 0,
						sizeof(gvt::render::data::cuda_primitives::OptixHit)
								* cudaGvtCtx.rayCount, cudaGvtCtx.stream));

		cudaPrepOptixRays(cudaGvtCtx.traceRays, cudaGvtCtx.valid,
				cudaGvtCtx.rayCount, cudaGvtCtx.rays, &cudaGvtCtx, false,
				cudaGvtCtx.stream);

		::optix::prime::Model model = adapter->getScene();
		RTPquery query;

		rtpQueryCreate(model->getRTPmodel(), RTP_QUERY_TYPE_CLOSEST, &query);

		RTPbufferdesc rays;
		rtpBufferDescCreate(
				OptixContext::singleton()->context()->getRTPcontext(),
				RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
				RTP_BUFFER_TYPE_CUDA_LINEAR, cudaGvtCtx.traceRays, &rays);

		rtpBufferDescSetRange(rays, 0, cudaGvtCtx.rayCount);

		rtpQuerySetRays(query, rays);

		RTPbufferdesc hits;
		rtpBufferDescCreate(
				OptixContext::singleton()->context()->getRTPcontext(),
				RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR,
				cudaGvtCtx.traceHits, &hits);
		rtpBufferDescSetRange(hits, 0, cudaGvtCtx.rayCount);

		rtpQuerySetHits(query, hits);

		rtpQuerySetCudaStream(query, cudaGvtCtx.stream);
		rtpQueryExecute(query, RTP_QUERY_HINT_ASYNC);
		//rtpQueryFinish(query);

	}

	void operator()() {
#ifdef GVT_USE_DEBUG
		boost::timer::auto_cpu_timer t_functor(
				"OptixMeshAdapter: thread trace time: %w\n");
#endif


		int thread = begin > 0 ? 1 : 0;

		int localEnd = (end - begin);
		size_t localDispatchSize = 0;
		localDispatch.reserve((end - begin) * 2);

		gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx =
				*(OptixContext::singleton()->_cudaGvtCtx[thread]);


		//Mesh instance specific data
		gpuErrchk(
				cudaMemcpyAsync(cudaGvtCtx.normi, &(normi[0]),
						sizeof(glm::mat3), cudaMemcpyHostToDevice,
						cudaGvtCtx.stream));
		gpuErrchk(
				cudaMemcpyAsync(cudaGvtCtx.minv, &(minv[0]), sizeof(glm::mat4),
						cudaMemcpyHostToDevice, cudaGvtCtx.stream));


		cudaGvtCtx.mesh = adapter->cudaMesh;
		cudaGvtCtx.dispatchCount = 0;

		for (size_t localIdx = 0; localIdx < localEnd; localIdx += packetSize) {

			const size_t localPacketSize =
					(localIdx + packetSize > localEnd) ?
							(localEnd - localIdx) : packetSize;

			gpuErrchk(
					cudaMemsetAsync(cudaGvtCtx.valid, 1, sizeof(bool) * packetSize, cudaGvtCtx.stream));

			cudaGvtCtx.rayCount = localPacketSize;
			gvt::render::actor::RayVector::iterator localRayList = rayList
					+ begin + localIdx;

	
			cudaSetRays(localRayList, localPacketSize, cudaGvtCtx.rays,
					localIdx, cudaGvtCtx.stream, cudaRaysBuff);


			cudaGvtCtx.validRayLeft = true;
			while (cudaGvtCtx.validRayLeft) {

				cudaGvtCtx.validRayLeft = false;
				cudaGvtCtx.shadowRayCount = 0;

				traceRays(cudaGvtCtx);

				shade(&cudaGvtCtx);

				gpuErrchk(cudaStreamSynchronize(cudaGvtCtx.stream)); //get shadow ray count

				traceShadowRays(cudaGvtCtx);

				gpuErrchk(cudaStreamSynchronize(cudaGvtCtx.stream)); //get validRayLeft

//				if (cudaGvtCtx.validRayLeft)
//					printf("Valid Rays left..\n");

			}


			cudaGetRays(localDispatchSize, cudaGvtCtx, disp_Buff, localDispatch,
					rayList);

		}

		// copy localDispatch rays to outgoing rays queue
		std::unique_lock < std::mutex > moved(adapter->_outqueue);

		for (int i = 0; i < localDispatchSize; i++) {
			moved_rays.push_back(localDispatch[i]);
		}

		moved.unlock();


	}
};

void OptixMeshAdapter::trace(gvt::render::actor::RayVector &rayList,
		gvt::render::actor::RayVector &moved_rays, glm::mat4 *m,
        glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
        size_t _begin, size_t _end) {
#ifdef GVT_USE_DEBUG
	boost::timer::auto_cpu_timer t_functor("OptixMeshAdapter: trace time: %w\n");
#endif

	if (_end == 0)
		_end = rayList.size();

	this->begin = _begin;
	this->end = _end;

	size_t localWork = end-begin;


	*m_pinned = *m;
	m = m_pinned;


	*minv_pinned = *minv;
	minv = minv_pinned;

	*normi_pinned = *normi;
	normi = normi_pinned;

	// pull out instance transform data
	GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: getting instance transform data");

	gpuErrchk(cudaDeviceSynchronize());

	tbb::task_group _tasks;
	bool parallel = true;

	_tasks.run(
			[&]() {
				gvt::render::actor::RayVector::iterator localRayList = rayList.begin()+ _begin;
				size_t begin=0;
				size_t end=(parallel && localWork >= 2* packetSize) ? (localWork/2) : localWork;

				OptixParallelTrace(this, localRayList, moved_rays,  m,
						minv, normi, counter, begin, end, disp_Buff[0], cudaRaysBuff[0])();

			});

	if (parallel && localWork >= 2 * packetSize) {

		_tasks.run(
				[&]() {
					gvt::render::actor::RayVector::iterator localRayList = rayList.begin() + _begin ;
					size_t begin= localWork / 2;
					size_t end=localWork;

					OptixParallelTrace(this, localRayList, moved_rays, m,
							minv, normi, counter, begin, end, disp_Buff[1], cudaRaysBuff[1])();

				});
	}

	_tasks.wait();

	GVT_DEBUG(DBG_ALWAYS,
			"OptixMeshAdapter: Forwarding rays: " << moved_rays.size());

}
