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

#include <gvt/core/schedule/TaskScheduling.h> // used for threads

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Mesh.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

#include <atomic>
#include <thread>
#include <future>

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>
/*
 #include <boost/tuple/tuple.hpp>
 #include <boost/container/vector.hpp>
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>
#include <optix.h>
#include <optix_cuda_interop.h>
#include "optix_prime/optix_prime.h"

#include "OptixMeshAdapter.cuh"

#include <tbb/task_group.h>

#include <float.h>

//boost::timer::cpu_timer shading;
//boost::timer::cpu_timer copydata;
//boost::timer::cpu_timer convertingRaysToCUDA;
//boost::timer::cpu_timer convertingRaysFromCUDA;
//boost::timer::cpu_timer addToMoved;

__inline__ void cudaRayToGvtRay(
		const gvt::render::data::cuda_primitives::Ray& cudaRay,
		gvt::render::actor::Ray& gvtRay) {

	memcpy(&(gvtRay.origin[0]), &(cudaRay.origin.x), sizeof(float) * 4);
	memcpy(&(gvtRay.direction[0]), &(cudaRay.direction.x), sizeof(float) * 4);
//	memcpy(&(gvtRay.inverseDirection[0]), &(cudaRay.inverseDirection.x),
//			sizeof(float) * 4);
	memcpy(&(gvtRay.color.rgba[0]), &(cudaRay.color.rgba[0]),
			sizeof(float) * 4);
	gvtRay.color.t = cudaRay.color.t;
	//gvtRay.id = cudaRay.id;
	gvtRay.depth = cudaRay.depth;
	gvtRay.w = cudaRay.w;
	gvtRay.t = cudaRay.t;
	gvtRay.t_min = cudaRay.t_min;
	gvtRay.t_max = cudaRay.t_max;
	gvtRay.type = cudaRay.type;

	//same performance but unreliable due to potencial unpredictable stuct field order
	//memcpy(&(gvtRay.data), &(cudaRay.data), 96);
}

__inline__ void gvtRayToCudaRay(const gvt::render::actor::Ray& gvtRay,
		gvt::render::data::cuda_primitives::Ray& cudaRay) {

	memcpy(&(cudaRay.origin.x), &(gvtRay.origin[0]), sizeof(float) * 4);
	memcpy(&(cudaRay.direction.x), &(gvtRay.direction[0]), sizeof(float) * 4);
//	memcpy(&(cudaRay.inverseDirection.x), &(gvtRay.inverseDirection[0]),
//			sizeof(float) * 4);
	memcpy(&(cudaRay.color.rgba[0]), &(gvtRay.color.rgba[0]),
			sizeof(float) * 4);
	cudaRay.color.t = gvtRay.color.t;
	//cudaRay.id = gvtRay.id;
	cudaRay.depth = gvtRay.depth;
	cudaRay.w = gvtRay.w;
	cudaRay.t = gvtRay.t;
	cudaRay.t_min = gvtRay.t_min;
	cudaRay.t_max = gvtRay.t_max;
	cudaRay.type = gvtRay.type;

	//same performance but unreliable due to potencial unpredictable stuct field order
	//memcpy(&(cudaRay.data), &(gvtRay.data), 96);
}

int3*
cudaCreateFacesToNormals(
		boost::container::vector<boost::tuple<int, int, int> > gvt_face_to_normals) {

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
cudaCreateNormals(
		boost::container::vector<gvt::core::math::Point4f> gvt_normals) {

	float4* normalsBuff;

	gpuErrchk(
			cudaMalloc((void ** ) &normalsBuff,
					sizeof(float4) * gvt_normals.size()));

	std::vector<float4> normals;
	for (int i = 0; i < gvt_normals.size(); i++) {

		float4 v = make_float4(gvt_normals[i].x, gvt_normals[i].y,
				gvt_normals[i].z, gvt_normals[i].w);
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
		boost::container::vector<boost::tuple<int, int, int> > gvt_faces) {

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

gvt::render::data::cuda_primitives::Material *
cudaCreateMaterial(gvt::render::data::primitives::Material *gvtMat) {

	gvt::render::data::cuda_primitives::Material cudaMat;
	gvt::render::data::cuda_primitives::Material *cudaMat_ptr;

	gpuErrchk(
			cudaMalloc((void ** ) &cudaMat_ptr,
					sizeof(gvt::render::data::cuda_primitives::Material)));

	if (dynamic_cast<gvt::render::data::primitives::Lambert *>(gvtMat) != NULL) {

		gvtMat->pack((unsigned char *) &(cudaMat.lambert.kd));

		cudaMat.type = gvt::render::data::cuda_primitives::LAMBERT;

	} else if (dynamic_cast<gvt::render::data::primitives::Phong *>(gvtMat) !=
	NULL) {

		unsigned char *buff =
				new unsigned char[sizeof(gvt::core::math::Vector4f) * 2
						+ sizeof(float)];

		gvtMat->pack(buff);

		cudaMat.phong.kd = *(float4 *) buff;
		cudaMat.phong.ks = *(float4 *) (buff + 16);
		cudaMat.phong.alpha = *(float *) (buff + 32);

		delete[] buff;
		cudaMat.type = gvt::render::data::cuda_primitives::PHONG;

	} else if (dynamic_cast<gvt::render::data::primitives::BlinnPhong *>(gvtMat)
			!= NULL) {

		unsigned char *buff =
				new unsigned char[sizeof(gvt::core::math::Vector4f) * 2
						+ sizeof(float)];

		gvtMat->pack(buff);

		cudaMat.blinn.kd = *(float4 *) buff;
		cudaMat.blinn.ks = *(float4 *) (buff + 16);
		cudaMat.blinn.alpha = *(float *) (buff + 32);

		delete[] buff;
		cudaMat.type = gvt::render::data::cuda_primitives::BLINN;

	} else {
		std::cout << "Unknown material" << std::endl;
		return NULL;
	}

	gpuErrchk(
			cudaMemcpy(cudaMat_ptr, &cudaMat,
					sizeof(gvt::render::data::cuda_primitives::Material),
					cudaMemcpyHostToDevice));

	return cudaMat_ptr;
}

void cudaSetRays(gvt::render::actor::RayVector::iterator gvtRayVector,
		int localRayCount,
		gvt::render::data::cuda_primitives::Ray *cudaRays_devPtr, int localIdx,
		cudaStream_t& stream,
		gvt::render::data::cuda_primitives::Ray* cudaRays) {

	//convertingRaysToCUDA.resume();

	const int offset_rays =
			localRayCount > std::thread::hardware_concurrency() ?
					localRayCount / std::thread::hardware_concurrency() : 100;

	std::vector<std::future<void>> _tasks;

	// clang-format off
//	for (int i = 0; i < localRayCount; i += offset_rays) {
//		_tasks.push_back(
//				std::async(std::launch::async,
//						[&](const int ii, const int end) {
	const int ii = 0;
	const int end = localRayCount;
	for (int jj = ii; jj < end; jj++) {
		//if (jj < localRayCount) {

		//gvt::render::data::cuda_primitives::Ray r;

		gvtRayToCudaRay(gvtRayVector[jj], cudaRays[jj]);
		cudaRays[jj].mapToHostBufferID = localIdx + jj;
		//cudaRays[jj]=r;
		//}

	}

//						}, i, i + offset_rays));
//	}
//
//	for (auto &f : _tasks)
//		f.wait();

	//convertingRaysToCUDA.stop();

	//copydata.resume();

	gpuErrchk(
			cudaMemcpyAsync(cudaRays_devPtr, &cudaRays[0],
					sizeof(gvt::render::data::cuda_primitives::Ray)
							* localRayCount, cudaMemcpyHostToDevice, stream));

	//copydata.stop();

}

void cudaGetRays(size_t& localDispatchSize,
		gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx,
		gvt::render::data::cuda_primitives::Ray* disp_tmp,
		gvt::render::actor::RayVector& localDispatch,
		gvt::render::actor::RayVector::iterator &rayList) {

	//copydata.resume();

	gpuErrchk(
			cudaMemcpyAsync(&disp_tmp[0], cudaGvtCtx.dispatch,
					sizeof(gvt::render::data::cuda_primitives::Ray)
							* cudaGvtCtx.dispatchCount, cudaMemcpyDeviceToHost,
					cudaGvtCtx.stream));

	gpuErrchk(cudaStreamSynchronize(cudaGvtCtx.stream));

	//copydata.stop();

	//convertingRaysFromCUDA.resume();

	const int offset_rays =
			cudaGvtCtx.dispatchCount > std::thread::hardware_concurrency() ?
					cudaGvtCtx.dispatchCount
							/ std::thread::hardware_concurrency() :
					100;

	std::vector<std::future<void>> _tasks;

	for (int i = 0; i < cudaGvtCtx.dispatchCount; i += offset_rays) {
		_tasks.push_back(
				std::async(std::launch::async,
						[&](const int ii, const int end) {

							for (int jj = ii; jj < end; jj++) {
								if (jj < cudaGvtCtx.dispatchCount) {
									gvt::render::actor::Ray& gvtRay =
									localDispatch[localDispatchSize + jj];
									const gvt::render::data::cuda_primitives::Ray& cudaRay =
									disp_tmp[jj];

									const gvt::render::actor::Ray& originalRay =
									rayList[cudaRay.mapToHostBufferID];

									cudaRayToGvtRay(cudaRay, gvtRay);

									gvtRay.setDirection(gvtRay.direction);
									gvtRay.id = originalRay.id;

									if (gvtRay.type
											!= gvt::render::data::cuda_primitives::Ray::SHADOW
											&& originalRay.domains.size() != 0) {

										gvtRay.domains = originalRay.domains;
									} else {
										gvtRay.domains = *(new std::vector<gvt::render::actor::isecDom>());
										gvtRay.domains.clear();
									}

									//avoid contructor
									//localDispatch.push_back(gvtR);
								}
							}

						}, i, i + offset_rays));
	}

	for (auto &f : _tasks)
		f.wait();

	localDispatchSize += cudaGvtCtx.dispatchCount;
	cudaGvtCtx.dispatchCount = 0;

	//convertingRaysFromCUDA.stop();

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

	return cudaMesh;
}

void gvt::render::data::cuda_primitives::CudaGvtContext::initCudaBuffers(
		int packetSize) {

	printf("Initializing CUDA buffers...\n");

	gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

	auto lightNodes = root["Lights"].getChildren();
	std::vector<gvt::render::data::scene::Light *> lights;
	lights.reserve(2);
	for (auto lightNode : lightNodes) {
		auto color = lightNode["color"].value().toVector4f();

		if (lightNode.name() == std::string("PointLight")) {
			auto pos = lightNode["position"].value().toVector4f();
			lights.push_back(
					new gvt::render::data::scene::PointLight(pos, color));
		} else if (lightNode.name() == std::string("AmbientLight")) {
			lights.push_back(new gvt::render::data::scene::AmbientLight(color));
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
using namespace gvt::core::math;

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

OptixContext *OptixContext::_singleton;					//= new OptixContext();

OptixMeshAdapter::OptixMeshAdapter(gvt::core::DBNodeH node) :
		Adapter(node), packetSize(GVT_OPTIX_PACKET_SIZE), optix_context_(
				OptixContext::singleton()->context()) {

	// Get GVT mesh pointer
	GVT_ASSERT(optix_context_.isValid(), "Optix Context is not valid");
	Mesh *mesh = (Mesh *) node["ptr"].value().toULongLong();
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

	cudaMallocHost(&(m_pinned),
			sizeof(gvt::core::math::AffineTransformMatrix<float>));

	cudaMallocHost(&(minv_pinned),
			sizeof(gvt::core::math::AffineTransformMatrix<float>));


	cudaMallocHost(&(normi_pinned), sizeof(gvt::core::math::Matrix3f));

	// Setup the buffer to hold our vertices.
	//
	std::vector<float> vertices;
	std::vector<int> faces;

	vertices.resize(numVerts * 3);
	faces.resize(numTris * 3);

	const int offset_verts =
			numVerts > std::thread::hardware_concurrency() ?
					numVerts / std::thread::hardware_concurrency() : 100;

	std::vector<std::future<void>> _tasks;

	// clang-format off
	for (int i = 0; i < numVerts; i += offset_verts) {
		_tasks.push_back(
				std::async(std::launch::async,
						[&](const int ii, const int end) {
							for (int jj = ii; jj < end && jj < numVerts; jj++) {
								vertices[jj * 3 + 0] = mesh->vertices[jj][0];
								vertices[jj * 3 + 1] = mesh->vertices[jj][1];
								vertices[jj * 3 + 2] = mesh->vertices[jj][2];
							}
						}, i, i + offset_verts));
	}
	// clang-format on

	const int offset_tris =
			numTris > std::thread::hardware_concurrency() ?
					numTris / std::thread::hardware_concurrency() : 100;

	// clang-format off
	for (int i = 0; i < numTris; i += offset_tris) {
		_tasks.push_back(
				std::async(std::launch::async,
						[&](const int ii, const int end) {
							for (int jj = ii; jj < end && jj < numTris; jj++) {
								gvt::render::data::primitives::Mesh::Face f = mesh->faces[jj];
								faces[jj * 3 + 0] = f.get<0>();
								faces[jj * 3 + 1] = f.get<1>();
								faces[jj * 3 + 2] = f.get<2>();
							}
						}, i, i + offset_tris));
	}
	// clang-format on

	for (auto &f : _tasks)
		f.wait();

	printf("Copying geometry to device...\n");

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
	/**
	 * Pointer to OptixMeshAdapter to get Embree scene information
	 */
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
	 * Number of rays to work on at once [load balancing].
	 */
	//const size_t workSize;
	/**
	 * Index into the shared `rayList`.  Atomically incremented to 'grab'
	 * the next set of rays.
	 */
	//std::atomic<size_t> &sharedIdx;
	/**
	 * DB reference to the current instance
	 */
	gvt::core::DBNodeH instNode;

	/**		f.get()
	 *
	 * Stored transformation matrix in the current instance
	 */
	const gvt::core::math::AffineTransformMatrix<float> *m;

	/**
	 * Stored inverse transformation matrix in the current instance
	 */
	const gvt::core::math::AffineTransformMatrix<float> *minv;

	/**
	 * Stored upper33 inverse matrix in the current instance
	 */
	const gvt::core::math::Matrix3f *normi;

	/**
	 * Stored transformation matrix in the current instance
	 */
	//const std::vector<gvt::render::data::scene::Light *> &lights;
	/**
	 * Count the number of rays processed by the current trace() call.
	 *
	 * Used for debugging purposes
	 */
	std::atomic<size_t> &counter;

	/**
	 * Thread local outgoing ray queue
	 */
	gvt::render::actor::RayVector localDispatch;

	gvt::render::data::cuda_primitives::Ray* disp_Buff;

	gvt::render::data::cuda_primitives::Ray* cudaRaysBuff;

	/**
	 * List of shadow rays to be processed
	 */
	//gvt::render::actor::RayVector shadowRays;
	/**
	 * Size of Embree packet
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
			//std::atomic<size_t> &sharedIdx, const size_t workSize,
			gvt::core::DBNodeH instNode,
			gvt::core::math::AffineTransformMatrix<float> *m,
			gvt::core::math::AffineTransformMatrix<float> *minv,
			gvt::core::math::Matrix3f *normi,
			//std::vector<gvt::render::data::scene::Light *> &lights,
			std::atomic<size_t> &counter, const size_t begin, const size_t end,
			gvt::render::data::cuda_primitives::Ray* disp_Buff,
			gvt::render::data::cuda_primitives::Ray* cudaRaysBuff) :
			adapter(adapter), rayList(rayList), moved_rays(moved_rays), instNode(
					instNode), m(m), minv(minv), normi(normi), counter(counter), packetSize(
					adapter->getPacketSize()), begin(begin), end(end), disp_Buff(
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

//		shading = boost::timer::cpu_timer();
//		shading.stop();
//		copydata = boost::timer::cpu_timer();
//		copydata.stop();
//		convertingRaysFromCUDA = boost::timer::cpu_timer();
//		convertingRaysFromCUDA.stop();
//		convertingRaysToCUDA = boost::timer::cpu_timer();
//		convertingRaysToCUDA.stop();
//		addToMoved = boost::timer::cpu_timer();
//		addToMoved.stop();

		int thread = begin > 0 ? 1 : 0;

		int localEnd = (end - begin);
		size_t localDispatchSize = 0;
		localDispatch.reserve((end - begin) * 2);

		//cudaMallocHost(&(disp_Buff), sizeof ( gvt::render::data::cuda_primitives::Ray)*packetSize * 2);
		//udaMallocHost(&(cudaRaysBuff), sizeof ( gvt::render::data::cuda_primitives::Ray)*packetSize);

		gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx =
				*(OptixContext::singleton()->_cudaGvtCtx[thread]);

		//copydata.resume();

		//Mesh instance specific data
		gpuErrchk(
				cudaMemcpyAsync(cudaGvtCtx.normi, &(normi->n[0]),
						sizeof(gvt::render::data::cuda_primitives::Matrix3f),
						cudaMemcpyHostToDevice, cudaGvtCtx.stream));
		gpuErrchk(
				cudaMemcpyAsync(cudaGvtCtx.minv, &(minv->n[0]),
						sizeof(gvt::render::data::cuda_primitives::Matrix4f),
						cudaMemcpyHostToDevice, cudaGvtCtx.stream));

		//copydata.stop();

		cudaGvtCtx.mesh = adapter->cudaMesh;
		cudaGvtCtx.dispatchCount = 0;

		for (size_t localIdx = 0; localIdx < localEnd; localIdx += packetSize) {

			const size_t localPacketSize =
					(localIdx + packetSize > localEnd) ?
							(localEnd - localIdx) : packetSize;

			printf(
					"[%d]: localPacketSize: %zu localIdx: %d packetSize: %zu raylistSize: %zu  workBegin: %zu\ workEnd: %zu\n",
					thread, localPacketSize, localIdx, packetSize,
					(end - begin), begin, end);

			gpuErrchk(
					cudaMemsetAsync(cudaGvtCtx.valid, 1, sizeof(bool) * packetSize, cudaGvtCtx.stream));

			cudaGvtCtx.rayCount = localPacketSize;
			gvt::render::actor::RayVector::iterator localRayList = rayList
					+ localIdx;

			cudaSetRays(localRayList, localPacketSize, cudaGvtCtx.rays,
					localIdx, cudaGvtCtx.stream, cudaRaysBuff);

			//shading.resume();

			cudaGvtCtx.validRayLeft = true;
			while (cudaGvtCtx.validRayLeft) {

				cudaGvtCtx.validRayLeft = false;
				cudaGvtCtx.shadowRayCount = 0;

				traceRays(cudaGvtCtx);

				shade(&cudaGvtCtx);

				gpuErrchk(cudaStreamSynchronize(cudaGvtCtx.stream)); //get shadow ray count

				traceShadowRays(cudaGvtCtx);

				gpuErrchk(cudaStreamSynchronize(cudaGvtCtx.stream)); //get validRayLeft

				if (cudaGvtCtx.validRayLeft)
					printf("Valid Rays left..\n");

			}

			//shading.stop();

			cudaGetRays(localDispatchSize, cudaGvtCtx, disp_Buff, localDispatch,
					rayList);

		}

		//addToMoved.resume();
		// copy localDispatch rays to outgoing rays queue
		boost::unique_lock<boost::mutex> moved(adapter->_outqueue);

		for (int i = 0; i < localDispatchSize; i++) {
			moved_rays.push_back(localDispatch[i]);
		}

		moved.unlock();
		//addToMoved.stop();

//		std::cout << "adapater optix-cuda: tracing-shading time: "
//				<< shading.format();
//		std::cout << "adapater optix-cuda: copy data to device time: "
//				<< copydata.format();
//		std::cout << "adapater optix-cuda: convertingRaysToCUDA time: "
//				<< convertingRaysToCUDA.format();
//		std::cout << "adapater optix-cuda: convertingRaysFromCUDA time: "
//				<< convertingRaysFromCUDA.format();
//		std::cout << "adapater optix-cuda: addToMoved time: "
//				<< addToMoved.format();

	}
};

void OptixMeshAdapter::trace(gvt::render::actor::RayVector &rayList,
		gvt::render::actor::RayVector &moved_rays, gvt::core::DBNodeH instNode,
		size_t _begin, size_t _end) {
#ifdef GVT_USE_DEBUG
	boost::timer::auto_cpu_timer t_functor("OptixMeshAdapter: trace time: %w\n");
#endif

	if (_end == 0)
		_end = rayList.size();

	this->begin = _begin;
	this->end = _end;

	gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

	gvt::core::math::AffineTransformMatrix<float> * m =
			(gvt::core::math::AffineTransformMatrix<float> *) instNode["mat"].value().toULongLong();


	*m_pinned = *m;
	m = m_pinned;

	gvt::core::math::AffineTransformMatrix<float> * minv =
			(gvt::core::math::AffineTransformMatrix<float> *) instNode["matInv"].value().toULongLong();

	*minv_pinned = *minv;
	minv = minv_pinned;

	gvt::core::math::Matrix3f * normi =
			(gvt::core::math::Matrix3f *) instNode["normi"].value().toULongLong();


	*normi_pinned = *normi;
	normi = normi_pinned;

	// pull out instance transform data
	GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: getting instance transform data");

	gpuErrchk(cudaDeviceSynchronize());

	tbb::task_group _tasks;
	bool parallel = true;

	_tasks.run(
			[&]() {
				gvt::render::actor::RayVector::iterator localRayList = rayList.begin();
				OptixParallelTrace(this, localRayList, moved_rays, instNode, m,
						minv, normi, counter, 0,
						(parallel && _end >= 2* packetSize) ? (_end/2) : _end, disp_Buff[0], cudaRaysBuff[0])();

			});

	if (parallel && _end >= 2 * packetSize) {

		_tasks.run(
				[&]() {
					gvt::render::actor::RayVector::iterator localRayList = rayList.begin() + (rayList.size() / 2);

					OptixParallelTrace(this, localRayList, moved_rays, instNode, m,
							minv, normi, counter, (_end/2), _end, disp_Buff[1], cudaRaysBuff[1])();

				});
	}

	_tasks.wait();

	GVT_DEBUG(DBG_ALWAYS,
			"OptixMeshAdapter: Forwarding rays: " << moved_rays.size());

}
