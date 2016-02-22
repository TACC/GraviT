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

#include <float.h>

boost::timer::cpu_timer shading;
boost::timer::cpu_timer copydata;
boost::timer::cpu_timer clearDispatch;
boost::timer::cpu_timer convertingRays;
boost::timer::cpu_timer convertingRaysDispatch;



void cudaRayToGvtRay(gvt::render::data::cuda_primitives::Ray& cudaRay,
		gvt::render::actor::Ray& gvtRay){


	memcpy(&(gvtRay.origin[0]), &(cudaRay.origin.x), sizeof(float)*4);
	memcpy(&(gvtRay.direction[0]), &(cudaRay.direction.x), sizeof(float)*4);
	memcpy(&(gvtRay.inverseDirection[0]), &(cudaRay.inverseDirection.x), sizeof(float)*4);
	memcpy(&(gvtRay.color.rgba[0]), &(cudaRay.color.rgba[0]), sizeof(float)*4);
	gvtRay.color.t = cudaRay.color.t;
	gvtRay.id=cudaRay.id;
	gvtRay.depth=cudaRay.depth;
	gvtRay.w=cudaRay.w;
	gvtRay.t=cudaRay.t;
	gvtRay.t_min=cudaRay.t_min;
	gvtRay.t_max=cudaRay.t_max;
	gvtRay.type=cudaRay.type;
}

void gvtRayToCudaRay(	gvt::render::actor::Ray& gvtRay,
		gvt::render::data::cuda_primitives::Ray& cudaRay){


	memcpy(&(cudaRay.origin.x), &(gvtRay.origin[0]), sizeof(float)*4);
	memcpy(&(cudaRay.direction.x), &(gvtRay.direction[0]), sizeof(float)*4);
	memcpy(&(cudaRay.inverseDirection.x), &(gvtRay.inverseDirection[0]), sizeof(float)*4);
	memcpy(&(cudaRay.color.rgba[0]), &(gvtRay.color.rgba[0]), sizeof(float)*4);
	cudaRay.color.t = gvtRay.color.t;
	cudaRay.id=gvtRay.id;
	cudaRay.depth=gvtRay.depth;
	cudaRay.w=gvtRay.w;
	cudaRay.t=gvtRay.t;
	cudaRay.t_min=gvtRay.t_min;
	cudaRay.t_max=gvtRay.t_max;
	cudaRay.type=gvtRay.type;
}

int3*
cudaCreateFacesToNormals(boost::container::vector<boost::tuple<int, int, int> >
						gvt_face_to_normals) {


	int3* faces_to_normalsBuff;

	cudaMalloc((void **)&faces_to_normalsBuff,
	             sizeof(int3)*gvt_face_to_normals.size());

	std::vector<int3> faces_to_normals;
	for (int i =0; i < gvt_face_to_normals.size(); i++){

		const boost::tuple<int, int, int> &f = gvt_face_to_normals[i];


		int3 v = make_int3(f.get<0>(),
			f.get<1>(),
			f.get<2>());

		faces_to_normals.push_back(v);
	}

	 cudaMemcpy(faces_to_normalsBuff, &faces_to_normals[0],
	             sizeof(int3)*faces_to_normals.size(),
	             cudaMemcpyHostToDevice);

	 return faces_to_normalsBuff;

}

float4*
cudaCreateNormals(boost::container::vector<gvt::core::math::Point4f>
						gvt_normals) {


	float4* normalsBuff;

	cudaMalloc((void **)&normalsBuff,
	             sizeof(float4)*gvt_normals.size());

	std::vector<float4> normals;
	for (int i =0; i < gvt_normals.size(); i++){

		float4 v =make_float4(gvt_normals[i].x,gvt_normals[i].y,gvt_normals[i].z,gvt_normals[i].w);
		normals.push_back(v);
	}

	 cudaMemcpy(normalsBuff, &normals[0],
	             sizeof(float4)*gvt_normals.size(),
	             cudaMemcpyHostToDevice);

	 return normalsBuff;

}

int3*
cudaCreateFaces(boost::container::vector<boost::tuple<int, int, int> >
						gvt_faces) {


	int3* facesBuff;

	cudaMalloc((void **)&facesBuff,
	             sizeof(int3)*gvt_faces.size());

	std::vector<int3> faces;
	for (int i =0; i < gvt_faces.size(); i++){

		const boost::tuple<int, int, int> &f = gvt_faces[i];


		int3 v = make_int3(f.get<0>(),
			f.get<1>(),
			f.get<2>());
		faces.push_back(v);
	}

	 cudaMemcpy(facesBuff, &faces[0],
	             sizeof(int3)*gvt_faces.size(),
	             cudaMemcpyHostToDevice);

	 return facesBuff;

}

gvt::render::data::cuda_primitives::Material *
cudaCreateMaterial(gvt::render::data::primitives::Material *gvtMat) {

  gvt::render::data::cuda_primitives::Material cudaMat;
  gvt::render::data::cuda_primitives::Material *cudaMat_ptr;

  cudaMalloc((void **)&cudaMat_ptr,
             sizeof(gvt::render::data::cuda_primitives::Material));

  if (dynamic_cast<gvt::render::data::primitives::Lambert *>(gvtMat) != NULL) {

    gvtMat->pack((unsigned char *)&(cudaMat.lambert.kd));

    cudaMat.type = gvt::render::data::cuda_primitives::LAMBERT;

  } else if (dynamic_cast<gvt::render::data::primitives::Phong *>(gvtMat) !=
             NULL) {

    unsigned char *buff =
        new unsigned char[sizeof(gvt::core::math::Vector4f) * 2 +
                          sizeof(float)];

    gvtMat->pack(buff);

    cudaMat.phong.kd = *(float4 *)buff;
    cudaMat.phong.ks = *(float4 *)(buff + 16);
    cudaMat.phong.alpha = *(float *)(buff + 32);

    delete[] buff;
    cudaMat.type = gvt::render::data::cuda_primitives::PHONG;

  } else if (dynamic_cast<gvt::render::data::primitives::BlinnPhong *>(
                 gvtMat) != NULL) {

    unsigned char *buff =
        new unsigned char[sizeof(gvt::core::math::Vector4f) * 2 +
                          sizeof(float)];

    gvtMat->pack(buff);

    cudaMat.blinn.kd = *(float4 *)buff;
    cudaMat.blinn.ks = *(float4 *)(buff + 16);
    cudaMat.blinn.alpha = *(float *)(buff + 32);

    delete[] buff;
    cudaMat.type = gvt::render::data::cuda_primitives::BLINN;

  } else {
    std::cout << "Unknown material" << std::endl;
    return NULL;
  }

  cudaMemcpy(cudaMat_ptr, &cudaMat,
             sizeof(gvt::render::data::cuda_primitives::Material),
             cudaMemcpyHostToDevice);

  return cudaMat_ptr;
}

void
cudaGetRays(gvt::render::actor::RayVector::iterator gvtRayVector, int localRayCount,
		  gvt::render::data::cuda_primitives::Ray *cudaRays_devPtr,
		  int localIdx) {



	convertingRays.resume();
  std::vector< gvt::render::data::cuda_primitives::Ray> cudaRays;

  for (int i = 0; i < localRayCount; i++) {
	  gvt::render::data::cuda_primitives::Ray r;


	  gvtRayToCudaRay(gvtRayVector[i],r);
	  r.mapToHostBufferID = localIdx+i;
    cudaRays.push_back(r);

  }

  convertingRays.stop();

  copydata.resume();
 gpuErrchk(cudaMemcpy(cudaRays_devPtr, &cudaRays[0],
             sizeof(gvt::render::data::cuda_primitives::Ray) *
             localRayCount,
             cudaMemcpyHostToDevice));

  copydata.stop();


}

void
cudaGetLights(std::vector<gvt::render::data::scene::Light *> gvtLights,
		gvt::render::data::cuda_primitives::Light * cudaLights_devPtr) {


  gvt::render::data::cuda_primitives::Light *cudaLights =
      new gvt::render::data::cuda_primitives::Light[gvtLights.size()];



  for (int i = 0; i < gvtLights.size(); i++) {

    if (dynamic_cast<gvt::render::data::scene::AmbientLight *>(gvtLights[i]) !=
        NULL) {

      gvt::render::data::scene::AmbientLight *l =
          dynamic_cast<gvt::render::data::scene::AmbientLight *>(gvtLights[i]);

      memcpy(&(cudaLights[i].ambient.position.x), &(l->position.x),
             sizeof(float4));
      memcpy(&(cudaLights[i].ambient.color.x), &(l->color.x), sizeof(float4));

      cudaLights[i].type =
          gvt::render::data::cuda_primitives::LIGH_TYPE::AMBIENT;

    } else if (dynamic_cast<gvt::render::data::scene::PointLight *>(
                   gvtLights[i]) != NULL) {

      gvt::render::data::scene::PointLight *l =
          dynamic_cast<gvt::render::data::scene::PointLight *>(gvtLights[i]);

      memcpy(&(cudaLights[i].point.position.x), &(l->position.x),
             sizeof(float4));
      memcpy(&(cudaLights[i].point.color.x), &(l->color.x), sizeof(float4));

      cudaLights[i].type = gvt::render::data::cuda_primitives::LIGH_TYPE::POINT;

    } else {
      std::cout << "Unknown light" << std::endl;
      return ;
    }
  }

  cudaMemcpy(cudaLights_devPtr, cudaLights,
             sizeof(gvt::render::data::cuda_primitives::Light) *
                 gvtLights.size(),
             cudaMemcpyHostToDevice);

  delete[] cudaLights;

}

gvt::render::data::cuda_primitives::Mesh
cudaInstanceMesh(gvt::render::data::primitives::Mesh* mesh){

	gvt::render::data::cuda_primitives::Mesh cudaMesh;

	cudaMesh.faces_to_normals = cudaCreateFacesToNormals(mesh->faces_to_normals);
	cudaMesh.normals = cudaCreateNormals(mesh->normals);
	cudaMesh.faces = cudaCreateFaces(mesh->faces);
	cudaMesh.mat = cudaCreateMaterial(mesh->mat);

	return cudaMesh;
}


void  gvt::render::data::cuda_primitives::CudaGvtContext::initCudaBuffers(int packetSize){

printf("Initializing CUDA buffers...\n");

	 gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();



	  auto lightNodes = root["Lights"].getChildren();
	  std::vector<gvt::render::data::scene::Light *> lights;
	  lights.reserve(2);
	  for (auto lightNode : lightNodes) {
	    auto color = lightNode["color"].value().toVector4f();

	    if (lightNode.name() == std::string("PointLight")) {
	      auto pos = lightNode["position"].value().toVector4f();
	      lights.push_back(new gvt::render::data::scene::PointLight(pos, color));
	    } else if (lightNode.name() == std::string("AmbientLight")) {
	      lights.push_back(new gvt::render::data::scene::AmbientLight(color));
	    }
	  }

      gvt::render::data::cuda_primitives::Light *cudaLights_devPtr;
      gpuErrchk(cudaMalloc((void **)&cudaLights_devPtr,
	                  sizeof(gvt::render::data::cuda_primitives::Light) *
	                  lights.size()));

	   cudaGetLights(lights, cudaLights_devPtr);




	   gvt::render::data::cuda_primitives::Matrix3f *normiBuff;
	   gpuErrchk(cudaMalloc((void **) &normiBuff,
	   				sizeof(gvt::render::data::cuda_primitives::Matrix3f)));

	   		gvt::render::data::cuda_primitives::Matrix4f *minvBuff;
	   		gpuErrchk(cudaMalloc((void **) &minvBuff,
	   				sizeof(gvt::render::data::cuda_primitives::Matrix4f)));




       gvt::render::data::cuda_primitives::OptixRay *cudaOptixRayBuff;
       gpuErrchk(cudaMalloc((void **) &cudaOptixRayBuff,
			sizeof(gvt::render::data::cuda_primitives::OptixRay) * packetSize
					* lights.size()));

	gvt::render::data::cuda_primitives::OptixHit *cudaHitsBuff;
	gpuErrchk(cudaMalloc((void **) &cudaHitsBuff,
			sizeof(gvt::render::data::cuda_primitives::OptixHit) * packetSize
					* lights.size()));

	gvt::render::data::cuda_primitives::Ray *shadowRaysBuff;
	gpuErrchk(cudaMalloc((void **) &shadowRaysBuff,
			sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize
					* lights.size()));

	bool *validBuff;
	gpuErrchk(cudaMalloc((void **) &validBuff, sizeof(bool) * packetSize));

	// Size is rayCount but depends on instance, so majoring to double packetsize
	gvt::render::data::cuda_primitives::Ray * dispatchBuff;
	gpuErrchk(cudaMalloc((void **) &dispatchBuff,
			sizeof(gvt::render::data::cuda_primitives::Ray) *
	/*(end-begin)*/packetSize * 2));

	gvt::render::data::cuda_primitives::Ray *cudaRays_devPtr;
	gpuErrchk(cudaMalloc((void **) &cudaRays_devPtr,
			sizeof(gvt::render::data::cuda_primitives::Ray) * packetSize));

	set_random_states(packetSize);

	gpuErrchk(cudaMalloc(&devicePtr,
				sizeof(gvt::render::data::cuda_primitives::CudaGvtContext)));

		rays=cudaRays_devPtr;
 		 traceRays = cudaOptixRayBuff;
 		 traceHits = cudaHitsBuff;
 		 this->lights = cudaLights_devPtr;
 		shadowRays = shadowRaysBuff;
 		 nLights = lights.size();
 		valid = validBuff;
 		 dispatch = dispatchBuff;
 		 normi=normiBuff;
 		 minv=minvBuff;


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

OptixContext *OptixContext::_singleton;//= new OptixContext();

OptixMeshAdapter::OptixMeshAdapter(gvt::core::DBNodeH node)
    : Adapter(node), packetSize(GVT_OPTIX_PACKET_SIZE),
      optix_context_(OptixContext::singleton()->context()) {

  // Get GVT mesh pointer
  GVT_ASSERT(optix_context_.isValid(), "Optix Context is not valid");
  Mesh *mesh = (Mesh *)node["ptr"].value().toULongLong();
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
    cudaGetDeviceCount(&devCount);
    GVT_ASSERT(
        devCount,
        "You choose optix render, but no cuda capable devices are present");

    for (int i = 0; i < devCount; i++) {
      cudaGetDeviceProperties(&prop, i);
      if (prop.kernelExecTimeoutEnabled == 0)
        activeDevices.push_back(i);
      // Oversubcribe the GPU
      packetSize = prop.multiProcessorCount *
    		  prop.maxThreadsPerMultiProcessor;
    }
    if (!activeDevices.size()) {
      activeDevices.push_back(0);
    }
    optix_context_->setCudaDeviceNumbers(activeDevices);
  }



	OptixContext::singleton()->initCuda(packetSize);
	cudaMesh = cudaInstanceMesh(mesh);

  // Setup the buffer to hold our vertices.
  //
  std::vector<float> vertices;
  std::vector<int> faces;

  vertices.resize(numVerts * 3);
  faces.resize(numTris * 3);

  const int offset_verts =
      100; // numVerts / std::thread::hardware_concurrency();

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

  const int offset_tris = 100; // numTris / std::thread::hardware_concurrency();

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
      RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, &vertices[0]);

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

OptixMeshAdapter::~OptixMeshAdapter() {}

struct OptixParallelTrace {
  /**
   * Pointer to OptixMeshAdapter to get Embree scene information
   */
  gvt::render::adapter::optix::data::OptixMeshAdapter *adapter;

  /**
   * Shared ray list used in the current trace() call
   */
  gvt::render::actor::RayVector &rayList;

  /**
   * Shared outgoing ray list used in the current trace() call
   */
  gvt::render::actor::RayVector &moved_rays;

  /**
   * Number of rays to work on at once [load balancing].
   */
  const size_t workSize;

  /**
   * Index into the shared `rayList`.  Atomically incremented to 'grab'
   * the next set of rays.
   */
  std::atomic<size_t> &sharedIdx;

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
  const std::vector<gvt::render::data::scene::Light *> &lights;

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

  /**
   * List of shadow rays to be processed
   */
  gvt::render::actor::RayVector shadowRays;

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
      gvt::render::actor::RayVector &rayList,
      gvt::render::actor::RayVector &moved_rays, std::atomic<size_t> &sharedIdx,
      const size_t workSize, gvt::core::DBNodeH instNode,
      gvt::core::math::AffineTransformMatrix<float> *m,
      gvt::core::math::AffineTransformMatrix<float> *minv,
      gvt::core::math::Matrix3f *normi,
      std::vector<gvt::render::data::scene::Light *> &lights,
      std::atomic<size_t> &counter, const size_t begin, const size_t end)
      : adapter(adapter), rayList(rayList), moved_rays(moved_rays),
        sharedIdx(sharedIdx), workSize(workSize), instNode(instNode), m(m),
        minv(minv), normi(normi), lights(lights), counter(counter),
        packetSize(adapter->getPacketSize()), begin(begin), end(end) {
  }

  /**
   * Convert a set of rays from a vector into a prepOptixRays ray packet.
   *
   * \param optixrays     reference of optix ray datastructure
   * \param valid         aligned array of 4 ints to mark valid rays
   * \param resetValid    if true, reset the valid bits, if false, re-use old
   * valid to know which to convert
   * \param packetSize    number of rays to convert
   * \param rays          vector of rays to read from
   * \param startIdx      starting point to read from in `rays`
   */
  void prepOptixRays(std::vector<OptixRay> &optixrays, std::vector<bool> &valid,
                     const bool resetValid, const int localPacketSize,
                     const gvt::render::actor::RayVector &rays,
                     const size_t startIdx) {
    for (int i = 0; i < localPacketSize; i++) {
      if (valid[i]) {
        const Ray &r = rays[startIdx + i];
        const auto origin = (*minv) * r.origin; // transform ray to local space
        const auto direction = (*minv) * r.direction;
        OptixRay optix_ray;
        optix_ray.origin[0] = origin[0];
        optix_ray.origin[1] = origin[1];
        optix_ray.origin[2] = origin[2];
        optix_ray.t_min = 0;
        optix_ray.direction[0] = direction[0];
        optix_ray.direction[1] = direction[1];
        optix_ray.direction[2] = direction[2];
        optix_ray.t_max = FLT_MAX;
        optixrays[i] = optix_ray;
      }
    }
  }

  /**
   * Test occlusion for stored shadow rays.  Add missed rays
   * to the dispatch queue.
   */
  void traceShadowRays(  gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx) {

	    ::optix::prime::Model model = adapter->getScene();

    RTPquery query ;
    rtpQueryCreate(model->getRTPmodel(), RTP_QUERY_TYPE_CLOSEST, &query) ;

      cudaPrepOptixRays( cudaGvtCtx.traceRays,  NULL,
    		  cudaGvtCtx.shadowRayCount, cudaGvtCtx.shadowRays,
                                &cudaGvtCtx, true);

      RTPbufferdesc desc;
      rtpBufferDescCreate(
          OptixContext::singleton()->context()->getRTPcontext(),
          RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
          RTP_BUFFER_TYPE_CUDA_LINEAR, cudaGvtCtx.traceRays, &desc);

      rtpBufferDescSetRange(desc, 0, cudaGvtCtx.shadowRayCount);
      rtpQuerySetRays(query, desc) ;


      RTPbufferdesc desc2;
          rtpBufferDescCreate(
              OptixContext::singleton()->context()->getRTPcontext(),
              RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V,
              RTP_BUFFER_TYPE_CUDA_LINEAR, cudaGvtCtx.traceHits, &desc2);


         rtpBufferDescSetRange(desc2, 0, cudaGvtCtx.shadowRayCount);
         rtpQuerySetHits(query, desc2) ;


      // Execute our query and wait for it to finish.
         rtpQueryExecute(query,RTP_QUERY_HINT_ASYNC);
         rtpQueryFinish(query);

         cudaProcessShadows(&cudaGvtCtx);

 }

  void traceRays(
	gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx) {

		cudaMemset(cudaGvtCtx.traceRays, 0,
		sizeof(gvt::render::data::cuda_primitives::OptixRay)
		*cudaGvtCtx.rayCount);

		cudaMemset( cudaGvtCtx.traceHits, 0,
		sizeof(gvt::render::data::cuda_primitives::OptixHit)
		*cudaGvtCtx.rayCount);

		cudaPrepOptixRays( cudaGvtCtx.traceRays, cudaGvtCtx.valid,
				cudaGvtCtx.rayCount, cudaGvtCtx.rays,
				 &cudaGvtCtx, false);

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
		rtpBufferDescSetRange(hits, 0,cudaGvtCtx.rayCount);

		rtpQuerySetHits(query, hits);

		rtpQueryExecute(query,RTP_QUERY_HINT_ASYNC);
		rtpQueryFinish(query);

	}

void operator()() {
#ifdef GVT_USE_DEBUG
		boost::timer::auto_cpu_timer t_functor(
				"OptixMeshAdapter: thread trace time: %w\n");
#endif


		  shading = boost::timer::cpu_timer();
		  copydata =  boost::timer::cpu_timer();
		  clearDispatch = boost::timer::cpu_timer();
		  convertingRays =boost::timer::cpu_timer();
		  convertingRaysDispatch =boost::timer::cpu_timer();

		localDispatch.reserve((end - begin) * 2);
		gvt::render::data::cuda_primitives::Ray *disp_tmp =
				new gvt::render::data::cuda_primitives::Ray[packetSize * 2];

		gvt::render::data::cuda_primitives::CudaGvtContext& cudaGvtCtx =
				*(OptixContext::singleton()->_cudaGvtCtx);

		copydata.start();

		//Mesh instance specific data
		cudaMemcpy(cudaGvtCtx.normi, &(normi->n[0]),
				sizeof(gvt::render::data::cuda_primitives::Matrix3f),
				cudaMemcpyHostToDevice);
		cudaMemcpy(cudaGvtCtx.minv, &(minv->n[0]),
				sizeof(gvt::render::data::cuda_primitives::Matrix4f),
				cudaMemcpyHostToDevice);

		cudaGvtCtx.mesh = adapter->cudaMesh;

		copydata.stop();

		for (size_t localIdx = 0; localIdx < end; localIdx += packetSize) {

			const size_t localPacketSize =
					(localIdx + packetSize > end) ?
							(end - localIdx) : packetSize;

			printf(
					"localPacketSize: %zu localIdx: %d packetSize: %zu raylistSize: %zu workEnd: %zu\n",
					localPacketSize, localIdx, packetSize, rayList.size(), end);


			copydata.resume();
			cudaMemset(cudaGvtCtx.valid, 1, sizeof(bool) * packetSize);

			cudaGvtCtx.rayCount = localPacketSize;
			gvt::render::actor::RayVector::iterator localRayList =
					rayList.begin() + localIdx;

			copydata.stop();

			cudaGetRays(localRayList, localPacketSize, cudaGvtCtx.rays,
					localIdx);


		    shading.resume();


			cudaGvtCtx.validRayLeft = true;
			while (cudaGvtCtx.validRayLeft) {

				cudaGvtCtx.validRayLeft = false;

				cudaGvtCtx.dispatchCount = 0;
				cudaGvtCtx.shadowRayCount = 0;

				traceRays(cudaGvtCtx);
				shade(&cudaGvtCtx);
				traceShadowRays(cudaGvtCtx);

				if (cudaGvtCtx.validRayLeft)
					printf("Valid Rays left..\n");

			}

		    shading.stop();

		    clearDispatch.resume();
			//Clear dispatch
			{



				copydata.resume();

				//int* disp = new int[cudaGvtCtx.dispatchCount];
				cudaMemcpy(&disp_tmp[0], cudaGvtCtx.dispatch,
						sizeof(gvt::render::data::cuda_primitives::Ray)
								* cudaGvtCtx.dispatchCount,
						cudaMemcpyDeviceToHost);
				copydata.stop();

				convertingRaysDispatch.resume();
				for (int i = 0; i < cudaGvtCtx.dispatchCount; i++) {

					//
					gvt::render::data::cuda_primitives::Ray& r = disp_tmp[i];
					Ray gvtRay;
					cudaRayToGvtRay(r, gvtRay);

					if (r.type
							!= gvt::render::data::cuda_primitives::Ray::SHADOW) {

						gvtRay.domains =
								rayList[r.mapToHostBufferID].domains;
					}

					localDispatch.push_back(gvtRay);

				}
				convertingRaysDispatch.stop();

			}
		    clearDispatch.stop();

		}

#ifdef GVT_USE_DEBUG
		size_t shadow_count = 0;
		size_t primary_count = 0;
		size_t secondary_count = 0;
		size_t other_count = 0;
		for (auto &r : localDispatch) {
			switch (r.type) {
				case gvt::render::actor::Ray::SHADOW:
				shadow_count++;
				break;
				case gvt::render::actor::Ray::PRIMARY:
				primary_count++;
				break;
				case gvt::render::actor::Ray::SECONDARY:
				secondary_count++;
				break;
				default:
				other_count++;
				break;
			}
		}
		GVT_DEBUG(DBG_ALWAYS, "Local dispatch : "
				<< localDispatch.size() << ", types: primary: "
				<< primary_count << ", shadow: " << shadow_count
				<< ", secondary: " << secondary_count
				<< ", other: " << other_count);
#endif

		// copy localDispatch rays to outgoing rays queue
		boost::unique_lock<boost::mutex> moved(adapter->_outqueue);
		moved_rays.insert(moved_rays.end(), localDispatch.begin(),
				localDispatch.end());
		moved.unlock();



	    std::cout << "adapater optix-cuda: tracing-shading time: " << shading.format();
	    std::cout << "adapater optix-cuda: copy data to device time: " << copydata.format();
	    std::cout << "adapater optix-cuda: clearDispatch time: " << clearDispatch.format();
	    std::cout << "adapater optix-cuda: convertingRaysDispatch time: " << convertingRaysDispatch.format();
	    std::cout << "adapater optix-cuda: convertingRays time: " << convertingRays.format();


		delete[] disp_tmp;



	}
};

void OptixMeshAdapter::trace(gvt::render::actor::RayVector &rayList,
                             gvt::render::actor::RayVector &moved_rays,
                             gvt::core::DBNodeH instNode, size_t _begin,
                             size_t _end) {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t_functor("OptixMeshAdapter: trace time: %w\n");
#endif

  if (_end == 0)
    _end = rayList.size();

  this->begin = _begin;
  this->end = _end;

  std::atomic<size_t> sharedIdx(begin); // shared index into rayList
  const size_t numThreads = 1; // std::thread::hardware_concurrency();
  gvt::core::schedule::asyncExec::instance()->numThreads;
  const size_t workSize =(end - begin) ;
//      std::max((size_t)1, std::min((size_t)GVT_OPTIX_PACKET_SIZE,
//                                   (size_t)((end - begin) / numThreads)));

  const size_t numworkers =
      std::max((size_t)1, std::min((size_t)numThreads,
                                   (size_t)((end - begin) / workSize)));

  // pull out information out of the database, create local structs that will be
  // passed into the parallel struct
  gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

  // pull out instance transform data
  GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: getting instance transform data");
  gvt::core::math::AffineTransformMatrix<float> *m =
      (gvt::core::math::AffineTransformMatrix<float> *)instNode["mat"]
          .value()
          .toULongLong();
  gvt::core::math::AffineTransformMatrix<float> *minv =
      (gvt::core::math::AffineTransformMatrix<float> *)instNode["matInv"]
          .value()
          .toULongLong();
  gvt::core::math::Matrix3f *normi =
      (gvt::core::math::Matrix3f *)instNode["normi"].value().toULongLong();

  //
  // TODO: wrap this db light array -> class light array conversion in some sort
  // of helper function
  // `convertLights`: pull out lights list and convert into gvt::Lights format
  // for now
  auto lightNodes = root["Lights"].getChildren();
  std::vector<gvt::render::data::scene::Light *> lights;
  lights.reserve(2);
  for (auto lightNode : lightNodes) {
    auto color = lightNode["color"].value().toVector4f();

    if (lightNode.name() == std::string("PointLight")) {
      auto pos = lightNode["position"].value().toVector4f();
      lights.push_back(new gvt::render::data::scene::PointLight(pos, color));
    } else if (lightNode.name() == std::string("AmbientLight")) {
      lights.push_back(new gvt::render::data::scene::AmbientLight(color));
    }
  }
  GVT_DEBUG(DBG_ALWAYS,
            "OptixMeshAdapter: converted "
                << lightNodes.size()
                << " light nodes into structs: size: " << lights.size());
  // end `convertLights`
  //

  // # notes
  // 20150819-2344: alim: boost threads vs c++11 threads don't seem to have much
  // of a runtime difference
  // - I was not re-using the c++11 threads though, was creating new ones every
  // time

  std::vector<std::future<void>> _tasks;

  for (size_t rc = 0; rc < numworkers; ++rc) {
    _tasks.push_back(std::async(std::launch::deferred, [&]() {
      OptixParallelTrace(this, rayList, moved_rays, sharedIdx, workSize,
                         instNode, m, minv, normi, lights, counter, begin,
                         end)();
    }));
  }

  for (auto &t : _tasks)
    t.wait();

  // GVT_DEBUG(DBG_ALWAYS, "OptixMeshAdapter: Processed rays: " << counter);
  GVT_DEBUG(DBG_ALWAYS,
            "OptixMeshAdapter: Forwarding rays: " << moved_rays.size());

  // rayList.clear();
}
