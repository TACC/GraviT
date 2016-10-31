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
#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_H

#include "gvt/render/Adapter.h"

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>



/**
 * CUDA shading API /////
 */
#include "Mesh.cuh"
#include "Ray.cuh"
#include "Light.cuh"
#include "Material.cuh"
#include "curand_kernel.h"
#include "OptixMeshAdapter.cuh"

void shade( gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx);

curandState* set_random_states(int N);

void cudaPrepOptixRays(gvt::render::data::cuda_primitives::OptixRay* optixrays, bool* valid,
                  const int localPacketSize, gvt::render::data::cuda_primitives::Ray* rays,
                    gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx, bool,
                    cudaStream_t& stream);


void cudaProcessShadows(gvt::render::data::cuda_primitives::CudaGvtContext* cudaGvtCtx);

/**
 * /////// CUDA shading API
 */

namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace data {



struct OptixContext {


  OptixContext() {
	  optix_context_ = ::optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
  }

  void initCuda(int packetSize){
	  if (!_cudaGvtCtx[0]){

	      cudaMallocHost(&(_cudaGvtCtx[0]), sizeof (gvt::render::data::cuda_primitives::CudaGvtContext));
	      cudaMallocHost(&(_cudaGvtCtx[1]), sizeof (gvt::render::data::cuda_primitives::CudaGvtContext));

	      _cudaGvtCtx[0]->initCudaBuffers(packetSize);
	      _cudaGvtCtx[1]->initCudaBuffers(packetSize);
	  }
  }

  static OptixContext *singleton() {
    if (!_singleton) {

      _singleton = new OptixContext();
      printf("Initizalized cuda-optix adpater...\n");

    }
    return _singleton;
  };

  ::optix::prime::Context &context() { return optix_context_; }

  static OptixContext *_singleton;
  ::optix::prime::Context optix_context_;
  /*
   * Two contexts due to memory transfer and computation overlap
   */
  gvt::render::data::cuda_primitives::CudaGvtContext* _cudaGvtCtx[2] = {NULL, NULL};

};

class OptixMeshAdapter : public gvt::render::Adapter {
public:
  /**
   * Construct the Embree mesh adapter.  Convert the mesh
   * at the given node to Embree's format.
   *
   * Initializes Embree the first time it is called.
   */
  OptixMeshAdapter(gvt::render::data::primitives::Mesh *mesh);

  /**
   * Release Embree copy of the mesh.
   */
  virtual ~OptixMeshAdapter();

  ::optix::prime::Model getScene() const { return optix_model_; }

  unsigned long long getPacketSize() const { return packetSize; }

  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays, glm::mat4 *m,
                     glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
                     size_t begin = 0, size_t end = 0);

  gvt::render::data::cuda_primitives::Mesh cudaMesh;

  unsigned long long packetSize;


protected:
  /**
   * Handle to Optix context.
   */
  ::optix::prime::Context optix_context_;

  /**
   * Handle to Optix model.
   */
  ::optix::prime::Model optix_model_;

  float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();

  /**
   * Currently selected packet size flag.
   */

  size_t begin, end;

	gvt::render::data::cuda_primitives::Ray* disp_Buff[2];

	gvt::render::data::cuda_primitives::Ray* cudaRaysBuff[2];

	glm::mat4 *m_pinned;
	glm::mat4 *minv_pinned;
	glm::mat3* normi_pinned;
};
}
}
}
}
}

#endif /*GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_H*/
