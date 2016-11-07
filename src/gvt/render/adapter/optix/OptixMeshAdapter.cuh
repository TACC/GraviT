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
 * File:   OptixMeshAdapter.cuh
 * Author: Roberto Ribeiro
 *
 * Created on February 4, 2016, 11:00 PM
 */

#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_CUH
#define GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_CUH

#include "curand_kernel.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"gpu-assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ float cudaRand( );

namespace gvt {
namespace render {
namespace data {
namespace cuda_primitives {


struct CudaGvtContext {

	void initCudaBuffers(int packetSize);

	__inline__ CudaGvtContext* toGPU() {

		if (dirty)
				{
				gpuErrchk(cudaMemcpyAsync(devicePtr, this,
						sizeof(gvt::render::data::cuda_primitives::CudaGvtContext),
						cudaMemcpyHostToDevice, stream));

				dirty=false;
				}

		return (CudaGvtContext*)devicePtr;
	}

	__inline__ void  toHost() {


		gpuErrchk(cudaMemcpyAsync(this, devicePtr,
					sizeof(gvt::render::data::cuda_primitives::CudaGvtContext),
					cudaMemcpyDeviceToHost, stream));

	}

	__inline__ CudaGvtContext* devPtr(){
		return (CudaGvtContext*)devicePtr;
	}

	bool dirty;
	//shared, allocated once per runtime
	Ray * rays;
	Ray * shadowRays;
	OptixRay* traceRays;
	OptixHit* traceHits;
	Ray * dispatch;
	Light* lights;
	bool* valid;
    int nLights;
	int rayCount;
	volatile int shadowRayCount;
	volatile int dispatchCount;
	bool validRayLeft;

	 //per domain, allocated once per Adapter instance
	 //copied in adapter construct, updated per trace
	 Mesh mesh;

	 //per instance, allocated once per runtime
	 //copied per trace
	 Matrix3f* normi;
	 Matrix4f* minv;


	 cudaStream_t stream;

private:
	 void * devicePtr;

};

}
}
}
}


#endif
