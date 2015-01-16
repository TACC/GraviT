#include <cuda.h>
#include "optix_dataformat.h"

__global__ void GVT2OPTIX( float* gvtray, float* optixray, int nrays) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id > nrays) return;
    optixray[id] = gvtray[id];
}
