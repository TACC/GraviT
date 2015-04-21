#include <cuda.h>
#include <vector>
#include <gvt/render/adapter/optix/data/Transforms.h>

#include <iostream> 

__global__ void GVT2OPTIX( float* gvtray, float* optixray, int nrays) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id > nrays) return;
    optixray[id] = gvtray[id];
}

