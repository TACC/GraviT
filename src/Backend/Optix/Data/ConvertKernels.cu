#include <cuda.h>
#include <vector>
#include "gvt_optix.h"

//#include <thrust/device_vector.h>
//
//
//#include <thrust/device_vector.h> 
//#include <thrust/transform.h> 
//#include <thrust/sequence.h> 
//#include <thrust/copy.h> 
//#include <thrust/fill.h> 
//#include <thrust/replace.h> 
//#include <thrust/functional.h> 
#include <iostream> 

__global__ void GVT2OPTIX( float* gvtray, float* optixray, int nrays) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id > nrays) return;
    optixray[id] = gvtray[id];
}

//    template<>
//    static std::vector<GVT::Data::OptixRayFormat> 
//    transform_impl<RayVector, std::vector<OptixRayFormat> >::transform(const RayVector& rays) {
//        
//        std::vector<GVT::Data::OptixRayFormat> vector;
//        
//        return vector;
//    }

namespace GVT {

namespace Data {



}
}