#include "CudaHelpers.h"
#include <thrust/transform.h>

namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace domain {
/*
struct gvtRays2OptixRays {

  gvtRays2OptixRays() {}

  __host__ __device__ gvt::render::adapter::optix::data::OptixRay
  operator()(const gvt::render::actor::Ray &gvt_ray) {
    gvt::render::adapter::optix::data::OptixRay optix_ray;
    optix_ray.origin[0] = gvt_ray.origin.n[0];
    optix_ray.origin[1] = gvt_ray.origin.n[1];
    optix_ray.origin[2] = gvt_ray.origin.n[2];
    optix_ray.direction[0] = gvt_ray.direction.n[0];
    optix_ray.direction[1] = gvt_ray.direction.n[1];
    optix_ray.direction[2] = gvt_ray.direction.n[2];
    return optix_ray;
  }
};

void transformrays(gvtRaysDevice &grays, gvtOptixRaysDevice &orays) {
  thrust::transform(grays.begin(), grays.end(), orays.begin(),
                    gvtRays2OptixRays());
};
*/
}; /* namespace domain */
}; /* namespace optix */
}; /* namespace adapter */
}; /* namespace render */
}; /* namesaoce gvt */
