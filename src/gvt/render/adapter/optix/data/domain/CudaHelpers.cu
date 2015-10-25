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
