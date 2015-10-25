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
#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_DOMAIN_OPTIX_DOMAIN_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_DOMAIN_OPTIX_DOMAIN_H

#include <string>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/optix/data/Transforms.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

//#include <gvt/core/Context.h>
#include <gvt/core/Math.h>
#include <optix_prime/optix_primepp.h>

//using namespace optix::prime;
namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace data {
namespace domain {

/// data adapter for NVIDIA OptiX Prime ray tracer
/** this helper class transforms geometry data from the GraviT internal format 
to the format expected by NVIDIA's OptiX Prime ray tracer
*/
class OptixDomain : public gvt::render::data::domain::GeometryDomain {
 public:
  OptixDomain();
  OptixDomain(const OptixDomain& domain);
  OptixDomain(gvt::render::data::domain::GeometryDomain* domain);
  explicit OptixDomain(const std::string& filename);
  OptixDomain(const std::string& filename,
              gvt::core::math::AffineTransformMatrix<float> m);
  virtual ~OptixDomain();
  virtual bool load();
  void trace(gvt::render::actor::RayVector& rayList,
             gvt::render::actor::RayVector& moved_rays);
  // optix::prime::Context& optix_context() { return optix_context_; }
  ::optix::prime::Context& optix_context() { return optix_context_; }
  // optix::prime::Model& optix_model() { return optix_model_; }
  ::optix::prime::Model& optix_model() { return optix_model_; }

 private:
  gvt::core::math::Vector4f computeNormal(const uint32_t& triangle_id,
                                          const float& u, const float& v) const;
  void generateSecondaryRays(const gvt::render::actor::Ray& ray,
                             const gvt::core::math::Vector4f& normal,
                             gvt::render::actor::RayVector& rays);
  void generateShadowRays(const int& triangle_id,
                          const gvt::render::actor::Ray& ray,
                          const gvt::core::math::Vector4f& normal,
                          gvt::render::actor::RayVector& rays);
  void traceRay(uint32_t triangle_id, float t, float u, float v,
                gvt::render::actor::Ray& ray,
                gvt::render::actor::RayVector& rayList);
  void traceChunk(gvt::render::actor::RayVector& chunk,
                  gvt::render::actor::RayVector& next_list,
                  gvt::render::actor::RayVector& moved_rays);
  // optix::prime::Context optix_context_;
  ::optix::prime::Context optix_context_;
  // optix::prime::Model optix_model_;
  ::optix::prime::Model optix_model_;
  float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon(); 
  /*              thrust::device_vector<float> _vertices;
                thrust::device_vector<int> _faces;
                thrust::device_vector<float> _normals;
  */

  bool loaded_;
}; /* class Optix Domain */
}; /* namespace domain */
}; /* namespace data */
}; /* namespace optix */
}; /* namaspace adpter */
}; /* namespace render */
}; /* namespace gvt */

#endif  // GVT_RENDER_ADAPTER_OPTIX_DATA_DOMAIN_OPTIX_DOMAIN_H
