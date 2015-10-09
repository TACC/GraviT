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
