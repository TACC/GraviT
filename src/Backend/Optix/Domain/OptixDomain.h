#ifndef GVT_OPTIX_DOMAIN_H
#define GVT_OPTIX_DOMAIN_H

#include <string>

#include <GVT/Data/primitives/gvt_ray.h>
#include <GVT/Domain/Domain.h>
#include <GVT/Domain/GeometryDomain.h>
#include <GVT/Math/Vector.h>
#include <optix_prime/optix_primepp.h>

namespace GVT {

namespace Domain {

class OptixDomain : public GeometryDomain {
 public:
  OptixDomain();
  OptixDomain(const OptixDomain& domain);
  explicit OptixDomain(const std::string& filename);
  OptixDomain(const std::string& filename,
              GVT::Math::AffineTransformMatrix<float> m);
  virtual ~OptixDomain();
  virtual bool load();
  void trace(GVT::Data::RayVector& rayList, GVT::Data::RayVector& moved_rays);
  optix::prime::Context& optix_context() { return optix_context_; }
  optix::prime::Model& optix_model() { return optix_model_; }

 private:
  GVT::Math::Vector4f computeNormal(uint32_t triangle_id, float u,
                                    float v) const;
  void generateSecondaryRays(const GVT::Data::ray& ray,
                             const GVT::Math::Vector4f& normal,
                             Data::RayVector& rays);
  void generateShadowRays(int triangle_id, const GVT::Data::ray& ray,
                          const GVT::Math::Vector4f& normal,
                          GVT::Data::RayVector& rays);
  void traceRay(uint32_t triangle_id, float t, float u, float v,
                GVT::Data::ray& ray, GVT::Data::RayVector& rayList);
  void traceChunk(GVT::Data::RayVector& chunk, GVT::Data::RayVector& next_list,
                  GVT::Data::RayVector& moved_rays);
  optix::prime::Context optix_context_;
  optix::prime::Model optix_model_;
  bool loaded_;
};

}  // namespace Domain

}  // namespace GVT
#endif  // GVT_OPTIX_DOMAIN_H
