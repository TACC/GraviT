#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_DOMAIN_OPTIX_DOMAIN_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_DOMAIN_OPTIX_DOMAIN_H

#include <string>

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/Domains.h>
#include <gvt/core/Math.h>
#include <optix_prime/optix_primepp.h>

namespace gvt {
  namespace render {
    namespace adapter {
      namespace optix {
        namespace data {
          namespace domain {

            class OptixDomain : public GeometryDomain 
            {
             public:
              OptixDomain();
              OptixDomain(const OptixDomain& domain);
              explicit OptixDomain(const std::string& filename);
              OptixDomain(const std::string& filename,
                          gvt::core::math::AffineTransformMatrix<float> m);
              virtual ~OptixDomain();
              virtual bool load();
              void trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays);
              optix::prime::Context& optix_context() { return optix_context_; }
              optix::prime::Model& optix_model() { return optix_model_; }

             private:
              gvt::core::math::Vector4f computeNormal(uint32_t triangle_id, float u,
                                                float v) const;
              void generateSecondaryRays(const gvt::render::actor::Ray& ray,
                                         const gvt::core::math::Vector4f& normal,
                                         gvt::render::actor::RayVector& rays);
              void generateShadowRays(int triangle_id, const gvt::render::actor::Ray& ray,
                                      const gvt::core::math::Vector4f& normal,
                                      gvt::render::actor::RayVector& rays);
              void traceRay(uint32_t triangle_id, float t, float u, float v,
                            gvt::render::actor::Ray& ray, gvt::render::actor::RayVector& rayList);
              void traceChunk(gvt::render::actor::RayVector& chunk, gvt::render::actor::RayVector& next_list,
                              gvt::render::actor::RayVector& moved_rays);
              optix::prime::Context optix_context_;
              optix::prime::Model optix_model_;
              bool loaded_;
            };
          }
        }
      }
    }
  }
}

#endif  // GVT_RENDER_ADAPTER_OPTIX_DATA_DOMAIN_OPTIX_DOMAIN_H
