//
// EmbreeDomain.h
//

#ifndef GVT_RENDER_ADAPTER_EMBREE_DATA_DOMAIN_EMBREE_DOMAIN_H
#define GVT_RENDER_ADAPTER_EMBREE_DATA_DOMAIN_EMBREE_DOMAIN_H

#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/domain/GeometryDomain.h>
#include <gvt/render/data/Primitives.h>

// Embree includes
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
// end Embree includes

#include <string>

#ifdef __USE_TAU
#include <TAU.h>
#endif

namespace gvt {
    namespace render {
        namespace adapter {
            namespace embree {
                namespace data {
                    namespace domain {

                        class EmbreeDomain : public gvt::render::data::domain::GeometryDomain
                        {
                        public:
                            EmbreeDomain(gvt::render::data::domain::GeometryDomain* domain);
                            EmbreeDomain(std::string filename ="",gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
                            EmbreeDomain(const EmbreeDomain& other);
                            virtual ~EmbreeDomain();

                            virtual bool load();
                            virtual void free();

#if 0
                            Manta::RenderContext*   getRenderContext() { return rContext; }
                            Manta::DynBVH*          getAccelStruct() { return as; }
                            Manta::Mesh*            getMantaMesh() { return meshManta; }
#endif
                            RTCScene getScene() { return scene; }
                            unsigned getGeomId() { return geomId; }

                            void trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays);

                        protected:
                            static bool init;

                            RTCAlgorithmFlags packetSize;
                            RTCScene scene;
                            unsigned geomId;
#if 0
                            Manta::RenderContext* rContext;
                            Manta::DynBVH* as;
                            Manta::Mesh* meshManta;
#endif
                        };

                    }
                }
            }
        }
    }
}


#endif // GVT_RENDER_ADAPTER_EMBREE_DATA_DOMAIN_EMBREE_DOMAIN_H
