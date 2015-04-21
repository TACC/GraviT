//
// MantaDomain.h
//

#ifndef GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H
#define GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H

#include <gvt/render/adapter/manta/override/DynBVH.h>
#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/domain/GeometryDomain.h>
#include <gvt/render/data/Primitives.h>

// begin Manta includes
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/BBox.h>
#include <Interface/Context.h>
#include <Interface/LightSet.h>
#include <Interface/MantaInterface.h>
#include <Interface/Object.h>
#include <Interface/Scene.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Lights/PointLight.h>
#include <Model/Materials/Phong.h>
#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Model/Readers/PlyReader.h>
// end Manta includes

#include <string>

namespace gvt {
    namespace render {
        namespace adapter {
            namespace manta {
                namespace data {
                    namespace domain {

                        class MantaDomain : public gvt::render::data::domain::GeometryDomain 
                        {
                        public:
                            MantaDomain(gvt::render::data::domain::GeometryDomain* domain);
                            MantaDomain(std::string filename ="",gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
                            MantaDomain(const MantaDomain& other);
                            virtual ~MantaDomain();

                            virtual bool load();
                            virtual void free();

                            Manta::RenderContext*   getRenderContext() { return rContext; }
                            Manta::DynBVH*          getAccelStruct() { return as; }
                            Manta::Mesh*            getMantaMesh() { return meshManta; }
                            
                            void trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays);
                            
                        protected:
                            Manta::RenderContext* rContext;
                            Manta::DynBVH* as;
                            Manta::Mesh* meshManta;
                        };
                        
                    }
                }
            }
        }
    }
}


#endif // GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H
