//
// MantaMeshAdapter.h
//

#ifndef GVT_RENDER_ADAPTER_MANTA_DATA_MANTA_MESH_ADAPTER_H
#define GVT_RENDER_ADAPTER_MANTA_DATA_MANTA_MESH_ADAPTER_H

#include "gvt/render/Adapter.h"

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

// class MantaMeshAdapter : public gvt::render::data::domain::GeometryDomain
class MantaMeshAdapter : public gvt::render::Adapter
{
public:
    // MantaMeshAdapter(gvt::render::data::domain::GeometryDomain* domain);
    /**
     * Construct the Manta mesh adapter.  Convert the mesh
     * at the given node to Manta's format.
     *
     * Initializes Manta the first time it is called.
     */
    MantaMeshAdapter(gvt::core::DBNodeH node);

    // MantaMeshAdapter(std::string filename ="",gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
    // MantaMeshAdapter(const MantaMeshAdapter& other);
 
    /**
     * Release Manta copy of the mesh.
     */
    virtual ~MantaMeshAdapter();

    virtual bool load();
    virtual void free();

    Manta::RenderContext*   getRenderContext() { return rContext; }

    /**
     * Return the Manta DynBVH acceleration structure.
     */
    Manta::DynBVH*          getAccelStruct() { return as; }

    /**
     * Return pointer to the Manta mesh.
     */
    Manta::Mesh*            getMantaMesh() { return meshManta; }
    
    /**
     * Trace rays using the Manta adapter.
     *
     * \param rayList incoming rays
     * \param moved_rays outgoing rays [rays that did not hit anything]
     * \param instNode instance db node containing dataRef and transforms
     */
    virtual void trace(gvt::render::actor::RayVector& rayList,
                       gvt::render::actor::RayVector& moved_rays,
                       gvt::core::DBNodeH instNode);
    // void trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays);
    
protected:
    /**
     * Pointer to the Manta render context.
     */
    Manta::RenderContext* rContext;

    /**
     * Pointer to the Manta DynBVH acceleration structrue.
     */
    Manta::DynBVH* as;

    /**
     * Pointer to the Manta mesh.
     */
    Manta::Mesh* meshManta;
};
                        
}
}
}
}
}


#endif // GVT_RENDER_ADAPTER_MANTA_DATA_DOMAIN_MANTA_DOMAIN_H
