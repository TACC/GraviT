//
// MantaDomain.h
//

#ifndef GVT_MANTA_DOMAIN_H
#define GVT_MANTA_DOMAIN_H

#include <GVT/Domain/Domain.h>
#include <GVT/Domain/GeometryDomain.h>

#include <Backend/Manta/MantaOverride/DynBVH.h>

#include <string>

#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Interface/MantaInterface.h>
#include <Interface/Scene.h>
#include <Interface/Object.h>
#include <Interface/Context.h>
#include <Core/Geometry/BBox.h>
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Materials/Phong.h>
#include <Model/Readers/PlyReader.h>
#include <Interface/LightSet.h>
#include <Model/Lights/PointLight.h>

#include <GVT/Data/primitives.h>

namespace GVT {
    namespace Domain {

    

    class MantaDomain : public GeometryDomain {
    public:
        
        MantaDomain(GVT::Domain::GeometryDomain* domain);
        MantaDomain(string filename ="",GVT::Math::AffineTransformMatrix<float> m = GVT::Math::AffineTransformMatrix<float>(true));
        MantaDomain(const MantaDomain& other);
        virtual ~MantaDomain();

        virtual bool load();
        virtual void free();
        
        void trace(GVT::Data::RayVector& rayList, GVT::Data::RayVector& moved_rays);
        
        Manta::RenderContext* rContext;
        Manta::DynBVH* as;
        Manta::Mesh* meshManta;
    protected:

        //std::vector<Manta::BBox> TraverseBVH(int nodeId, int depth, int maxDepth);
        
        
        
        
        
    };

};
};


#endif // GVT_MANTA_DOMAIN_H
