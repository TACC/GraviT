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
        MantaDomain(string filename ="",GVT::Math::AffineTransformMatrix<float> m = GVT::Math::AffineTransformMatrix<float>(true));
        MantaDomain(const MantaDomain& other);
        virtual ~MantaDomain();

    protected:

        //std::vector<Manta::BBox> TraverseBVH(int nodeId, int depth, int maxDepth);

    };

};
};


#endif // GVT_MANTA_DOMAIN_H
