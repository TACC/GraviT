//
// MantaDomain.C
//

#include <GVT/common/utils.h>
#include <Domain/MantaDomain.h>
#include <Data/gvt_manta.h>
namespace GVT {
    namespace Domain {
        
        MantaDomain::MantaDomain(string filename, GVT::Math::AffineTransformMatrix<float> m) : GVT::Domain::GeometryDomain(filename,m) {
        }

        MantaDomain::MantaDomain(const MantaDomain& other) : GVT::Domain::GeometryDomain(other) {
        }
        
        MantaDomain::~MantaDomain() {
                //GeometryDomain::~GeometryDomain();
            
        }
    };
};


