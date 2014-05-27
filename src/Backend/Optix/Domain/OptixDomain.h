

#ifndef GVT_OPTIX_DOMAIN_H
#define GVT_OPTIX_DOMAIN_H

#include <string>

#include <GVT/Domain/Domain.h>
#include <GVT/Domain/GeometryDomain.h>
#include <GVT/Data/primitives.h>
namespace GVT {
    namespace Domain {
        class OptixDomain : public GeometryDomain {
        public:
            OptixDomain(std::string filename = "");
            OptixDomain(const OptixDomain& other);
            virtual ~OptixDomain();
        };
    };
};


#endif // GVT_OPTIX_DOMAIN_H
