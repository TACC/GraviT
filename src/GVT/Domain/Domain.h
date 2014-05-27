//
// Domain.h
//

#ifndef GVT_DOMAIN_H
#define GVT_DOMAIN_H

#include <GVT/Data/primitives.h>
#include <GVT/Math/GVTMath.h>
#include <vector>
using namespace std;

namespace GVT {
    namespace Domain {

        class Domain {
        protected:

            Domain() {};
            Domain(GVT::Math::AffineTransformMatrix<float> m);

            Domain(const Domain &other);
            virtual ~Domain();
            
        public:

            virtual bool intersect(GVT::Data::ray& r, GVT::Data::isecDomList& inter);
            
            virtual void marchIn(GVT::Data::ray& r);
            virtual void marchOut(GVT::Data::ray& r);
            

            virtual bool load();
            virtual void free();
            virtual int size() = 0;
            virtual int sizeInBytes() = 0;

            virtual GVT::Data::ray toLocal(GVT::Data::ray r);

            virtual GVT::Data::ray toWorld(GVT::Data::ray r);

            virtual GVT::Math::Vector4f toLocal(const GVT::Math::Vector4f& r);

            virtual GVT::Math::Vector4f toWorld(const GVT::Math::Vector4f& r);

            virtual GVT::Math::Vector4f localToWorldNormal(const GVT::Math::Vector4f &v);
            virtual GVT::Data::box3D getWorldBoundingBox();

            virtual void setBoundingBox(GVT::Data::box3D bb);
            
            
            virtual GVT::Data::box3D getBounds(int type);

            virtual bool domainIsLoaded();

            virtual int getDomainID();

            virtual void setDomainID(int id);

            // Public variables
            GVT::Math::AffineTransformMatrix<float> m;
            GVT::Math::AffineTransformMatrix<float> minv;
            GVT::Math::Matrix3f normi;
            GVT::Data::box3D boundingBox;

            int domainID;

            bool isLoaded;

        };

    };
};
#endif // GVT_DOMAIN_H
