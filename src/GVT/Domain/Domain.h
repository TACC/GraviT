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

            Domain(GVT::Math::AffineTransformMatrix<float> m = GVT::Math::AffineTransformMatrix<float>(true)) : m(m) {
                minv = m.inverse();
                normi = m.upper33().inverse().transpose();
            }

            Domain(const Domain &other) {
                m = other.m;
                minv = other.minv;
                normi = other.normi;
            }
            
            virtual ~Domain() {
            }

        public:

            GVT::Math::AffineTransformMatrix<float> m;
            GVT::Math::AffineTransformMatrix<float> minv;
            GVT::Math::Matrix3f normi;

            virtual bool Intersect(GVT::Data::ray&, vector<int>&) = 0;
            virtual bool LoadData() = 0;
            virtual void FreeData() = 0;
            virtual int Size() = 0;
            virtual int SizeInBytes() = 0;

            virtual GVT::Data::ray toLocal(GVT::Data::ray r) {
                GVT::Data::ray ray(r);
                ray.origin = minv * ray.origin;
                ray.direction = minv * ray.direction;
                return ray;
            }

            virtual GVT::Data::ray toWorld(GVT::Data::ray r) {
                GVT::Data::ray ray(r);
                ray.origin = m * ray.origin;
                ray.direction = m * ray.direction;
                return ray;
            }
            
            virtual GVT::Math::Vector4f toLocal(const GVT::Math::Vector4f& r) {
                GVT::Math::Vector4f ret = minv * r;
                ret.normalize();
                return ret;
            }

            virtual GVT::Math::Vector4f toWorld(const GVT::Math::Vector4f& r) {
                GVT::Math::Vector4f ret = m * r;
                ret.normalize();
                return ret;
            }

            virtual GVT::Math::Vector4f localToWorldNormal(const GVT::Math::Vector4f &v) {
                GVT::Math::Vector3f ret = normi * (GVT::Math::Vector3f)v;
                ret.normalize();
                return ret;
            }
        };

    };
};
#endif // GVT_DOMAIN_H
