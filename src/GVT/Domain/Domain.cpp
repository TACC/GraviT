
#include "Domain.h"


namespace GVT {
    namespace Domain {

        Domain::Domain(GVT::Math::AffineTransformMatrix<float> m) : m(m), domainID(-1), isLoaded(false) {
            minv = m.inverse();
            normi = m.upper33().inverse().transpose();
        }

        Domain::Domain(const Domain &other) {
            m = other.m;
            minv = other.minv;
            normi = other.normi;
        }

        Domain::~Domain() {
        }

        bool Domain::intersect(GVT::Data::ray* r, GVT::Data::isecDomList& inter) {
            float t;
            if (getWorldBoundingBox().intersectDistance(r, t) && t > GVT::Data::ray::RAY_EPSILON) {
                inter.push_back(GVT::Data::isecDom(domainID, t));
                return true;
            }
            return false;
        };


        // TODO : This code is broken

        void Domain::marchIn(GVT::Data::ray* r) {
            GVT::Data::box3D wBox = getWorldBoundingBox();
            float t = FLT_MAX;
            r->setDirection(-r->direction);
            while(wBox.inBox(r->origin)) {
                if(wBox.intersectDistance(r,t)) r->origin += r->direction * t;
                r->origin += r->direction * GVT::Data::ray::RAY_EPSILON;
            }
            r->setDirection(-r->direction);

        };
        // TODO : This code is broken
        void Domain::marchOut(GVT::Data::ray* r) {
            GVT::Data::box3D wBox = getWorldBoundingBox();
            float t = FLT_MAX;
            //int i =0;
            
            if(wBox.intersectDistance(r,t)) r->origin += r->direction * t;
            while(wBox.intersectDistance(r,t)) {
                r->origin += r->direction * t;
                r->origin += r->direction * GVT::Data::ray::RAY_EPSILON;
            }
            r->origin += r->direction * GVT::Data::ray::RAY_EPSILON;
        };
        
        void Domain::trace(GVT::Data::RayVector& rayList, GVT::Data::RayVector& moved_rays) {
            GVT_ASSERT(false,"Trace function for this domain was not implemented");
        }

        bool Domain::load() {
            GVT_ASSERT(false, "Calling domain load generic function\n");
            return false;
        }

        void Domain::free() {
            GVT_WARNING(false, "Calling domain free generic function\n");
            return;
        }

        GVT::Data::ray Domain::toLocal(GVT::Data::ray& r) {
            GVT_ASSERT((&r),"NULL POINTER");
            GVT::Data::ray ray(r);
            ray.origin = minv * ray.origin;
            ray.direction = minv * ray.direction;
            return ray;
        }

        GVT::Data::ray Domain::toWorld(GVT::Data::ray& r) {
            GVT_ASSERT((&r),"NULL POINTER");
            GVT::Data::ray ray(r);
            ray.origin = m * ray.origin;
            ray.direction = m * ray.direction;
            return ray;
        }

        GVT::Math::Vector4f Domain::toLocal(const GVT::Math::Vector4f& r) {
            GVT::Math::Vector4f ret = minv * r;
            ret.normalize();
            return ret;
        }

        GVT::Math::Vector4f Domain::toWorld(const GVT::Math::Vector4f& r) {
            GVT::Math::Vector4f ret = m * r;
            ret.normalize();
            return ret;
        }

        GVT::Math::Vector4f Domain::localToWorldNormal(const GVT::Math::Vector4f &v) {
            GVT::Math::Vector3f ret = normi * (GVT::Math::Vector3f)v;
            ret.normalize();
            return ret;
        }

        GVT::Data::box3D Domain::getWorldBoundingBox() {
            return getBounds(1);
        }

        void Domain::setBoundingBox(GVT::Data::box3D bb) {
            boundingBox = bb;
        }

        GVT::Data::box3D Domain::getBounds(int type = 0) {
            if (type == 0) {
                return boundingBox;
            } else {
                GVT::Data::box3D bb; // = boundingBox;
                bb.bounds[0] = m * boundingBox.bounds[0];
                bb.bounds[1] = m * boundingBox.bounds[1];
                return bb;

            }
        }

        bool Domain::domainIsLoaded() {
            return isLoaded;
        }

        int Domain::getDomainID() {
            return domainID;
        }

        void Domain::setDomainID(int id) {
            domainID = id;
        }
    }
}
