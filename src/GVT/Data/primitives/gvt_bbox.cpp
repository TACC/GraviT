
#include "gvt_ray.h"
#include "gvt_bbox.h"


namespace GVT {
    namespace Data {

        int inline GetIntersection(float fDst1, float fDst2, GVT::Math::Point4f P1, GVT::Math::Point4f P2, GVT::Math::Point4f &Hit) {
            if ((fDst1 * fDst2) >= 0.0f) return 0;
            if (fDst1 == fDst2) return 0;
            Hit = P1 + (P2 - P1) * (-fDst1 / (fDst2 - fDst1));
            return 1;
        }

        int inline InBox(GVT::Math::Point4f Hit, GVT::Math::Point4f B1, GVT::Math::Point4f B2, const int Axis) {
            if (Axis == 1 && Hit.z > B1.z && Hit.z < B2.z && Hit.y > B1.y && Hit.y < B2.y) return 1;
            if (Axis == 2 && Hit.z > B1.z && Hit.z < B2.z && Hit.x > B1.x && Hit.x < B2.x) return 1;
            if (Axis == 3 && Hit.x > B1.x && Hit.x < B2.x && Hit.y > B1.y && Hit.y < B2.y) return 1;
            return 0;
        }

        // returns true if line (L1, L2) intersects with the box (B1, B2)
        // returns intersection point in Hit

        int inline CheckLineBox(GVT::Math::Point4f B1, GVT::Math::Point4f B2, GVT::Math::Point4f L1, GVT::Math::Point4f L2, GVT::Math::Point4f &Hit) {
            if (L2.x < B1.x && L1.x < B1.x) return false;
            if (L2.x > B2.x && L1.x > B2.x) return false;
            if (L2.y < B1.y && L1.y < B1.y) return false;
            if (L2.y > B2.y && L1.y > B2.y) return false;
            if (L2.z < B1.z && L1.z < B1.z) return false;
            if (L2.z > B2.z && L1.z > B2.z) return false;
            if (L1.x > B1.x && L1.x < B2.x &&
                    L1.y > B1.y && L1.y < B2.y &&
                    L1.z > B1.z && L1.z < B2.z) {
                Hit = L1;
                return true;
            }
            if ((GetIntersection(L1.x - B1.x, L2.x - B1.x, L1, L2, Hit) && InBox(Hit, B1, B2, 1))
                    || (GetIntersection(L1.y - B1.y, L2.y - B1.y, L1, L2, Hit) && InBox(Hit, B1, B2, 2))
                    || (GetIntersection(L1.z - B1.z, L2.z - B1.z, L1, L2, Hit) && InBox(Hit, B1, B2, 3))
                    || (GetIntersection(L1.x - B2.x, L2.x - B2.x, L1, L2, Hit) && InBox(Hit, B1, B2, 1))
                    || (GetIntersection(L1.y - B2.y, L2.y - B2.y, L1, L2, Hit) && InBox(Hit, B1, B2, 2))
                    || (GetIntersection(L1.z - B2.z, L2.z - B2.z, L1, L2, Hit) && InBox(Hit, B1, B2, 3)))
                return true;

            return false;
        }

        box3D::box3D() {

        }

        GVT::Math::Point4f box3D::getHitpoint(const GVT::Data::ray& ray) const {
            GVT::Math::Point4f hit;
            CheckLineBox(bounds[0], bounds[1], ray.origin, (GVT::Math::Point4f)((GVT::Math::Vector4f)ray.origin + ray.direction * 1.e6f), hit);
            return hit;
        }

        box3D::box3D(GVT::Math::Point4f vmin, GVT::Math::Point4f vmax) {
            for (int i = 0; i < 4; i++) {
                bounds[0][i] = std::min(vmin[i], vmax[i]);
                bounds[1][i] = std::max(vmin[i], vmax[i]);
            }
        }

        box3D::box3D(const box3D &other) {
            for (int i = 0; i < 4; i++) {
                bounds[0][i] = std::min(other.bounds[0][i], other.bounds[1][i]);
                bounds[1][i] = std::max(other.bounds[0][i], other.bounds[1][i]);
            }
        }

        bool box3D::intersect(const GVT::Data::ray& r) const {
            float t;
            return intersectDistance(r, t);

        }

        bool box3D::inBox(const GVT::Data::ray& r) const {
            return inBox(r.origin);
        }

        bool box3D::inBox(const GVT::Math::Point4f &origin) const {
            bool TT[3];
//            
//            GVT::Math::Vector4f lb = bounds[0] - origin;
//            GVT::Math::Vector4f ub = bounds[1] - origin;
//            
            TT[0] = ((bounds[0].x - origin.x) <= FLT_EPSILON && (bounds[1].x - origin.x) >= -FLT_EPSILON);
            if(!TT[0]) return false;
            TT[1] = ((bounds[0].y - origin.y) <= FLT_EPSILON && (bounds[1].y - origin.y) >= -FLT_EPSILON);
            if(!TT[0]) return false;
            TT[2] = ((bounds[0].z - origin.z) <= FLT_EPSILON && (bounds[1].z - origin.z) >= -FLT_EPSILON);
            if(!TT[0]) return false;
            return (TT[0] && TT[1] && TT[2]);
        }

        void box3D::merge(const box3D &other) {
            bounds[0][0] = min(other.bounds[0][0], bounds[0][0]);
            bounds[0][1] = min(other.bounds[0][1], bounds[0][1]);
            bounds[0][2] = min(other.bounds[0][2], bounds[0][2]);

            bounds[1][0] = max(other.bounds[1][0], bounds[1][0]);
            bounds[1][1] = max(other.bounds[1][1], bounds[1][1]);
            bounds[1][2] = max(other.bounds[1][2], bounds[1][2]);

        }

        void box3D::expand(GVT::Math::Point4f & v) {
            bounds[0][0] = min(bounds[0][0], v[0]);
            bounds[0][1] = min(bounds[0][1], v[1]);
            bounds[0][2] = min(bounds[0][2], v[2]);

            bounds[1][0] = max(bounds[1][0], v[0]);
            bounds[1][1] = max(bounds[1][1], v[1]);
            bounds[1][2] = max(bounds[1][2], v[2]);
        }

        bool box3D::intersectDistance(const GVT::Data::ray& ray, float& t) const {
            
            float t1 = (bounds[0].x - ray.origin.x) * ray.inverseDirection.x;
            float t2 = (bounds[1].x - ray.origin.x) * ray.inverseDirection.x;
            float t3 = (bounds[0].y - ray.origin.y) * ray.inverseDirection.y;
            float t4 = (bounds[1].y - ray.origin.y) * ray.inverseDirection.y;
            float t5 = (bounds[0].z - ray.origin.z) * ray.inverseDirection.z;
            float t6 = (bounds[1].z - ray.origin.z) * ray.inverseDirection.z;

            float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
            float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

            if (tmax < 0 || tmin > tmax) return false;
            
            t = (tmin > 0) ? t = tmin : tmax;
            
            return (t > FLT_EPSILON);
            
        };
    };
};
