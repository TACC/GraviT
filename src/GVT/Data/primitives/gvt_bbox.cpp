
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

        GVT::Math::Point4f box3D::getHitpoint(const GVT::Data::ray& r) const {
            GVT::Math::Point4f hit;
            CheckLineBox(bounds[0], bounds[1], r.origin, (GVT::Math::Point4f)((GVT::Math::Vector4f)r.origin + r.direction * 1.e6f), hit);
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

        bool box3D::intersect(const GVT::Data::ray &r) const {
            //            float tmin, tmax, tymin, tymax, tzmin, tzmax;
            //
            //            tmin = (bounds[r.sign[0]].x - r.origin.x) * r.inverseDirection.x;
            //            tmax = (bounds[1 - r.sign[0]].x - r.origin.x) * r.inverseDirection.x;
            //            tymin = (bounds[r.sign[1]].y - r.origin.y) * r.inverseDirection.y;
            //            tymax = (bounds[1 - r.sign[1]].y - r.origin.y) * r.inverseDirection.y;
            //            if ((tmin > tymax) || (tymin > tmax))
            //                return false;
            //            if (tymin > tmin)
            //                tmin = tymin;
            //            if (tymax < tmax)
            //                tmax = tymax;
            //            tzmin = (bounds[r.sign[2]].z - r.origin.z) * r.inverseDirection.z;
            //            tzmax = (bounds[1 - r.sign[2]].z - r.origin.z) * r.inverseDirection.z;
            //            if ((tmin > tzmax) || (tzmin > tmax))
            //                return false;
            //            if (tzmin > tmin)
            //                tmin = tzmin;
            //            if (tzmax < tmax)
            //                tmax = tzmax;
            //            if (tmin > r.tmin) r.tmin = tmin;
            //            if (tmax < r.tmax) r.tmax = tmax;
            //            return (tmax > tmin && tmin > 0);
            float t;
            return intersectDistance(r, t);

        }

        bool box3D::intersect(const GVT::Data::ray &r, float& tmin, float& tmax) const {
            float tymin, tymax, tzmin, tzmax;

            tmin = (bounds[r.sign[0]].x - r.origin.x) * r.inverseDirection.x;
            tmax = (bounds[1 - r.sign[0]].x - r.origin.x) * r.inverseDirection.x;
            tymin = (bounds[r.sign[1]].y - r.origin.y) * r.inverseDirection.y;
            tymax = (bounds[1 - r.sign[1]].y - r.origin.y) * r.inverseDirection.y;
            if ((tmin > tymax) || (tymin > tmax))
                return false;
            if (tymin > tmin)
                tmin = tymin;
            if (tymax < tmax)
                tmax = tymax;
            tzmin = (bounds[r.sign[2]].z - r.origin.z) * r.inverseDirection.z;
            tzmax = (bounds[1 - r.sign[2]].z - r.origin.z) * r.inverseDirection.z;
            if ((tmin > tzmax) || (tzmin > tmax))
                return false;
            if (tzmin > tmin)
                tmin = tzmin;
            if (tzmax < tmax)
                tmax = tzmax;
            if (tmin > r.tmin) r.tmin = tmin;
            if (tmax < r.tmax) r.tmax = tmax;
            return (tmax > tmin && tmin > 0);
        }

        bool box3D::inBox(const GVT::Data::ray &r) const {
            //            return InBox(r.origin,bounds[0],bounds[1],0) && InBox(r.origin,bounds[0],bounds[1],1) && InBox(r.origin,bounds[0],bounds[1],2);
            return inBox(r.origin);

        }

        bool box3D::inBox(const GVT::Math::Point4f &origin) const {
            bool TT[3];
            TT[0] = ((bounds[0].x - origin.x) <= FLT_EPSILON && (bounds[1].x - origin.x) >= -FLT_EPSILON);
            TT[1] = ((bounds[0].y - origin.y) <= FLT_EPSILON && (bounds[1].y - origin.y) >= -FLT_EPSILON);
            TT[2] = ((bounds[0].z - origin.z) <= FLT_EPSILON && (bounds[1].z - origin.z) >= -FLT_EPSILON);
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
#if 0

        bool box3D::intersectDistance(const GVT::Data::ray& r, float& t) const {
            //            GVT::Math::Point4f p = r.origin;
            //            GVT::Math::Vector4f d = r.direction;

            GVT::Math::Vector4f scale;
            for (int i = 0; i < 4; i++)
                scale[i] = (bounds[1][i] - bounds[0][i]);

            GVT::Math::Vector4f translate = (-bounds[0] - scale / 2.f);


            GVT::Math::AffineTransformMatrix<float> mtrans = GVT::Math::AffineTransformMatrix<float>::createTranslation(translate[0], translate[1], translate[2]);
            GVT::Math::AffineTransformMatrix<float> mscale = GVT::Math::AffineTransformMatrix<float>::createScale(1.f / scale[0], 1.f / scale[1], 1.f / scale[2]);



            GVT::Math::AffineTransformMatrix<float> m0 = mscale * mtrans;
            GVT::Math::AffineTransformMatrix<float> mi = m0.inverse();

            //            GVT_DEBUG(DBG_ALWAYS,"Ttt = " << scale << " " << translate);
            //            GVT_DEBUG(DBG_ALWAYS,"BB0 = " << (bounds[0]) << " " << (bounds[1]));
            //            GVT_DEBUG(DBG_ALWAYS,"BBM = " << (m0*bounds[0]) << " " << (m0*bounds[1]));

            GVT::Math::Point4f p = m0 * r.origin;
            GVT::Math::Vector4f d = m0.upper33() * (GVT::Math::Vector3f)r.direction;

            int it;
            float x, y, bestT;
            int mod0, mod1, mod2, bestIndex;

            bestT = FLT_MAX;
            bestIndex = -1;

            for (it = 0; it < 6; it++) {
                mod0 = it % 3;

                if (d[mod0] == 0) {
                    continue;
                }

                t = ((it / 3) - 0.5 - p[mod0]) / d[mod0];

                if (t < GVT::Data::ray::RAY_EPSILON || t > bestT) {
                    continue;
                }

                mod1 = (it + 1) % 3;
                mod2 = (it + 2) % 3;
                x = p[mod1] + t * d[mod1];
                y = p[mod2] + t * d[mod2];

                if (x <= 0.5 && x >= -0.5 && y <= 0.5 && y >= -0.5) {
                    if (bestT > t) {
                        bestT = t;
                        bestIndex = it;
                    }
                }

            }

            if (bestIndex < 0) return false;

            //t = bestT;

            // if (t < 0) return false;

            GVT::Math::Point4f p2 = p + d * t;
            t = ((mi * p2) - r.origin).length();


            return true;

        }

#endif

        bool box3D::intersectDistance(const GVT::Data::ray& r, float& t) const {

            float t1 = (bounds[0].x - r.origin.x) * r.inverseDirection.x;
            float t2 = (bounds[1].x - r.origin.x) * r.inverseDirection.x;
            float t3 = (bounds[0].y - r.origin.y) * r.inverseDirection.y;
            float t4 = (bounds[1].y - r.origin.y) * r.inverseDirection.y;
            float t5 = (bounds[0].z - r.origin.z) * r.inverseDirection.z;
            float t6 = (bounds[1].z - r.origin.z) * r.inverseDirection.z;

            float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
            float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

            if (tmax < 0) {
                t = tmax;
                return false;
            }

            if (tmin > tmax) {
                t = tmax;
                return false;
            }

            t = tmin;
            return true;
        };
    };
};
