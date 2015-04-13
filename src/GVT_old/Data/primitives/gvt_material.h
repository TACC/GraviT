/* 
 * File:   gvt_material.h
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:07 PM
 */

#ifndef GVT_MATERIAL_H
#define	GVT_MATERIAL_H
#include <boost/container/vector.hpp>

#include <GVT/Math/GVTMath.h>
#include <GVT/Data/primitives/gvt_ray.h>
#include <GVT/Data/primitives/gvt_lightsource.h>
#include <time.h>


namespace GVT {
    namespace Data {

        class Material {
        public:
            Material();
            Material(const Material& orig);
            virtual ~Material();

            virtual GVT::Math::Vector4f shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, const GVT::Data::LightSource* lightSource);
            virtual GVT::Data::RayVector ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);
            virtual GVT::Data::RayVector secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);
            
            GVT::Math::Vector4f CosWeightedRandomHemisphereDirection2(GVT::Math::Vector4f n) {
                float Xi1 = (float) rand() / (float) RAND_MAX;
                float Xi2 = (float) rand() / (float) RAND_MAX;

                float theta = acos(sqrt(1.0 - Xi1));
                float phi = 2.0 * 3.1415926535897932384626433832795 * Xi2;

                float xs = sinf(theta) * cosf(phi);
                float ys = cosf(theta);
                float zs = sinf(theta) * sinf(phi);

                GVT::Math::Vector3f y(n);
                GVT::Math::Vector3f h = y;
                if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
                    h[0] = 1.0;
                else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
                    h[1] = 1.0;
                else
                    h[2] = 1.0;


                GVT::Math::Vector3f x = (h ^ y);
                GVT::Math::Vector3f z = (x ^ y);

                GVT::Math::Vector4f direction = x * xs + y * ys + z * zs;
                return direction.normalize();
            }
        protected:


        };

        class Lambert : public Material {
        public:
            Lambert(const GVT::Math::Vector4f& kd = GVT::Math::Vector4f());
            Lambert(const Lambert& orig);
            virtual ~Lambert();

            virtual GVT::Math::Vector4f shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, const GVT::Data::LightSource* lightSource);
            virtual GVT::Data::RayVector ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);
            virtual GVT::Data::RayVector secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);

        protected:

            GVT::Math::Vector4f kd;
        };

        class Phong : public Material {
        public:
            Phong(const GVT::Math::Vector4f& kd = GVT::Math::Vector4f(), const GVT::Math::Vector4f& ks = GVT::Math::Vector4f(), const float& alpha = 1.f);
            Phong(const Phong& orig);
            virtual ~Phong();

            virtual GVT::Math::Vector4f shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, const GVT::Data::LightSource* lightSource);
            virtual GVT::Data::RayVector ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);
            virtual GVT::Data::RayVector secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);

        protected:

            GVT::Math::Vector4f kd;
            GVT::Math::Vector4f ks;
            float alpha;

        };

        class BlinnPhong : public Material {
        public:
            BlinnPhong(const GVT::Math::Vector4f& kd = GVT::Math::Vector4f(), const GVT::Math::Vector4f& ks = GVT::Math::Vector4f(), const float& alpha = 1.f);
            BlinnPhong(const BlinnPhong& orig);
            virtual ~BlinnPhong();

            virtual GVT::Math::Vector4f shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, const GVT::Data::LightSource* lightSource);
            virtual GVT::Data::RayVector ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);
            virtual GVT::Data::RayVector secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples);

        protected:

            GVT::Math::Vector4f kd;
            GVT::Math::Vector4f ks;
            float alpha;

        };
    };
};


#endif	/* GVT_MATERIAL_H */

