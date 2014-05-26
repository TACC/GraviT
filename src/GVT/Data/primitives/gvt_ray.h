/* 
 * File:   ray.h
 * Author: jbarbosa
 *
 * Created on March 28, 2014, 1:29 PM
 */

#ifndef RAY_H
#define	RAY_H

#include <GVT/common/debug.h>
#include <boost/container/vector.hpp>
#include <GVT/Data/scene/Color.h>
#include <GVT/Math/GVTMath.h>

namespace GVT {
    namespace Data {

        class ray {
        public:

            enum RayType {
                PRIMARY,
                SHADOW,
                SECUNDARY
            };

            //GVT_CONVERTABLE_OBJ(GVT::Data::ray);

            ray(GVT::Math::Point4f origin = GVT::Math::Point4f(0, 0, 0, 1), GVT::Math::Vector4f direction = GVT::Math::Vector4f(0, 0, 0, 0), float contribution = 1.f, RayType type = PRIMARY, int depth = 10);
            ray(ray &ray, GVT::Math::AffineTransformMatrix<float> &m);
            //ray(Ray& ray);
//            ray operator=(ray &ray){
//                origin = ray.origin;
//                direction = (ray.direction).normalized();
//                setDirection(direction);
//                t = ray.t;
//                tmin = ray.tmax;
//                tmax = ray.tmax;
//                color = ray.color;
//                domains = ray.domains;
//                id = ray.id;
//                r = ray.r;
//                b = ray.b;
//                type = ray.type;
//                w = ray.w;
//                depth = ray.depth;
//                return *this;
//            }
//            
//            ray operator=(ray ray){
//                origin = ray.origin;
//                direction = (ray.direction).normalized();
//                setDirection(direction);
//                t = ray.t;
//                tmin = ray.tmax;
//                tmax = ray.tmax;
//                color = ray.color;
//                domains = ray.domains;
//                id = ray.id;
//                r = ray.r;
//                b = ray.b;
//                type = ray.type;
//                w = ray.w;
//                depth = ray.depth;
//                return *this;
//            }
            
            ray(const ray& orig);
            virtual ~ray();

            ray(const unsigned char* buf) {
                GVT_DEBUG(DBG_ALWAYS, "Here ... ");
                origin = GVT::Math::Vector4f((float*) buf);
                buf += origin.packedSize();
                direction = GVT::Math::Vector4f((float*) buf);
                buf += direction.packedSize();
                id = *((int*) buf);
                buf += sizeof (int);
                b = *((int*) buf);
                buf += sizeof (int);
                type = *((int*) buf);
                buf += sizeof (int);
                r = *((double*) buf);
                buf += sizeof (double);
                w = *((double*) buf);
                buf += sizeof (double);
                t = *((double*) buf);
                buf += sizeof (double);
                tmax = *((double*) buf);
                buf += sizeof (double);
                color = COLOR_ACCUM(buf);
                buf += color.packedSize();
                int domain_size = *((int*) buf);
                buf += sizeof (int);
                for (int i = 0; i < domain_size; ++i, buf += sizeof (int))
                    domains.push_back(*((int*) buf));

            }

            void setDirection(GVT::Math::Vector4f dir);
            void setDirection(double *dir);
            void setDirection(float *dir);

            int packedSize() {
                int total_size = origin.packedSize() + direction.packedSize() + color.packedSize();
                total_size = 4 * sizeof (int) + 4 * sizeof (double);
                total_size = sizeof (int) * domains.size();
                return total_size;
            }

            inline int pack(unsigned char* buffer) {

                unsigned char* buf = buffer;

                buffer += origin.pack(buf);
                buffer += direction.pack(buf);
                *((int*) buf) = id;
                buf += sizeof (int);
                *((int*) buf) = b;
                buf += sizeof (int);
                *((int*) buf) = type;
                buf += sizeof (int);
                *((double*) buf) = r;
                buf += sizeof (double);
                *((double*) buf) = w;
                buf += sizeof (double);
                *((double*) buf) = t;
                buf += sizeof (double);
                *((double*) buf) = tmax;
                buf += sizeof (double);
                buf += color.pack(buf);
                *((int*) buf) = domains.size();
                buf += sizeof (int);
                for (int i = 0; i < domains.size(); ++i, buf += sizeof (int))
                    *((int*) buf) = domains[i];

                return packedSize();
            }

            friend ostream& operator<<(ostream& stream, GVT::Data::ray const& ray) {
                stream << ray.origin << "-->" << ray.direction << "[" << ray.type << "]";
                return stream;
            }


            mutable GVT::Math::Point4f origin;
            mutable GVT::Math::Vector4f direction;
            mutable GVT::Math::Vector4f inverseDirection;
            mutable int sign[3];


            int id; ///<! index into framebuffer
            int b; ///<! bounce count for ray 
            float r; ///<! sample rate
            float w; ///<! weight of image contribution
            mutable float t;
            mutable float tmin, tmax;
            mutable float tprim;
            mutable float origin_domain;
            COLOR_ACCUM color;
            boost::container::vector<int> domains;
            boost::container::vector<int> visited;
            int type;

            const static float RAY_EPSILON;
            int depth;

        };

        typedef boost::container::vector<GVT::Data::ray> RayVector;
    };
};
#endif	/* RAY_H */

