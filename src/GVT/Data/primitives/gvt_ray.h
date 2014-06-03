/* 
 * File:   ray.h
 * Author: jbarbosa
 *
 * Created on March 28, 2014, 1:29 PM
 */

#ifndef RAY_H
#define	RAY_H

#include <GVT/common/debug.h>
#include <GVT/Data/scene/Color.h>
#include <GVT/Math/GVTMath.h>


#include <boost/container/vector.hpp>
#include <boost/container/set.hpp>
#include <boost/smart_ptr.hpp>

#include <boost/aligned_storage.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/pool_alloc.hpp>
namespace GVT {
    namespace Data {

        typedef struct intersection{
            
            int domain;
            float d;
            
            intersection(int dom) : domain(dom),d(FLT_MAX) {}
            intersection(int dom, float dist) : domain(dom),d(dist) {}
            
            operator int(){return domain;}
            operator float(){return d;}
            friend inline bool operator == (const intersection& lhs, const intersection& rhs ) { return (lhs.d == rhs.d) && (lhs.d == rhs.d); } 
            friend inline bool operator < (const intersection& lhs, const intersection& rhs ) { return (lhs.d < rhs.d) || ((lhs.d==rhs.d) && (lhs.domain < rhs.domain)); } 
            
        } isecDom;
        typedef boost::container::vector<isecDom> isecDomList;

        
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
            ray(const ray& orig);
            ray(const unsigned char* buf);

            virtual ~ray();


            void setDirection(GVT::Math::Vector4f dir);
            void setDirection(double *dir);
            void setDirection(float *dir);

            int packedSize();

            int pack(unsigned char* buffer);

            friend ostream& operator<<(ostream& stream, GVT::Data::ray const& ray) {
                stream << ray.origin << "-->" << ray.direction << "[" << ray.type << "]";
                return stream;
            }


            mutable GVT::Math::Point4f origin;
            mutable GVT::Math::Vector4f direction;
            mutable GVT::Math::Vector4f inverseDirection;
//            mutable int sign[3];


            int id; ///<! index into framebuffer
            int depth; ///<! sample rate 
//            float r; ///<! sample rate
            float w; ///<! weight of image contribution
            mutable float t;
            COLOR_ACCUM color;
            isecDomList domains;
            int type;

            const static float RAY_EPSILON;
            
            
            
        protected:

        };

      
        typedef std::vector< GVT::Data::ray, boost::pool_allocator<GVT::Data::ray> > RayVector;


    };
};
#endif	/* RAY_H */

