/* 
 * File:   vec3.h
 * Author: jbarbosa
 *
 * Created on February 27, 2014, 2:30 PM
 */

#ifndef GVT_BBOX_H
#define	GVT_BBOX_H

#include <GVT/common/debug.h>
#include <GVT/Math/GVTMath.h>
#include <GVT/Data/primitives/gvt_ray.h>
#include <fstream>


namespace GVT {
    namespace Data {
        class box3D {
        public:
            GVT::Math::Point4f bounds[2];
  
            box3D();
            box3D(GVT::Math::Point4f vmin,  GVT::Math::Point4f vmax);
            
            box3D(const box3D &other);
            
            bool intersect(const GVT::Data::ray& r) const;
            bool intersect(const GVT::Data::ray& r, float& tmin, float& tmax) const;
            bool inBox(const GVT::Data::ray& r) const;
            bool inBox(const GVT::Math::Point4f &r) const;
            GVT::Math::Point4f getHitpoint(const GVT::Data::ray& r) const;
            bool intersectDistance(const GVT::Data::ray& r, float& t) const;
            void merge(const box3D &other);
            void expand(GVT::Math::Point4f& v);

            friend std::ostream & operator <<(std::ostream &os, const box3D &bbox) {
                
                //TODO: fix this;
                //os << bbox.bounds[0] << " x ";
                //os << bbox.bounds[1];
                // return os << bbox.bounds[0] << " : " << bbox.bounds[1];
                //return os;
            }       
            
            template<typename cast>
            operator cast() {
                GVT_ASSERT(false,"Cast operator not available from GVT BBox");
            }
        };
    };
};
#endif	/* GVT_BBOX_H */

