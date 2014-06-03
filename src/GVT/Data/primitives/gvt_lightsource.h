/* 
 * File:   gvt_lightsource.h
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:18 PM
 */

#ifndef GVT_LIGHTSOURCE_H
#define	GVT_LIGHTSOURCE_H

#include <GVT/Data/primitives/gvt_ray.h>
#include <GVT/Math/GVTMath.h>

#include "gvt_bbox.h"

namespace GVT {
    namespace Data {

        class LightSource {
        public:
            LightSource(const GVT::Math::Point4f position = GVT::Math::Point4f());
            LightSource(const LightSource& orig);
            virtual ~LightSource();
            
            virtual GVT::Math::Vector4f contribution(const GVT::Data::ray& ray)  const;
       
            GVT::Math::Point4f position;
            
            virtual GVT::Data::box3D getWorldBoundingBox() {
                
                GVT::Data::box3D bb(position,position);
                
                return bb;
            }
            

        };

        class AmbientLightSource : public LightSource {
        public:
            AmbientLightSource(const GVT::Math::Vector4f color=GVT::Math::Vector4f(1.f,1.f,1.f,0.f));
            AmbientLightSource(const AmbientLightSource& orig);
            virtual ~AmbientLightSource();
            
            virtual GVT::Math::Vector4f contribution(const GVT::Data::ray& ray) const;
            
            GVT::Math::Vector4f color;
            
//            virtual GVT::Data::box3D getWorldBoundingBox() {
//                
//                GVT::Data::box3D bb(GVT::Math::Point4f(), GVT::Math::Point4f());
//                
//                return bb;
//            }
        };
        
        class PointLightSource : public LightSource {
        public:
            PointLightSource(const GVT::Math::Point4f position = GVT::Math::Point4f(), const GVT::Math::Vector4f color=GVT::Math::Vector4f(1.f,1.f,1.f,0.f));
            PointLightSource(const PointLightSource& orig);
            virtual ~PointLightSource();
            
            virtual GVT::Math::Vector4f contribution(const GVT::Data::ray& ray) const;
            
            GVT::Math::Vector4f color;
            
        };
    }
}


#endif	/* GVT_LIGHTSOURCE_H */

