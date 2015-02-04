/* 
 * File:   gvt_lightsource.cpp
 * Author: jbarbosa
 * 
 * Created on April 18, 2014, 3:18 PM
 */

#include "gvt_lightsource.h"
#include "GVT/Data/derived_types.h"
namespace GVT {
    namespace Data {

        LightSource::LightSource(const GVT::Math::Point4f position) : position(position) {
        }

        LightSource::LightSource(const LightSource& orig) : position(orig.position) {
        }

        LightSource::~LightSource() {
        }

        GVT::Math::Vector4f LightSource::contribution(const GVT::Data::ray& ray) const {

            return GVT::Data::Color();
        }

        PointLightSource::PointLightSource(const GVT::Math::Point4f position, const GVT::Math::Vector4f color) : LightSource(position), color(color) {

        }

        PointLightSource::PointLightSource(const PointLightSource& orig) : LightSource(orig), color(orig.color) {

        }
        PointLightSource::~PointLightSource() {
            
        }

        GVT::Math::Vector4f PointLightSource::contribution(const GVT::Data::ray& ray) const {
            float distance = 1.f / ((GVT::Math::Vector4f)position-ray.origin).length();
            distance =  (distance > 1.f) ? 
                1.f : distance;
            return color * (distance + 0.5f);
        }
        
        AmbientLightSource::AmbientLightSource(const GVT::Math::Vector4f color) : LightSource(), color(color) {
            
        }

        AmbientLightSource::AmbientLightSource(const AmbientLightSource& orig) : LightSource(orig), color(orig.color) {

        }
        AmbientLightSource::~AmbientLightSource() {
        }

        GVT::Math::Vector4f AmbientLightSource::contribution(const GVT::Data::ray& ray) const {
            return color;
        }
    }
}