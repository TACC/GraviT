/* 
 * File:   Light.cpp
 * Author: jbarbosa
 * 
 * Created on April 18, 2014, 3:18 PM
 */

#include "gvt/render/data/scene/Light.h"
#include "gvt/render/data/DerivedTypes.h"

using namespace gvt::core::math;
using namespace gvt::render::actor;
using namespace gvt::render::data::scene;

 Light::Light(const Point4f position) 
    : position(position) 
 {
 }

 Light::Light(const Light& orig) 
    : position(orig.position) 
 {
 }

 Light::~Light() 
 {
 }

Vector4f Light::contribution(const Ray& ray) const 
{
    return Color();
}

PointLight::PointLight(const Point4f position, const Vector4f color) 
    : Light(position), color(color) 
{

}

PointLight::PointLight(const PointLight& orig) 
    : Light(orig), color(orig.color) 
{
}

PointLight::~PointLight() 
{

}

Vector4f PointLight::contribution(const Ray& ray) const 
{
    float distance = 1.f / ((Vector4f)position-ray.origin).length();
    distance =  (distance > 1.f) ? 
    1.f : distance;
    return color * (distance + 0.5f);
}

AmbientLight::AmbientLight(const Vector4f color) 
    : Light(), color(color) 
{
}

AmbientLight::AmbientLight(const AmbientLight& orig) 
    : Light(orig), color(orig.color) 
{
}

AmbientLight::~AmbientLight() 
{
}

Vector4f AmbientLight::contribution(const Ray& ray) const 
{
    return color;
}
