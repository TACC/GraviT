/* 
 * File:   Light.h
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:18 PM
 */

#ifndef GVT_RENDER_DATA_SCENE_LIGHT_H
#define	GVT_RENDER_DATA_SCENE_LIGHT_H

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/primitives/BBox.h>
#include <gvt/core/Math.h>

 namespace gvt {
    namespace render {
        namespace data {
            namespace scene {
                /// base class for light sources
                /** \sa AmbientLight, PointLight
                */
                class Light 
                {
                public:
                    Light(const gvt::core::math::Point4f position = gvt::core::math::Point4f());
                    Light(const Light& orig);
                    virtual ~Light();
                    
                    virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray& ray)  const;
                    
                    gvt::core::math::Point4f position;
                    
                    virtual gvt::render::data::primitives::Box3D getWorldBoundingBox() 
                    {    
                        gvt::render::data::primitives::Box3D bb(position,position);   
                        return bb;
                    }
                };
                /// general lighting factor added to each successful ray intersection
                class AmbientLight : public Light 
                {
                public:
                    AmbientLight(const gvt::core::math::Vector4f color=gvt::core::math::Vector4f(1.f,1.f,1.f,0.f));
                    AmbientLight(const AmbientLight& orig);
                    virtual ~AmbientLight();
                    
                    virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray& ray) const;
                    
                    gvt::core::math::Vector4f color;
                };
                /// point light source
                class PointLight : public Light 
                {
                public:
                    PointLight(const gvt::core::math::Point4f position = gvt::core::math::Point4f(), const gvt::core::math::Vector4f color=gvt::core::math::Vector4f(1.f,1.f,1.f,0.f));
                    PointLight(const PointLight& orig);
                    virtual ~PointLight();
                    
                    virtual gvt::core::math::Vector4f contribution(const gvt::render::actor::Ray& ray) const;
                    
                    gvt::core::math::Vector4f color;
                    
                };
            }
        }
    }
}

#endif	/* GVT_RENDER_DATA_SCENE_LIGHT_H */

