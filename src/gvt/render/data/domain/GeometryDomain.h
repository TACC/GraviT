//
// GeometryDomain.h
//

#ifndef GVT_RENDER_DATA_DOMAIN_GEOMETRY_DOMAIN_H
#define GVT_RENDER_DATA_DOMAIN_GEOMETRY_DOMAIN_H

#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Light.h>

#include <iostream>
#include <vector>

namespace gvt {
    namespace render {
        namespace data {
            namespace domain {
                /// atomic geometry data unit for GraviT internal use
                /** Domain for geometry data. 
                \sa AbstractDomain, VolumeDomain
                */
                class GeometryDomain : public AbstractDomain 
                {
                public:
                   
                    GeometryDomain(std::string filename = "", gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true)) 
                    : AbstractDomain(m), mesh(NULL), filename(filename) 
                    {
                        if (filename != "") {
                            load();
                        }
                    }
                    
                    GeometryDomain(gvt::render::data::primitives::Mesh* mesh, 
                        gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true)) 
                    : AbstractDomain(m), mesh(NULL), filename("") 
                    {
                        if(mesh != NULL) {
                            this->mesh = mesh;
                            this->boundingBox = *(mesh->getBoundingBox());
                            isLoaded =true;
                        }
                    }
                    
                    
                    virtual ~GeometryDomain() {}

                    GeometryDomain(const GeometryDomain& other) : AbstractDomain(other) 
                    {
                        mesh = other.mesh;
                        lights = other.lights;
                        boundingBox = other.boundingBox;
                        filename = other.filename;
                        isLoaded = true;
                    }

                    virtual bool hasGeometry() 
                    { return isLoaded; }
                    
                    virtual int size() 
                    { return 0; }

                    virtual int sizeInBytes() 
                    { return 0; }

                    virtual std::vector<gvt::render::data::scene::Light*>& getLights()
                    { return lights; }

                    virtual void setLights(std::vector<gvt::render::data::scene::Light*>& l)
                    { lights = l; }

                    virtual gvt::render::data::primitives::Mesh* getMesh()
                    { return mesh; }

                    virtual bool load();
                    virtual void free();

                    friend std::ostream& operator<<(std::ostream&, GeometryDomain const&);

                    virtual void operator()() {}

                protected:
                    gvt::render::data::primitives::Mesh* mesh;
                    std::vector<gvt::render::data::scene::Light*> lights;
                    std::string filename;

                };
            }
        }
    }
}

#endif // GVT_RENDER_DATA_DOMAIN_GEOMETRY_DOMAIN_H
