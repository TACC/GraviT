/*
 * File:   BBox.h
 * Author: jbarbosa
 *
 * Created on February 27, 2014, 2:30 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_BBOX_H
#define	GVT_RENDER_DATA_PRIMITIVES_BBOX_H

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <fstream>

 namespace gvt {
    namespace render {
        namespace data {
            namespace primitives {
                /// bounding box for data and acceleration structures
                class Box3D {
                public:
                    gvt::core::math::Point4f bounds[2];

                    Box3D();
                    Box3D(gvt::core::math::Point4f vmin,  gvt::core::math::Point4f vmax);

                    Box3D(const Box3D &other);
                    bool intersect(const gvt::render::actor::Ray &r) const;
                    bool intersect(const gvt::render::actor::Ray &r, float& tmin, float& tmax) const;
                    bool inBox(const gvt::render::actor::Ray &r) const;
                    bool inBox(const gvt::core::math::Point4f &r) const;
                    gvt::core::math::Point4f getHitpoint(const gvt::render::actor::Ray& r) const;
                    bool intersectDistance(const gvt::render::actor::Ray& r, float& t) const;
                    void merge(const Box3D &other);
                    void expand(gvt::core::math::Point4f& v);
                    int wideRangingBoxDir() const;
                    gvt::core::math::Point4f centroid() const;
                    float surfaceArea() const;

                    friend std::ostream & operator <<(std::ostream &os, const Box3D &bbox) 
                    {
                        os << bbox.bounds[0] << " x ";
                        os << bbox.bounds[1];
                        return os;
                    }       

                    template<typename cast>
                    operator cast() 
                    {
                        GVT_ASSERT(false,"Cast operator not available from gvt::render::data::primitives::BBox");
                    }
                };
            }
        }
    }
}
#endif	/* GVT_RENDER_DATA_PRIMITIVES_BBOX_H */

