
#ifndef GVT_RENDER_ADAPTER_EMBREE_DATA_TRANSFORMS_H
#define	GVT_RENDER_ADAPTER_EMBREE_DATA_TRANSFORMS_H


#include <gvt/core/data/Transform.h>

#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/Primitives.h>

#include <vector>

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

namespace gvt {
    namespace render {
        namespace adapter {
            namespace embree {
                namespace data {
                    GVT_TRANSFORM_TEMPLATE; // see gvt/core/data/Transform.h

                    template<>
                    struct transform_impl<float[3], gvt::core::math::Point4f >
                    {
                        static inline gvt::core::math::Vector4f transform(const float r[3])
                        {
                            return gvt::core::math::Point4f(r[0], r[1], r[2], 1.f);
                        }
                    };


                    template<>
                    struct transform_impl<float[3], gvt::core::math::Vector4f >
                    {
                        static inline gvt::core::math::Vector4f transform(const float r[3])
                        {
                            return gvt::core::math::Point4f(r[0], r[1], r[2], 0.f);
                        }
                    };


                    template<>
                    struct transform_impl<gvt::render::actor::Ray, RTCRay>
                    {
                        static inline RTCRay transform(const gvt::render::actor::Ray& r)
                        {
                            RTCRay ray;
                            ray.org[0] = r.origin[0];
                            ray.org[1] = r.origin[1];
                            ray.org[2] = r.origin[2];
                            ray.dir[0] = r.direction[0];
                            ray.dir[1] = r.direction[1];
                            ray.dir[2] = r.direction[2];
                            return ray;
                        }
                    };

                    template<>
                    struct transform_impl<RTCRay, gvt::render::actor::Ray>
                    {
                        static inline gvt::render::actor::Ray transform(const RTCRay& r)
                        {
                            gvt::core::math::Point4f o(r.org[0], r.org[1], r.org[2], 1);
                            gvt::core::math::Vector4f d(r.dir[0], r.dir[1], r.dir[2], 0);

                            return gvt::render::actor::Ray(
                                    gvt::render::adapter::embree::data::transform<float[3], gvt::core::math::Point4f>(r.org),
                                    gvt::render::adapter::embree::data::transform<float[3], gvt::core::math::Vector4f>(r.dir));
                        }
                    };

                }
            }
        }
    }
}

#endif	/* GVT_RENDER_ADAPTER_EMBREE_DATA_TRANSFORMS_H */

