/* 
 * File:   AdaptParam.h
 * Author: jbarbosa
 *
 * Created on December 18, 2013, 4:33 PM
 */

#ifndef GVT_RENDER_ALGORITHM_ADAPT_PARAM_H
#define	GVT_RENDER_ALGORITHM_ADAPT_PARAM_H

#include <gvt/core/Debug.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/Domains.h>

namespace gvt {
    namespace render {
        namespace algorithm {
            
            template<class T> struct AdaptParam 
            {
                std::map<int, gvt::render::actor::RayVector>& queue;
                gvt::render::actor::RayVector& moved_rays;
                int domTarget;

                T* dom;
                //GVT::Env::RayTracerAttributes& rta;
                gvt::render::data::scene::ColorAccumulator* colorBuf;

                long& ray_counter;
                long& domain_counter;

                AdaptParam(std::map<int, 
                    gvt::render::actor::RayVector>& queue, 
                    gvt::render::actor::RayVector& moved_rays,
                    int domTarget, 
                    gvt::render::data::domain::AbstractDomain* dom,
                    gvt::render::data::scene::ColorAccumulator* colorBuf, 
                    long& ray_counter, 
                    long& domain_counter) 
                : queue(queue), moved_rays(moved_rays), domTarget(domTarget), dom((T*)dom), 
                colorBuf(colorBuf), ray_counter(ray_counter), domain_counter(domain_counter) 
                {}
            };
        }
    }
}


#endif	/* GVT_RENDER_ALGORITHM_ADAPT_PARAM_H */

