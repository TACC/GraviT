/* 
 * File:   adapt_param.h
 * Author: jbarbosa
 *
 * Created on December 18, 2013, 4:33 PM
 */

#ifndef GVT_ADAPT_PARAM_H
#define	GVT_ADAPT_PARAM_H

#include <GVT/common/debug.h>
#include <GVT/Environment/RayTracerAttributes.h>
#include <GVT/Domain/domains.h>

namespace GVT {
    namespace Backend {

        template<class T>
        struct adapt_param {
            std::map<int, GVT::Data::RayVector>& queue;
            GVT::Data::RayVector& moved_rays;
            int domTarget;

            T* dom;
            //GVT::Env::RayTracerAttributes& rta;
            ColorAccumulator* colorBuf;

            long& ray_counter;
            long& domain_counter;

            adapt_param(std::map<int, GVT::Data::RayVector>& queue, GVT::Data::RayVector& moved_rays,
                    int domTarget, GVT::Domain::Domain* dom,
                    ColorAccumulator* colorBuf, long& ray_counter, long& domain_counter) :

            queue(queue), moved_rays(moved_rays), domTarget(domTarget), dom((T*)
            dom), colorBuf(colorBuf), ray_counter(
            ray_counter), domain_counter(domain_counter) {

            }


        };
    };
};


#endif	/* GVT_ADAPT_PARAM_H */

