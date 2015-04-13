/* 
 * File:   MetaProcessQueue.h
 * Author: jbarbosa
 *
 * Created on February 28, 2014, 1:49 PM
 */

#ifndef GVT_RENDER_ALGORITHM_META_PROCESS_QUEUE_H
#define	GVT_RENDER_ALGORITHM_META_PROCESS_QUEUE_H

#include <gvt/core/Debug.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/algorithm/AdaptParam.h>
#include <boost/thread.hpp>

namespace gvt {
    namespace render {
        namespace algorithm {
            
            template<class T> class ProcessQueue 
            {
              
                boost::shared_mutex _inqueue;
                boost::mutex _outqueue;
                
            public:
                AdaptParam<T>* param;

                ProcessQueue(AdaptParam<T>* param) 
                : param(param) 
                {}

                virtual ~ProcessQueue() 
                {
                    delete param;
                }

                void operator()() 
                {
                    GVT_ASSERT_BACKTRACE(false,"Not implemented");
                }
            protected:

                void IntersectDomain(gvt::render::actor::Ray& ray, gvt::render::actor::RayVector& newRays) 
                {
                    GVT_ASSERT_BACKTRACE(false,"Not implemented");
                }

                void TraverseDomain(gvt::render::actor::Ray& ray, gvt::render::actor::RayVector& newRays) 
                {
                    GVT_ASSERT_BACKTRACE(false,"Not implemented");
                }
            };
        }
    }
}

#endif	/* GVT_RENDER_ALGORITHM_META_PROCESS_QUEUE_H */

