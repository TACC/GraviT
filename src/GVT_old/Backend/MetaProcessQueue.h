/* 
 * File:   MetaProcessQueue.h
 * Author: jbarbosa
 *
 * Created on February 28, 2014, 1:49 PM
 */

#ifndef METAPROCESSQUEUE_H
#define	METAPROCESSQUEUE_H

#include <GVT/common/debug.h>
#include <boost/thread.hpp>
#include <GVT/Environment/RayTracerAttributes.h>
#include <GVT/Domain/domains.h>
#include "adapt_param.h"

namespace GVT {
    namespace Backend {

        template<class T>
        class ProcessQueue {
            //	std::map<int, GVT::Data::RayVector>& queue;
            //	GVT::Data::RayVector& moved_rays;
            //	int domTarget;
            //
            //	T* dom /* XXX TODO fixme */;
            //	RayTracerAttributes& rta;
            //	ColorAccumulator* colorBuf;
            //	long& ray_counter;
            //	long& domain_counter;

            //float &sample_ratio;
            //unsigned char*& vtf;
            
            boost::shared_mutex _inqueue;
            boost::mutex _outqueue;
            
        public:
            adapt_param<T>* param;

            ProcessQueue(adapt_param<T>* param) : param(param) {

            }

            virtual ~ProcessQueue() {
                delete param;
            }

//            virtual GVT::Data::ray& get(GVT::Data::RayVector &queue, int idx) {
//                boost::shared_lock<boost::shared_mutex> _lock(_inqueue);
//                return queue[idx];
//            }
//            
//            virtual GVT::Data::ray& pop(GVT::Data::RayVector &queue) {
//                boost::upgrade_lock<boost::shared_mutex> lock(_inqueue);
//                boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
//                if(queue.empty()) return NULL;
//                GVT::Data::ray& ray = queue.back();
//                queue.pop_back();
//                return ray;
//                
//            }
//                        
//            virtual void push(GVT::Data::RayVector &queue, GVT::Data::ray& r) {
//                boost::upgrade_lock<boost::shared_mutex> lock(_inqueue);
//                boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
//                queue.push_back(r);
//            }
//            
//            virtual void dispatch(GVT::Data::RayVector &queue, GVT::Data::ray& r) {
//                boost::lock_guard<boost::mutex> _lock(_outqueue);
//                queue.push_back(r);
//            }

            void operator()() {
                //		/DEBUG("Not implemented");
                GVT_ASSERT_BACKTRACE(false,"Not implemented");
            }
        protected:

            void IntersectDomain(GVT::Data::ray& ray, GVT::Data::RayVector& newRays) {
                //DBG_BACKTRACE("Not implemented");
                //std::cout << "Not implemented" << std::endl;
                GVT_ASSERT_BACKTRACE(false,"Not implemented");
            }

            void TraverseDomain(GVT::Data::ray& ray, GVT::Data::RayVector& newRays) {
                //		/DBG_BACKTRACE("Not implemented");
                //		std::cout << "Not implemented" << std::endl;
                GVT_ASSERT_BACKTRACE(false,"Not implemented");
            }
        };

    };
};
#endif	/* METAPROCESSQUEUE_H */

