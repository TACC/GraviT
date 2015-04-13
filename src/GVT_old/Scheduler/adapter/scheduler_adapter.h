/* 
 * File:   sched_adapter.h
 * Author: jbarbosa
 *
 * Created on February 18, 2014, 1:38 PM
 */

#ifndef SCHED_ADAPTER_H
#define	SCHED_ADAPTER_H

#include "scheduler_base_adapter.h"

template<class DomainType>
struct scheduler_adapter : public scheduler_base_adapter {
    std::map<int, RayVector> &queue;
    int &domTarget;
    RayVector &moved_rays;
    long &ray_counter;
    RayTracerAttributes& rta;
    ColorAccumulator* colorBuf;
    long &domain_counter;

    scheduler_adapter(
            std::map<int, RayVector> &queue,
            int &domTarget,
            RayVector &moved_rays,
            long &ray_counter,
//            RayTracerAttributes& rta,
            ColorAccumulator* colorBuf,
            long &domain_counter) : scheduler_base_adapter(),
    queue(queue), domTarget(domTarget), moved_rays(moved_rays),
    ray_counter(ray_counter), rta(rta), colorBuf(colorBuf),
    domain_counter(domain_counter) 
    {

    }



    virtual ~scheduler_adapter() {
        
    }

};

#endif	/* SCHED_ADAPTER_H */

