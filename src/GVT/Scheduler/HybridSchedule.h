/* 
 * File:   HybridSchedule.h
 * Author: jbarbosa
 *
 * Created on January 22, 2014, 7:56 PM
 */

#ifndef GVT_HYBRIDSCHEDULE_H
#define	GVT_HYBRIDSCHEDULE_H

#include "scheduler.h"

#include <GVT/Tracer/Tracer.h>

template<class SCHEDULER>
struct HybridSchedule: public SchedulerBase {

    
    HybridSchedule() : SchedulerBase() {
        
    }

    virtual ~HybridSchedule(){
        
    }
    
    virtual void operator()() {
    }
};



#endif	/* GVT_HYBRIDSCHEDULE_H */

