/* 
 * File:   ImageSchedule.h
 * Author: jbarbosa
 *
 * Created on January 22, 2014, 7:56 PM
 */

#ifndef GVT_IMAGESCHEDULE_H
#define	GVT_IMAGESCHEDULE_H

#include "scheduler.h"


struct ImageSchedule: public SchedulerBase {
    
    
    ImageSchedule() : SchedulerBase() {
        
    }

    virtual ~ImageSchedule(){
        
    }
    
    virtual void operator()() {
        GVT_ASSERT_BACKTRACE(false,"Image Schedule");
    }
};

#endif	/* GVT_IMAGESCHEDULE_H */

