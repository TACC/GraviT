/* 
 * File:   DomainSchedule.h
 * Author: jbarbosa
 *
 * Created on January 22, 2014, 7:56 PM
 */

#ifndef GVT_DOMAINSCHEDULE_H
#define	GVT_DOMAINSCHEDULE_H

#include "scheduler.h"


struct DomainSchedule: public SchedulerBase {

    DomainSchedule() : SchedulerBase() {
        
    }

    virtual ~DomainSchedule() {
        
    }
    
    virtual void operator()() {
        GVT_ASSERT_BACKTRACE(false,"Domain Schedule");
    }
};

#endif	/* GVT_DOMAINSCHEDULE_H */

