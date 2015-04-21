/* 
 * File:   SchedulerBase.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:07 PM
 */

#ifndef GVT_CORE_SCHEDULE_SCHEDULER_BASE_H
#define	GVT_CORE_SCHEDULE_SCHEDULER_BASE_H

#include <gvt/core/Debug.h>

class SchedulerBase 
{
public:
    SchedulerBase() {}

    virtual ~SchedulerBase() {}
    
    virtual void operator()() 
    {
        GVT_ASSERT_BACKTRACE(false,"schedule not implemented");
    }
};
#endif	/* GVT_CORE_SCHEDULE_SCHEDULER_BASE_H */

