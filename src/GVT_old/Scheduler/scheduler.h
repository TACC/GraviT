/* 
 * File:   scheduler.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:07 PM
 */

#ifndef GVT_SCHEDULER_H
#define	GVT_SCHEDULER_H


struct SchedulerBase {
    
    
    SchedulerBase() {
        
    }

    virtual ~SchedulerBase() {
        
    }
    
    virtual void operator()() {
        GVT_ASSERT_BACKTRACE(false,"Not schedule implemented");
    }
};
#endif	/* GVT_SCHEDULER_H */

