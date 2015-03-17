/* 
 * File:   ImageSchedule.h
 * Author: jbarbosa
 *
 * Created on January 22, 2014, 7:56 PM
 */

#ifndef GVT_RENDER_SCHEDULE_IMAGE_SCHEDULER_H
#define	GVT_RENDER_SCHEDULE_IMAGE_SCHEDULER_H

#include <gvt/core/schedule/SchedulerBase.h>

 namespace gvt {
 	namespace render {
 		namespace schedule {

 			struct ImageScheduler: public SchedulerBase 
 			{
 				
 				ImageScheduler() : SchedulerBase() {}

 				virtual ~ImageScheduler() {}
 				
 				virtual void operator()() 
 				{
 					GVT_ASSERT_BACKTRACE(false,"Image Scheduler");
 				}
 			};
 		}
 	}
 }

#endif	/* GVT_RENDER_SCHEDULE_IMAGE_SCHEDULER_H */

