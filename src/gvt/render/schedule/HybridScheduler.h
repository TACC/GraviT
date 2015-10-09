/* 
 * File:   HybridSchedule.h
 * Author: jbarbosa
 *
 * Created on January 22, 2014, 7:56 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_SCHEDULER_H
#define	GVT_RENDER_SCHEDULE_HYBRID_SCHEDULER_H

#include <gvt/core/schedule/SchedulerBase.h>

 namespace gvt {
 	namespace render {
 		namespace schedule {

			/// scheduler placeholder for Hybrid schedule
			/** \sa HybridTracer
			*/
			template<class SCHEDULER>
 			struct HybridScheduler: public SchedulerBase 
 			{


 				HybridScheduler() : SchedulerBase() {}

 				virtual ~HybridScheduler() {}

 				virtual void operator()() 
 				{
 					GVT_ASSERT_BACKTRACE(false,"Hybrid Scheduler"); 
 				}
 			};
 		}
 	}
 }


#endif	/* GVT_RENDER_SCHEDULE_HYBRID_SCHEDULER_H */

