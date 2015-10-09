/* 
 * File:   DomainSchedule.h
 * Author: jbarbosa
 *
 * Created on January 22, 2014, 7:56 PM
 */

#ifndef GVT_RENDER_SCHEDULE_DOMAIN_SCHEDULER_H
#define	GVT_RENDER_SCHEDULE_DOMAIN_SCHEDULER_H

#include <gvt/core/schedule/SchedulerBase.h>

namespace gvt{
	namespace render {
		namespace schedule {

			/// scheduler placeholder for Domain schedule
			/** \sa DomainTracer
			*/
			struct DomainScheduler: public SchedulerBase {

				DomainScheduler() : SchedulerBase() {}

				virtual ~DomainScheduler() {}

				virtual void operator()() 
				{
					GVT_ASSERT_BACKTRACE(false,"Domain Scheduler");
				}
			};
		}
	}
}

#endif	/* GVT_RENDER_SCHEDULE_DOMAIN_SCHEDULER_H */

