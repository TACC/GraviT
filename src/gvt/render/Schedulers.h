/* 
 * File:   schedulers.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:23 PM
 */

#ifndef GVT_RENDER_SCHEDULERS_H
#define	GVT_RENDER_SCHEDULERS_H

#include <gvt/core/schedule/SchedulerBase.h>
#include <gvt/render/schedule/hybrid/SpreadSchedule.h>
#include <gvt/render/schedule/hybrid/GreedySchedule.h>
#include <gvt/render/schedule/hybrid/RayWeightedSpreadSchedule.h>
#include <gvt/render/schedule/hybrid/AdaptiveSendSchedule.h>

#include <gvt/render/schedule/hybrid/LoadAnotherSchedule.h>
#include <gvt/render/schedule/hybrid/LoadAnyOnceSchedule.h>
#include <gvt/render/schedule/hybrid/LoadManySchedule.h>
#include <gvt/render/schedule/hybrid/LoadOnceSchedule.h>

#include <gvt/render/schedule/ImageScheduler.h>
#include <gvt/render/schedule/DomainScheduler.h>
#include <gvt/render/schedule/HybridScheduler.h>

#endif	/* GVT_RENDER_SCHEDULERS_H */

