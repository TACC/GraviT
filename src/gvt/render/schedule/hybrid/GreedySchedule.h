/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
/*
 * File:   GreedyScheduler.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:12 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_GREEDY_SCHEDULE_H
#define GVT_RENDER_SCHEDULE_HYBRID_GREEDY_SCHEDULE_H

#include <gvt/core/Debug.h>
#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

namespace gvt {
namespace render {
namespace schedule {
namespace hybrid {
/// hybrid schedule that distributes currently requested data across all processes with no sorting
/** This schedule simply allocates requested domains to processes, regardless of number of pending
rays or previously loaded data.

This schedule has the following issues:
    - domains with large ids tend to get starved, since the greedy loop iterates from small id to
    large id
    - a new allocation may move already-loaded data to another process, incurring excess data loads
    - processes may remain idle if there are fewer requested domains than processes
    - a domain can be assigned to a process that does not have rays queued for it, incurring excess
    ray sends

    \sa LoadAnyOnceSchedule, SpreadSchedule
    */
struct GreedySchedule : public HybridScheduleBase {

  GreedySchedule(int *newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
      : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) {}

  virtual ~GreedySchedule() {}

  virtual void operator()() {
    for (int i = 0; i < size; ++i) newMap[i] = -1;

    gvt::core::Map<int, int> data2proc;
    for (int s = 0; s < size; ++s) {
      if (map_recv_bufs[s]) {
        // greedily grab next unclaimed domain
        for (int d = 1; d < map_size_buf[s]; d += 2) {
          if (data2proc.find(map_recv_bufs[s][d]) == data2proc.end()) {
            newMap[s] = map_recv_bufs[s][d];
            data2proc[map_recv_bufs[s][d]] = s;
          }
        }
      }
    }

    

  }
};
}
}
}
}

#endif /* GVT_RENDER_SCHEDULE_HYBRID_GREEDY_SCHEDULE_H */
