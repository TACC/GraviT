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
 * File:   Spread.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:14 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_SPREAD_SCHEDULE_H
#define GVT_RENDER_SCHEDULE_HYBRID_SPREAD_SCHEDULE_H

#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

namespace gvt {
namespace render {
namespace schedule {
namespace hybrid {
/// hybrid schedule that distributes requested data across available processes
/** This schedule simply allocates requested domains to availalbe processes,
regardless of number of pending rays, where a process is 'available' if none of its loaded data
is currently requested by any ray.

This schedule has the following issues:
    - domains with large ids tend to get starved, since the greedy loop iterates
      from small id to large id
    - processes may remain idle if there are fewer requested domains than processes
    - a domain can be assigned to a process that does not have rays queued for it,
      incurring excess ray sends

    \sa LoadAnyOnceSchedule, GreedySchedule, RayWeightedSpreadSchedule
    */
struct SpreadSchedule : public HybridScheduleBase {

  SpreadSchedule(int *newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
      : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) {}

  virtual ~SpreadSchedule() {}

  virtual void operator()() {

    

    for (int i = 0; i < size; ++i) newMap[i] = -1;

    gvt::core::Map<int, int> data2proc;
    gvt::core::Vector<int> queued;
    for (int s = 0; s < size; ++s) {
      if (map_recv_bufs[s]) {
        // add currently loaded data
        data2proc[map_recv_bufs[s][0]] = s; // this will evict previous entries, that's okay (I think)
        

        // add queued data
        for (int d = 1; d < map_size_buf[s]; d += 2) queued.push_back(map_recv_bufs[s][d]);
      }
    }

    // iterate over queued data, find which are already loaded somewhere
    gvt::core::Vector<int> homeless;
    for (size_t i = 0; i < queued.size(); ++i) {
      gvt::core::Map<int, int>::iterator it = data2proc.find(queued[i]);
      if (it != data2proc.end()) {
        newMap[it->second] = it->first;
        
      } else {
        homeless.push_back(queued[i]);
        
      }
    }

    // iterate over newMap, fill as many procs as possible with homeless data
    // could be dupes in the homeless list, so keep track of what's added
    for (int i = 0; (i < size) & (!homeless.empty()); ++i) {
      if (newMap[i] < 0) {
        while (!homeless.empty() && data2proc.find(homeless.back()) != data2proc.end()) homeless.pop_back();
        if (!homeless.empty()) {
          newMap[i] = homeless.back();
          data2proc[newMap[i]] = i;
          homeless.pop_back();
        }
      }
    }

    
  }
};
}
}
}
}

#endif /* GVT_RENDER_SCHEDULE_HYBRID_SPREAD_SCHEDULE_H */
