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
 * File:   LoadAnyOnceSchedule.h
 * Author: pnav
 *
 * if a domain has a bunch of rays, add the domain to an empty process.
 * this starts with the adaptive LoadOnce schedule and removes the check about whether
 * a domain is already loaded and clears the map so at present, it adds the domain to a process
 * that has the most rays pending, which might or might not be a domain already loaded, and it
 * loads it anywhere, regardless of whether what might have been loaded on the process previously
 *
 * TODO: starts at proc 0 each time, might get better balance if jiggle the start point
 *
 * Created on February 4, 2014, 3:41 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_LOAD_ANY_ONCE_SCHEDULE_H
#define GVT_RENDER_SCHEDULE_HYBRID_LOAD_ANY_ONCE_SCHEDULE_H

#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

namespace gvt {
namespace render {
namespace schedule {
namespace hybrid {
/// hybrid schedule that attempts to load the domains that have highest demand to
/** This schedule attempts to load domains that have high ray demand to all processes. This
proceeds greedily without regard to whether or where data is already loaded. This is similar
to the Greedy schedule, but the requested domains are sorted by number of pending rays before
assignment. Thus, the domain with the most pending rays is always included in the process set.

This schedule has the following issues:
    - domains are assigned to processes without regard to current domain assignments, which can
    incur extra data loads
    - domains are assigned without regard to current ray locations, which can incur extra ray sends

\sa LoadOnceSchedule, LoadAnotherSchedule, LoadManySchedule, GreedySchedule
*/
struct LoadAnyOnceSchedule : public HybridScheduleBase {

  LoadAnyOnceSchedule(int *newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
      : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) {}

  virtual ~LoadAnyOnceSchedule() {}

  virtual void operator()() {
    
    for (int i = 0; i < size; ++i) newMap[i] = -1; // clear map

    gvt::core::Map<int, int> data2proc;
    gvt::core::Map<int, int> data2size;
    gvt::core::Map<int, int> size2data;
    for (int s = 0; s < size; ++s) {
      if (map_recv_bufs[s]) {
        // add currently loaded data
        data2proc[map_recv_bufs[s][0]] = s; // this will evict previous entries.
                                            // that's okay since we don't want
                                            // to dup data (here)
        

        // add ray counts
        for (int d = 1; d < map_size_buf[s]; d += 2) {
          data2size[map_recv_bufs[s][d]] += map_recv_bufs[s][d + 1];
          
        }
      }
    }

    // convert data2size into size2data,
    // use data id to pseudo-uniqueify, since only need ordering
    for (gvt::core::Map<int, int>::iterator it = data2size.begin(); it != data2size.end(); ++it) {
      size2data[(it->second << 7) + it->first] = it->first;
    }

    // iterate over queued data, find which procs have the most rays
    // since size2data is sorted in increasing key order,
    // data that has the most rays pending will end up at the top of the bloated list
    gvt::core::Vector<int> bloated;
    for (gvt::core::Map<int, int>::iterator s2dit = size2data.begin(); s2dit != size2data.end(); ++s2dit) {
      bloated.push_back(s2dit->second);
      
    }

    // iterate over newMap, fill as many procs as possible with homeless data
    // could be dupes in the homeless list, so keep track of what's added
    for (int i = 0; (i < size) & (!bloated.empty()); ++i) {
      if (!bloated.empty()) {
        newMap[i] = bloated.back();
        data2proc[newMap[i]] = i;
        bloated.pop_back();
      }
    }

    
  }
};
}
}
}
}

#endif /* GVT_RENDER_SCHEDULE_HYBRID_LOAD_ANY_ONCE_SCHEDULE_H */
