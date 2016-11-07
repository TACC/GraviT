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
 * File:   LoadOnceSchedule.h
 * Author: pnav
 *
 * Created on February 4, 2014, 3:41 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_LOAD_ONCE_SCHEDULE_H
#define GVT_RENDER_SCHEDULE_HYBRID_LOAD_ONCE_SCHEDULE_H

#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

namespace gvt {
namespace render {
namespace schedule {
namespace hybrid {
/// hybrid schedule that attempts to load domains that have high demand
/** This schedule attempts to load domains that have high ray demand to any available processes,
where a process is 'available' if none of its loaded data is currently requested by any ray.
This schedule checks whether the domain is already loaded at a process, and if so,
does not load it again.

\sa LoadAnotherSchedule, LoadAnyOnceSchedule, LoadManySchedule
*/
struct LoadOnceSchedule : public HybridScheduleBase {

  LoadOnceSchedule(int *newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
      : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) {}

  virtual ~LoadOnceSchedule() {}

  virtual void operator()() {
    
    for (int i = 0; i < size; ++i) newMap[i] = -1;

    gvt::core::Map<int, int> data2proc;
    gvt::core::Map<int, int> data2size;
    gvt::core::Map<int, int> size2data;
    for (int s = 0; s < size; ++s) {
      if (map_recv_bufs[s]) {
        // add currently loaded data
        data2proc[map_recv_bufs[s][0]] = s; // this will evict previous entries.
                                            // that's okay since we don't want
                                            // to dup data
        

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

    // iterate over queued data, find which are already loaded somewhere
    // since size2data is sorted in increasing key order,
    // homeless data with most rays will end up at top of homeless list
    gvt::core::Vector<int> homeless;
    for (gvt::core::Map<int, int>::iterator d2sit = size2data.begin(); d2sit != size2data.end(); ++d2sit) {
      gvt::core::Map<int, int>::iterator d2pit = data2proc.find(d2sit->second);
      if (d2pit != data2proc.end()) {
        newMap[d2pit->second] = d2pit->first;
        
      } else {
        homeless.push_back(d2sit->second);
        
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

#endif /* GVT_RENDER_SCHEDULE_HYBRID_LOAD_ONCE_SCHEDULE_H */
