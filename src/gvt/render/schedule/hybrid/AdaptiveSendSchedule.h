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
 * File:   AdaptiveSend.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:18 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_ADAPTIVE_SEND_H
#define GVT_RENDER_SCHEDULE_HYBRID_ADAPTIVE_SEND_H

#include <gvt/core/Debug.h>
#include <gvt/core/Types.h>
#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

namespace gvt {
namespace render {
namespace schedule {
namespace hybrid {
/// hybrid schedule that attempts to load multiple data copies in response to ray load
/** This schedule attempts to detect high ray demand for particular data
and loads multiple copies of that data in order to balance ray load across multiple processes.

The current implementation is not particularly successful at this. Issues include:
    - data loads typically occur at remote processes, incurring data send costs
    - demand detection algorith needs improvement, particularly ray demand threshold over
    which additional data isloaded
    - eviction logic could be improved: at present the algorithm prefers to keep loaded data
    if even one ray needs it
*/
struct AdaptiveSendSchedule : public HybridScheduleBase {

  AdaptiveSendSchedule(int *newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
      : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) {}

  virtual ~AdaptiveSendSchedule() {}

  virtual void operator()() {
    
    for (int i = 0; i < size; ++i) newMap[i] = -1;

    gvt::core::Map<int, gvt::core::Vector<int> > cur_data2procs;
    gvt::core::Map<int, gvt::core::Vector<int> > send_data;
    gvt::core::Map<int, int> data2size;
    gvt::core::Map<int, int> procs_count;
    for (int s = 0; s < size; ++s) {
      if (map_recv_bufs[s]) {
        // add currently loaded data
        // track number of procs that have data loaded
        cur_data2procs[map_recv_bufs[s][0]].push_back(s);
        procs_count[map_recv_bufs[s][0]] += 1;
        

        // add data with queued rays
        // track number of rays needed (assumes map inits to zero)
        for (int d = 1; d < map_size_buf[s]; d += 2) {
          data2size[map_recv_bufs[s][d]] += map_recv_bufs[s][d + 1];
                  }
      }
    }

    // put data into priority buckets
    // based on number of rays waiting for data
    gvt::core::Map<int, gvt::core::Vector<int> > priority;
    gvt::core::Map<int, gvt::core::Vector<int> > zero_priority;
    // int add_a_proc = atts.GetNewProcThreshold();
    int add_a_proc = 5000; // XXX TODO: hacked!
    int avail = 0;
    for (gvt::core::Map<int, int>::iterator it = data2size.begin(); it != data2size.end(); ++it) {
      int idx = it->second / add_a_proc;
      
      if (idx > 0) {
        priority[idx].push_back(it->first);
      } else {
        gvt::core::Map<int, int>::iterator pdit = procs_count.find(it->first);
        if (pdit != procs_count.end()) {
          for (int i = 0; i < pdit->second; ++i) {
            

            zero_priority[i].push_back(cur_data2procs[pdit->first][i]); // push proc id onto stack, let
                                                                        // map sort into use-first
                                                                        // order
          }
          avail += pdit->second;
          
        }
      }
    }

    // track procs that have data with no pending rays
    // give them artificially high sort order so they'll be used first
    if (!priority.empty()) {
      for (gvt::core::Map<int, gvt::core::Vector<int> >::iterator it = cur_data2procs.begin(); it != cur_data2procs.end(); ++it) {
        if (data2size.find(it->first) == data2size.end()) {
          int pc = procs_count[it->first];
          for (int i = 0; i < pc; ++i) {
            
            zero_priority[(pc << 6) + i].push_back(it->second[i]);
          }
          avail += pc;
          
        }
      }
    }

    
    
    
    

    // while procs are available
    // get highest priority and allocate procs
    // gvt::core::Vector<int> excess;  // reclaim excess procs? or reclaim them when become zero priority?
    
    for (gvt::core::Map<int, gvt::core::Vector<int> >::reverse_iterator rit = priority.rbegin();
         (rit != priority.rend()) & (avail > 0); ++rit) {
      for (gvt::core::Vector<int>::iterator it = rit->second.begin(); (it != rit->second.end()) & (avail > 0); ++it) {
        int need = rit->first - procs_count[*it];
        
        if (procs_count[*it] > 0) {
          
          // add already-allocated procs
          int used;
          for (used = 0; (used < procs_count[*it]) & (need > 0); ++used) {
            int p = cur_data2procs[*it][used];
            newMap[p] = *it;
            --need;
            
          }

          // put remaining processor into empty queue,
          // with high sort order so they get used first
          if (used < procs_count[*it]) {
            int rem = (procs_count[*it] - used) << 10;
            for (int i = used; i < procs_count[*it]; ++i) {
              zero_priority[rem + i].push_back(cur_data2procs[*it].back());
              cur_data2procs[*it].pop_back();
              ++avail;
            }
            
          } else {
            need = (need > avail) ? avail : (need > 0) ? need : 0;
            avail -= need;
            
            // data already loaded, put rest in send queue
            gvt::core::Map<int, gvt::core::Vector<int> >::reverse_iterator zpi = zero_priority.rbegin();
            for (int i = 0; i < need; ++i) {
              int victim = zpi->second.back();
              newMap[victim] = *it;
              send_data[*it].push_back(victim);
              zpi->second.pop_back();
              if (zpi->second.empty()) {
                ++zpi;
              }
              
            }
            if (zpi != zero_priority.rbegin()) {
              --zpi;
              int target = zpi->first;
              
              zero_priority.erase(zero_priority.find(target), zero_priority.end());
            }
          }
        } else {
          // not loaded anywhere.  Load on one, put rest in to_send queue
          // mark to load on one
          need = (need > avail) ? avail : (need > 0) ? need : 0;
          avail -= need;

          
          gvt::core::Map<int, gvt::core::Vector<int> >::reverse_iterator zpi = zero_priority.rbegin();
          int victim = zpi->second.back();
          newMap[victim] = *it;
          cur_data2procs[*it].push_back(victim); // this proc will load the data, then send to the rest
          zpi->second.pop_back();
          if (zpi->second.empty()) {
            ++zpi;
          }
          

          // put rest in send queue
          // start from 1, to account for first just added
          for (int i = 1; i < need; ++i) {
            victim = zpi->second.back();
            newMap[victim] = *it;
            send_data[*it].push_back(victim);
            zpi->second.pop_back();
            if (zpi->second.empty()) {
              ++zpi;
            }
            
          }
          if (zpi != zero_priority.rbegin()) {
            --zpi;
            int target = zpi->first;
            
            zero_priority.erase(zero_priority.find(target), zero_priority.end());
          }
        }
      }
    }

    

    // allocate zero-priority if that's all that's present
    // use adaptive allocation scheme
    if (priority.empty()) {
      

      gvt::core::Map<int, int> size2data;
      // convert data2size into size2data,
      // use data id to pseudo-uniqueify, since only need ordering
      for (gvt::core::Map<int, int>::iterator it = data2size.begin(); it != data2size.end(); ++it) {
        size2data[(it->second << 7) + it->first] = it->first;
      }

      // iterate over queued data, find which are already loaded somewhere
      // if there are multiple procs with one data, just keep one
      gvt::core::Vector<int> homeless;
      for (gvt::core::Map<int, int>::iterator d2sit = size2data.begin(); d2sit != size2data.end(); ++d2sit) {
        gvt::core::Map<int, gvt::core::Vector<int> >::iterator d2pit = cur_data2procs.find(d2sit->second);
        if (d2pit != cur_data2procs.end()) {
          newMap[d2pit->second[0]] = d2pit->first;
          
        } else {
          homeless.push_back(d2sit->second);
          
        }
      }

      // iterate over newMap, fill as many procs as possible with homeless data
      // could be dupes in the homeless list, so keep track of what's added
      for (int i = 0; (i < size) & (!homeless.empty()); ++i) {
        if (newMap[i] < 0) {
          while (!homeless.empty() && cur_data2procs.find(homeless.back()) != cur_data2procs.end()) homeless.pop_back();
          if (!homeless.empty()) {
            newMap[i] = homeless.back();
            cur_data2procs[newMap[i]].push_back(i);
            homeless.pop_back();
            
          }
        }
      }
    }

    // build data send buffer
    // index is proc to receive, value is proc to send
    
    for (gvt::core::Map<int, gvt::core::Vector<int> >::iterator it = send_data.begin(); it != send_data.end(); ++it) {
      int to_recv = it->second.size();
      int to_send = cur_data2procs[it->first].size();
      
      for (int r = 0, s = 0; r < to_recv; ++r, ++s) {
        s = r % to_send; // XXX TODO pnav: 's' is likely an error here, but confirm and fix
        
        data_send_buf[it->second[r]] = cur_data2procs[it->first][s];
      }
    }

    

  }
};
}
}
}
}

#endif /* GVT_RENDER_SCHEDULE_HYBRID_ADAPTIVE_SEND_H */
