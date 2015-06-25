/*
 * File:   LoadAnyOnceSchedule.h
 * Author: pnav
 *
 * if a domain has a bunch of rays, add the domain to an empty process.
 * this starts with the adaptive LoadOnce schedule and removes the check about whether a domain is already loaded and clears the map
 * so at present, it adds the domain to a process that has the most rays pending, which might or might not be a domain already loaded, and it loads it anywhere, regardless of whether what might have been loaded on the process previously
 *
 * TODO: starts at proc 0 each time, might get better balance if jiggle the start point
 *
 * Created on February 4, 2014, 3:41 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_LOAD_ANY_ONCE_SCHEDULE_H
#define	GVT_RENDER_SCHEDULE_HYBRID_LOAD_ANY_ONCE_SCHEDULE_H

#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

#ifdef __USE_TAU
#include <TAU.h>
#endif

namespace gvt {
    namespace render {
        namespace schedule {
            namespace hybrid {
                 struct LoadAnyOnceSchedule : public HybridScheduleBase
                 {

                    LoadAnyOnceSchedule(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
                    : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf)
                    {}

                    virtual ~LoadAnyOnceSchedule() {}

                    virtual void operator()()
                    {
#ifdef __USE_TAU
 TAU_START("LoadAnyOnceSchedule.h.operator");
#endif
                        GVT_DEBUG(DBG_LOW,"in LoadAnyOnce schedule");
                        for (int i = 0; i < size; ++i)
                            newMap[i] = -1; // clear map

                        std::map<int, int> data2proc;
                        std::map<int, int> data2size;
                        std::map<int, int> size2data;
                        for (int s = 0; s < size; ++s)
                        {
                            if (map_recv_bufs[s])
                            {
                                // add currently loaded data
                                data2proc[map_recv_bufs[s][0]] = s; // this will evict previous entries. that's okay since we don't want to dup data (here)
                                GVT_DEBUG(DBG_LOW,"    noting currently " << s << " -> " << map_recv_bufs[s][0]);

                                // add ray counts
                                for (int d = 1; d < map_size_buf[s]; d += 2)
                                {
                                    data2size[map_recv_bufs[s][d]] += map_recv_bufs[s][d + 1];
                                    GVT_DEBUG(DBG_LOW,"        " << s << " has " << map_recv_bufs[s][d + 1] << " rays for data " << map_recv_bufs[s][d]);
                                }
                            }
                        }

                        // convert data2size into size2data,
                        // use data id to pseudo-uniqueify, since only need ordering
                        for (std::map<int, int>::iterator it = data2size.begin(); it != data2size.end(); ++it)
                        {
                            size2data[(it->second << 7) + it->first] = it->first;
                        }

                        // iterate over queued data, find which procs have the most rays
                        // since size2data is sorted in increasing key order,
                        // data that has the most rays pending will end up at the top of the bloated list
                        std::vector<int> bloated;
                        for (std::map<int, int>::iterator s2dit = size2data.begin(); s2dit != size2data.end(); ++s2dit)
                        {
                            bloated.push_back(s2dit->second);
                            GVT_DEBUG(DBG_LOW,"    noting domain " << s2dit->second << " is bloated with size " << s2dit->first);
                        }

                        // iterate over newMap, fill as many procs as possible with homeless data
                        // could be dupes in the homeless list, so keep track of what's added
                        for (int i = 0; (i < size) & (!bloated.empty()); ++i)
                        {
                            if (!bloated.empty()) {
                                newMap[i] = bloated.back();
                                data2proc[newMap[i]] = i;
                                bloated.pop_back();
                            }
                        }

                        GVT_DEBUG_CODE(DBG_LOW,
                            std::cerr << "new map size is " << size << std::endl;
                            for (int i = 0; i < size; ++i)
                                std::cerr << "    " << i << " -> " << newMap[i] << std::endl;
                            );
#ifdef __USE_TAU
 TAU_STOP("LoadAnyOnceSchedule.h.operator");
#endif

                    }
                };
            }
        }
    }
}

#endif	/* GVT_RENDER_SCHEDULE_HYBRID_LOAD_ANY_ONCE_SCHEDULE_H */

