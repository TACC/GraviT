/* 
 * File:   GreedyScheduler.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:12 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_GREEDY_SCHEDULE_H
#define	GVT_RENDER_SCHEDULE_HYBRID_GREEDY_SCHEDULE_H

#include <gvt/core/Debug.h>
#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

 namespace gvt {
    namespace render {
        namespace schedule {
            namespace hybrid { 
                /// hybrid schedule that distributes currently requested data across all processes with no sorting
                /** This schedule simply allocates requested domains to processes, regardless of number of pending rays
                or previously loaded data.

                This schedule has the following issues:
                    - domains with large ids tend to get starved, since the greedy loop iterates from small id to large id
                    - a new allocation may move already-loaded data to another process, incurring excess data loads
                    - processes may remain idle if there are fewer requested domains than processes
                    - a domain can be assigned to a process that does not have rays queued for it, incurring excess ray sends

                    \sa LoadAnyOnceSchedule, SpreadSchedule
                    */
                struct GreedySchedule : public HybridScheduleBase 
                {

                    GreedySchedule(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf) 
                    : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) 
                    {}

                    virtual ~GreedySchedule() {}

                    virtual void operator()() 
                    {
                        for (int i = 0; i < size; ++i)
                            newMap[i] = -1;

                        std::map< int, int > data2proc;
                        for (int s = 0; s < size; ++s) 
                        {
                            if (map_recv_bufs[s]) 
                            {
                                // greedily grab next unclaimed domain
                                for (int d = 1; d < map_size_buf[s]; d += 2) 
                                {
                                    if (data2proc.find(map_recv_bufs[s][d]) == data2proc.end()) 
                                    {
                                        newMap[s] = map_recv_bufs[s][d];
                                        data2proc[map_recv_bufs[s][d]] = s;
                                    }
                                }
                            }
                        }

                        GVT_DEBUG_CODE(DBG_LOW,
                            std::cerr << "new map size is " << size << std::endl;
                            for (int i = 0; i < size; ++i)
                                std::cerr << "    " << i << " -> " << newMap[i] << std::endl;
                            );
                    }
                };

            }
        }
    }
}

#endif	/* GVT_RENDER_SCHEDULE_HYBRID_GREEDY_SCHEDULE_H */

