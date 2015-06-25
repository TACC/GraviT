/*
 * File:   Spread.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:14 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_SPREAD_SCHEDULE_H
#define	GVT_RENDER_SCHEDULE_HYBRID_SPREAD_SCHEDULE_H

#include <gvt/render/schedule/hybrid/HybridScheduleBase.h>

#ifdef __USE_TAU
#include <TAU.h>
#endif

namespace gvt {
    namespace render {
        namespace schedule {
            namespace hybrid {
                 struct SpreadSchedule : public HybridScheduleBase
                 {

                    SpreadSchedule(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
                    : HybridScheduleBase(newMap, size, map_size_buf, map_recv_bufs, data_send_buf)
                    {}

                    virtual ~SpreadSchedule() {}

                    virtual void operator()()
                    {
#ifdef __USE_TAU
 TAU_START("SpreadSchedule.h.operator");
#endif

                        GVT_DEBUG(DBG_LOW,"in spread schedule");

                        for (int i = 0; i < size; ++i)
                            newMap[i] = -1;

                        std::map<int, int> data2proc;
                        std::vector< int > queued;
                        for (int s = 0; s < size; ++s)
                        {
                            if (map_recv_bufs[s])
                            {
                                // add currently loaded data
                                data2proc[map_recv_bufs[s][0]] = s; // this will evict previous entries, that's okay (I think)
                                GVT_DEBUG(DBG_LOW,"    noting " << map_recv_bufs[s][0] << " -> " << s);

                                // add queued data
                                for (int d = 1; d < map_size_buf[s]; d += 2)
                                    queued.push_back(map_recv_bufs[s][d]);
                            }
                        }

                        // iterate over queued data, find which are already loaded somewhere
                        std::vector<int> homeless;
                        for (int i = 0; i < queued.size(); ++i)
                        {
                            std::map<int, int>::iterator it = data2proc.find(queued[i]);
                            if (it != data2proc.end())
                            {
                                newMap[it->second] = it->first;
                                GVT_DEBUG(DBG_LOW,"    adding " << it->second << " -> " << it->first << " to map");
                            }
                            else
                            {
                                homeless.push_back(queued[i]);
                                GVT_DEBUG(DBG_LOW,"    noting " << queued[i] << " is homeless");
                            }
                        }

                        // iterate over newMap, fill as many procs as possible with homeless data
                        // could be dupes in the homeless list, so keep track of what's added
                        for (int i = 0; (i < size) & (!homeless.empty()); ++i)
                        {
                            if (newMap[i] < 0)
                            {
                                while (!homeless.empty()
                                    && data2proc.find(homeless.back()) != data2proc.end())
                                    homeless.pop_back();
                                if (!homeless.empty())
                                {
                                    newMap[i] = homeless.back();
                                    data2proc[newMap[i]] = i;
                                    homeless.pop_back();
                                }
                            }
                        }

                        GVT_DEBUG_CODE(DBG_LOW,
                            std::cerr << "new map size is " << size << std::endl;
                            for (int i = 0; i < size; ++i)
                                std::cerr << "    " << i << " -> " << newMap[i] << std::endl;
                            );
#ifdef __USE_TAU
 TAU_STOP("SpreadSchedule.h.operator");
#endif

                    }
                };
            }
        }
    }
}

#endif	/* GVT_RENDER_SCHEDULE_HYBRID_SPREAD_SCHEDULE_H */

