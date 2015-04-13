/* 
 * File:   GreedyScheduler.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:12 PM
 */

#ifndef GREEDYSCHEDULER_H
#define	GREEDYSCHEDULER_H


#include "HybridBaseSchedule.h"

struct GreedySchedule : public HybridBaseSchedule {

    GreedySchedule(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf) : HybridBaseSchedule(newMap, size, map_size_buf, map_recv_bufs, data_send_buf) {

    }

    virtual ~GreedySchedule() {

    }

    virtual void operator()() {
        for (int i = 0; i < size; ++i)
            newMap[i] = -1;

        std::map< int, int > data2proc;
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

        DEBUG(cerr << "new map size is " << size << endl;
        for (int i = 0; i < size; ++i)
                cerr << "    " << i << " -> " << newMap[i] << endl;
                );
    }

};


#endif	/* GREEDYSCHEDULER_H */

