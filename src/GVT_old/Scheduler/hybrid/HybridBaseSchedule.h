/* 
 * File:   HybridBaseSchedule.h
 * Author: jbarbosa
 *
 * Created on January 24, 2014, 2:03 PM
 */

#ifndef HYBRIDBASESCHEDULE_H
#define	HYBRIDBASESCHEDULE_H

struct HybridBaseSchedule {

    int * newMap; int &size; int *map_size_buf; int **map_recv_bufs;
    int *data_send_buf;
    
    HybridBaseSchedule(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf) : newMap(newMap), size(size), map_size_buf(map_size_buf), map_recv_bufs(map_recv_bufs), data_send_buf(data_send_buf) {
        
    }

    virtual ~HybridBaseSchedule() {
        
    }
private:

};

#endif	/* HYBRIDBASESCHEDULE_H */

