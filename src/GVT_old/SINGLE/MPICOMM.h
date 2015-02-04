/*
 * MPICOMM.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_MPICOMM_H
#define GVT_MPICOMM_H

#include <GVT/common/debug.h>
#include <mpi.h>
#include <iostream>
#include <map>

#include "SINGLENODE.h"

enum PARALLEL_ENUM {
    NOTPARALLEL,
    PARALLEL_MPI
};

enum COMM_SIDEDNESS_ENUM {
    COMM_SIDEDNESS_1,
    COMM_SIDEDNESS_2,
};


#define RAY_BUF_SIZE 10485760  // 10 MB per neighbor
#define SEND_THRESHOLD 1
#define DOM_MAP_BUF_SIZE 100


class MPICOMM : public SINGLE_NODE {
public:

//    std::vector<int> *to_send;
//    std::map<int, int> *data_size;
//    std::map<int, int> *local_sorted_doms;
//    std::map<int, vector<int> > *other_dom;
//
//    MPI_Comm comm;
//
//    unsigned char **in_ray_buf, **local_ray_buf;
//    int *in_ray_buf_ptr, *local_offset, *done_flags, *dom_map_buf;
//
//    MPI_Win *win_in_ray_buf, win_in_ray_buf_ptr, win_done_flags, win_dom_map_buf;

    MPICOMM(int size) : SINGLE_NODE(size) {
        

        
        this->world_size = MPI::COMM_WORLD.Get_size();
        this->rank = MPI::COMM_WORLD.Get_rank();
//        this->rays_start = 0;
//        int ray_portion = size / this->rank_size;
//        this->rays_start = this->rank * ray_portion;
//        this->rays_end = (this->rank + 1) == this->rank_size ? size : (this->rank + 1) * ray_portion; // tack on any odd rays to last proc
//        

        
    };

    virtual ~MPICOMM() {
    };

    void operator()(void) {
    }

};


#endif /* GVT_MPICOMM_H */
