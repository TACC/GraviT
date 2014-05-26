/* 
 * File:   comm_d.h
 * Author: jbarbosa
 *
 * Created on January 20, 2014, 5:22 PM
 */

#ifndef GVT_COMM_D_H
#define	GVT_COMM_D_H

struct comm_data {
    std::vector<int> *to_send;
    std::map<int, int> *data_size;
    std::map<int, int> *local_sorted_doms;
    std::map<int, vector<int> > *other_dom;

    MPI_Comm comm;

    unsigned char **in_ray_buf, **local_ray_buf;
    int *in_ray_buf_ptr, *local_offset, *done_flags, *dom_map_buf;

    MPI_Win *win_in_ray_buf, *win_in_ray_buf_ptr, win_done_flags,win_dom_map_buf;


    vector<int> to_del;
    int rank, size;
    map<int, int> n_index_map;
    int n_size;
    int *n_lut;
    int n_read_rank;
    int *n_write_rank;

    MPI_Comm *comms;


};

#endif	/* GVT_COMM_D_H */

