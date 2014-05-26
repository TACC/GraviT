/*
 * SINGLENODE.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_SINGLENODE_H
#define GVT_SINGLENODE_H


#include <GVT/Environment/RayTracerAttributes.h>
#include <GVT/Data/scene/Image.h>
#include <GVT/Tracer/Tracer.h>

class SINGLE_NODE  {
    
public:

    int rank, world_size;
    int rays_start, rays_end;


    SINGLE_NODE(int size) : world_size(size)  {
        rank = 0, world_size = 1;
        rays_start = 0;
        rays_end = size;
    };

    virtual ~SINGLE_NODE() {
    };

//    void getMyPlaceInTheworld(int &rank, int& rank_size, int& rays_start, int& rays_end) {
//        rank = this->rank;
//        rank_size = this->rank_size;
//        rays_start = this->rays_start;
//        rays_end = this->rays_end;
//    }

};

#endif /* GVT_SINGLENODE_H */
