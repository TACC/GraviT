/*
 * SingleNode.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_CORE_MPI_SINGLE_NODE_H
#define GVT_CORE_MPI_SINGLE_NODE_H

namespace gvt {
    namespace core {
        namespace mpi {
            class SingleNode  
            {    
            public:

                int rank, world_size;
                int rays_start, rays_end;

                SingleNode(int size) 
                : rank(0),world_size(1),rays_start(0),rays_end(size)
                {
                }

                virtual ~SingleNode() 
                {
                }

                //    void getMyPlaceInTheworld(int &rank, int& rank_size, int& rays_start, int& rays_end) {
                //        rank = this->rank;
                //        rank_size = this->rank_size;
                //        rays_start = this->rays_start;
                //        rays_end = this->rays_end;
                //    }

            };
        }
    }
}
#endif /* GVT_CORE_MPI_SINGLE_NODE_H */
