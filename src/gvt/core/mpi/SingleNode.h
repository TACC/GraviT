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
            };
        }
    }
}
#endif /* GVT_CORE_MPI_SINGLE_NODE_H */
