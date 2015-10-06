/*
 * Comm.h
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#ifndef GVT_CORE_MPI_COMM_H
#define GVT_CORE_MPI_COMM_H

#include <gvt/core/Debug.h>
#include <gvt/core/mpi/SingleNode.h>

#include <mpi.h>
#include <iostream>
#include <map>

 namespace gvt {
    namespace core {
        namespace mpi {
            /// enumeration for process parallel mode
            /** enumeration for process parallel mode. 
            note that aspects of GraviT will use thread-based parallelism independent of this setting.

            - NOTPARALLEL - not using process parallelism
            - PARALLEL_MPI - using MPI-based process parallelism

            \sa COMM_SIDEDNESS_ENUM
            */
            enum PARALLEL_ENUM {
                NOTPARALLEL,
                PARALLEL_MPI
            };

            /// MPI communication type used in process parallel mode
            /** communication type used during process parallel mode 
            - COMM_SIDEDNESS_1 - one-sided MPI communication
            - COMM_SIDEDNESS_2 - two-sided MPI communication

            \sa PARALLEL_ENUM
            */
            enum COMM_SIDEDNESS_ENUM {
                COMM_SIDEDNESS_1,
                COMM_SIDEDNESS_2,
            };

            class MPICOMM : public SingleNode 
            {
            public:
                MPICOMM(int size) : SingleNode(size) 
                {     
                    this->world_size = MPI::COMM_WORLD.Get_size();
                    this->rank = MPI::COMM_WORLD.Get_rank();
                }

                virtual ~MPICOMM() 
                {
                }

                void operator()(void) 
                {
                }
            };
        }
    }
}
#endif /* GVT_CORE_MPI_COMM_H */
