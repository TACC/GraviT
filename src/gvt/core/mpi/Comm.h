/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
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

#include <iostream>
#include <map>
#include <mpi.h>

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
// clang-format off
enum PARALLEL_ENUM {
  NOTPARALLEL,
  PARALLEL_MPI
};
// clang-format on

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

class MPICOMM : public SingleNode {
public:
  MPICOMM(int size) : SingleNode(size) {
    this->world_size = MPI::COMM_WORLD.Get_size();
    this->rank = MPI::COMM_WORLD.Get_rank();
  }

  virtual ~MPICOMM() {}

  void operator()(void) {}

  template <typename B> void gatherbuffer(B *buf, size_t size) {

    if (MPI::COMM_WORLD.Get_size() < 2) return;

    size_t partition_size = size / MPI::COMM_WORLD.Get_size();
    size_t next_neighbor = MPI::COMM_WORLD.Get_rank();
    size_t prev_neighbor = MPI::COMM_WORLD.Get_rank();

    B *acc = &buf[MPI::COMM_WORLD.Get_rank() * partition_size];
    B *gather = new B[partition_size * MPI::COMM_WORLD.Get_size()];

    std::vector<MPI::Request> Irecv_requests_status;

    for (int round = 0; round < MPI::COMM_WORLD.Get_size(); round++) {
      next_neighbor = (next_neighbor + 1) % MPI::COMM_WORLD.Get_size();
      prev_neighbor = (prev_neighbor != 0 ? prev_neighbor - 1 : MPI::COMM_WORLD.Get_size() - 1);

      if (next_neighbor != MPI::COMM_WORLD.Get_rank()) {
        B *send = &buf[next_neighbor * next_neighbor];
        MPI::COMM_WORLD.Isend(send, sizeof(B) * partition_size, MPI::BYTE, next_neighbor,
                              MPI::COMM_WORLD.Get_rank() | 0xF00000000000000);
      }
      if (prev_neighbor != MPI::COMM_WORLD.Get_rank()) {
        B *recv = &gather[prev_neighbor * partition_size];
        MPI::COMM_WORLD.Irecv(recv, sizeof(B) * partition_size, MPI::BYTE, prev_neighbor,
                              prev_neighbor | 0xF00000000000000);
      }
    }

    for (MPI::Request &request : Irecv_requests_status) {
      MPI::Request::Wait(request);
      MPI::Status status;
      MPI::Request::Get_status(status);
      unsigned int source = status.Get_source();
      for (int i = 0; i < partition_size; ++i) {
        acc[i] = gather[source * partition_size + i];
      }
    }

    MPI::COMM_WORLD.Gather(acc, sizeof(B) * partition_size, MPI::BYTE, sizeof(B) * partition_size, MPI::BYTE, 0);
    return;
  }
};
}
}
}
#endif /* GVT_CORE_MPI_COMM_H */
