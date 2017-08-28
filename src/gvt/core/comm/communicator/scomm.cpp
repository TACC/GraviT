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
#include "scomm.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <mpi.h>

#include <gvt/core/context/CoreContext.h>
#include <gvt/core/tracer/tracer.h>

namespace gvt {
namespace comm {
scomm::scomm() {}
void scomm::init(int argc, char *argv[], bool start_thread) {
  assert(!communicator::_instance);

  std::cout << "Instancing sync communicator..." << std::endl << std::flush;

  // To run MPI_THREAD_MULTIPLE in MVAPICH this is required but it causes severe overhead with MPI calls
  // MVAPICH2 http://mailman.cse.ohio-state.edu/pipermail/mvapich-discuss/2012-December/004151.html
  // MV2_ENABLE_AFFINITY=0
  int req = MPI_THREAD_MULTIPLE;
  int prov= MPI_THREAD_MULTIPLE;
  MPI_Init_thread(&argc, &argv, req, &prov);

  communicator::_instance = std::make_shared<scomm>();

  switch (prov) {
  case MPI_THREAD_SINGLE:
    std::cout << "MPI_THREAD_SINGLE" << std::endl;
    assert(communicator::_instance->lastid() == 1);
    break;
  case MPI_THREAD_FUNNELED:
    std::cout << "MPI_THREAD_FUNNELED" << std::endl;
    assert(communicator::_instance->lastid() == 1);
    break;
  case MPI_THREAD_SERIALIZED:
    std::cout << "MPI_THREAD_SERIALIZED" << std::endl;
    communicator::_MPI_THREAD_SERIALIZED = true; // Potential deadlock
    break;
  case MPI_THREAD_MULTIPLE:
    std::cout << "MPI_THREAD_MULTIPLE" << std::endl;
    communicator::_MPI_THREAD_SERIALIZED = true;
    break;
  default:
    std::cout << "Upppsssss" << std::endl;
  }

  communicator::init(argc, argv, start_thread);
}

void scomm::run() {
  std::cout << id() << " Communicator thread started" << std::endl;
  while (!_terminate) {
    {
      MPI_Status status;
      memset(&status, 0, sizeof(MPI_Status));
      int flag;
      aquireComm();
      MPI_Iprobe(MPI_ANY_SOURCE, CONTROL_SYSTEM_TAG, MPI_COMM_WORLD, &flag, &status);
      releaseComm();
      int n_bytes = 0;
      MPI_Get_count(&status, MPI_BYTE, &n_bytes);

      if (n_bytes > 0) {
        int sender = status.MPI_SOURCE;

        const int data_size = n_bytes - sizeof(Message::header);

        std::shared_ptr<Message> msg = std::make_shared<Message>(data_size);

        // std::cout << "Recv : " << n_bytes << " on " << id() << " from " << sender
        //         << std::flush << std::endl;

        aquireComm();
        MPI_Recv(msg->getMessage<void>(), n_bytes, MPI_BYTE, sender, CONTROL_SYSTEM_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        releaseComm();

        msg->size(data_size);
        std::lock_guard<std::mutex> l(minbox);
        if (msg->system_tag() == CONTROL_USER_TAG) {
          gvt::core::CoreContext &cntxt = *gvt::core::CoreContext::instance();
          cntxt.tracer()->MessageManager(msg);

          //        	_inbox.push_back(msg);
        }
        if (msg->system_tag() == CONTROL_VOTE_TAG) voting->processMessage(msg);
      }
    }
  }
}
}
}
