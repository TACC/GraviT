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
#ifndef MPICOMMLIB_ACOMM
#define MPICOMMLIB_ACOMM

#include <gvt/core/comm/communicator.h>

namespace gvt {
namespace comm {
    /**
     * @brief Pure assynchronous communicator
     *
     * This communicator assumes that all message passing is done asynchronously. If a compute threads sends a message
     * the message is placed in a queue to be sent later by the resident communication threads and a handler is return to
     * the calling control flow so that we can determine is the message was already sent or not.
     *
     */
struct acomm : public communicator {
  acomm();
  /**
   * Communicator singleton initialization
   * @param argc        Number of arguments in command line (required by MPI Init)
   * @param argv        Arguments in the application command line
   * @param start_thread Should it start a communication threads or not (If the application only uses on compute node)
   */
  static void init(int argc = 0, char *argv[] = nullptr, bool start_thread = true);
  /**
   * Send msg to compute node id
   * @param msg Message to be sent
   * @param id  Destination compute node id
   */
  virtual void send(std::shared_ptr<comm::Message> msg, std::size_t id);
  /**
   * Send msg to all compute nodes
   * @param msg Message to be sent
   */
  virtual void broadcast(std::shared_ptr<comm::Message> msg);
  /**
   * Method execute by the resident communication thread
   */
  virtual void run();

  std::vector<std::shared_ptr<Message> > _outbox; /**< Outbox message queue */
  std::mutex moutbox; /** Outbox protection mutex */
};
}
}

#endif /*MPICOMMLIB*/
