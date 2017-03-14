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
#ifndef MPICOMMLIB_SCOMM
#define MPICOMMLIB_SCOMM

#include <gvt/core/comm/communicator.h>
namespace gvt {
namespace comm {

/**
 * @brief Semi-synchronous communicator
 *
 *  The Semi-synchronous communicator operates under the assumption that
 *  all threads in the node can send messages and block to send it. However the
 *  reception of messages is handle by a resident thread that receives the message
 *  and ivokes a to the scheduler to process it. If the message is a voting request
 *   the thread calls a static method on the currently instanciated scheduler to check
 *   the termination criteria.
 *
 */
struct scomm : communicator {
    /**
     * @brief Constructor
     */
  scomm();
  /**
   * Communicator singleton initialization
   * @param argc        Number of arguments in command line (required by MPI Init)
   * @param argv        Arguments in the application command line
   * @param start_thread Should it start a communication threads or not (If the application only uses on compute node)
   */
  static void init(int argc = 0, char *argv[] = nullptr, bool start_thread = true);
  /**
   * Method execute by the resident communication thread
   */
  virtual void run();
};
}
}

#endif /*MPICOMMLIB*/
