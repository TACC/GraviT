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
#ifndef VOTE_H
#define VOTE_H

#include <gvt/core/comm/message.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace gvt {
namespace comm {
namespace vote {


/**
 * @brief Voting procedure
 *
 * Voting procedure based of two-pahse commit with 0-failure taulerance
 * since we are in a synchrounous world under mpi (a failure will terminate all nodes immediatly)
 * the nodes only need to agree if there is still work somewhere.
 *
 */
struct vote {


  std::function<bool(void)> _callback_check = nullptr; /**< Check is there is still work in node callback (implemented in the scheduler)*/
  std::function<void(bool)> _callback_update = nullptr; /**< Update all nodes done callback (implemented in the scheduler)*/
  std::mutex _only_one;

  enum ClientType { COORDINATOR, COHORT }; /**< Node type coordinator (always node 0 since 0=failure) or cohort */

 /**
  * @brief Vote state
  */
  enum VoteState {
    VOTE_UNKNOWN = 0, /**< Not in a voting cycle */
    PROPOSE,
    VOTE_ABORT,
    VOTE_COMMIT,
    VOTER_FINISED,
    DO_ABORT,
    DO_COMMIT,
    COORD_FINISHED,
    NUM_VOTE_TYPES
  };

  const static char *state_names[];

  /**
   * @brief Current ballot
   */
  struct Ballot {
    long ballotnumber = 0;
    VoteState vote = VOTE_UNKNOWN;
  } _ballot;

  std::atomic<std::size_t> _count;

  /**
   * @brief Vote construct
   * @param CallBackCheck   Check if there is no more work in node
   * @param CallBackUpdate   Inform scheduler that all nodes agree that there is no more work antwhere
   */
  vote(std::function<bool(void)> CallBackCheck = nullptr, std::function<void(bool)> CallBackUpdate = nullptr);
  /**
   * @brief Copy constructor
   */
  vote(const vote &other);
  /**
   * Initiale a voting round
   * @return if the init was sucessful or not
   */
  bool PorposeVoting();
  /**
   * Process received message invoked by the communicator thread when it receives a mesage tagged CONTROL_VOTE_TAG
   * @see Message
   * @param msg Raw message received by the communicator
   */
  void processMessage(std::shared_ptr<comm::Message> msg);

  /**
   * Assign operator
   */
  vote operator=(const vote &other) { return vote(other); }

private:
  void processCohort(std::shared_ptr<comm::Message> msg); /**< Process vote messages if node is a cohort */
  void processCoordinator(std::shared_ptr<comm::Message> msg); /**< Process vote messages if node is a Coordinator */


};
}
}
}

#endif /* VOTE_H */
