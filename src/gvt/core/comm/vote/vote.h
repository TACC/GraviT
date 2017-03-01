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

struct vote {

  std::function<bool(void)> _callback_check = nullptr;
  std::function<void(bool)> _callback_update = nullptr;
  std::mutex _only_one;

  enum ClientType { COORDINATOR, COHORT };

  enum VoteState {
    VOTE_UNKNOWN = 0,
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

  struct Ballot {
    long ballotnumber = 0;
    VoteState vote = VOTE_UNKNOWN;
  } _ballot;

  std::atomic<std::size_t> _count;

  vote(std::function<bool(void)> CallBackCheck = nullptr, std::function<void(bool)> CallBackUpdate = nullptr);
  vote(const vote &other);
  bool PorposeVoting();
  void processMessage(std::shared_ptr<comm::Message> msg);

  void processCohort(std::shared_ptr<comm::Message> msg);
  void processCoordinator(std::shared_ptr<comm::Message> msg);

  vote operator=(const vote &other) { return vote(other); }
};
}
}
}

#endif /* VOTE_H */
