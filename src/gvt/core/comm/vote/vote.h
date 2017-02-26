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
