#include <gvt/core/comm/communicator.h>
#include <gvt/core/comm/vote/vote.h>

#include <cassert>
#include <iostream>

namespace gvt {
namespace comm {
namespace vote {

const char *vote::state_names[] = { "VOTE_UNKNOWN", "PROPOSE",   "VOTE_ABORT",     "VOTE_COMMIT",   "VOTER_FINISED",
                                    "DO_ABORT",     "DO_COMMIT", "COORD_FINISHED", "NUM_VOTE_TYPES" };

vote::vote(std::function<bool(void)> CallBackCheck, std::function<void(bool)> CallBackUpdate)
    : _callback_check(CallBackCheck), _callback_update(CallBackUpdate), _count(0) {
  std::shared_ptr<comm::communicator> comm = comm::communicator::singleton();
  _ballot.vote = VOTE_UNKNOWN;
}

vote::vote(const vote &other) : vote(other._callback_check, other._callback_update) {
  _ballot.vote = other._ballot.vote;
}

bool vote::PorposeVoting() {
  std::shared_ptr<comm::communicator> comm = comm::communicator::singleton();
  if (comm->lastid() == 1) {
    _callback_update(true);
    return true;
  }

  if (comm->id() != 0) return false;
  // std::cout << "Coordinator propose " << state_names[_ballot.vote] << std::flush
  //           << std::endl;

  if (_ballot.vote != VOTE_UNKNOWN || !_callback_check()) return false;

  _count = 0;
  std::shared_ptr<gvt::comm::Message> msg = std::make_shared<gvt::comm::Message>(sizeof(Ballot));
  msg->system_tag(CONTROL_VOTE_TAG);
  Ballot &b = *msg->getMessage<Ballot>();
  b.ballotnumber = ++_ballot.ballotnumber;
  // std::cout << "Coordinator new ballot " << b.ballotnumber << " count " << _count
  //           << std::endl
  //           << std::flush;
  b.vote = _ballot.vote = PROPOSE;
  comm->broadcast(msg);

  return true;
}

void vote::processMessage(std::shared_ptr<comm::Message> msg) {
  assert(_callback_check && _callback_update);
  std::shared_ptr<gvt::comm::communicator> comm = comm::communicator::singleton();
  Ballot &b = *msg->getMessage<Ballot>();
  if (b.ballotnumber < _ballot.ballotnumber) {
    // std::cout << comm->id() << " : Ballot ignored "
    //           << " from " << msg->src() << " ballot " << b.ballotnumber << " current "
    //           << _ballot.ballotnumber << std::endl
    //           << std::flush;
    return;
  }

  _ballot.ballotnumber = b.ballotnumber;

  if (comm->id() == 0) {
    processCoordinator(msg);
    return;
  } else {
    processCohort(msg);
    return;
  }
}

void vote::processCoordinator(std::shared_ptr<comm::Message> msg) {
  assert(_callback_check && _callback_update);
  std::shared_ptr<gvt::comm::communicator> comm = comm::communicator::singleton();
  Ballot &b = *msg->getMessage<Ballot>();

  if (b.vote == VOTE_COMMIT || b.vote == VOTE_ABORT) {
    _count++;
  }

  if (b.vote == VOTE_ABORT && _ballot.vote == PROPOSE) {
    _ballot.vote = DO_ABORT;
  }

  // std::cout << comm->id() << " : Received : " << state_names[b.vote] << " in "
  //           << state_names[_ballot.vote] << " from " << msg->src() << " ballot "
  //           << b.ballotnumber << " current " << _ballot.ballotnumber << " count "
  //           << _count << "/" << (comm->lastid() - 1) << std::endl
  //           << std::flush;

  if (_count == (comm->lastid() - 1)) {

    if (_ballot.vote == DO_ABORT)
      b.vote = DO_ABORT;
    else
      b.vote = (_callback_check()) ? DO_COMMIT : DO_ABORT;
    // std::cout << comm->id() << " : Send : " << state_names[b.vote] << std::endl
    //           << std::flush;
    comm->broadcast(msg);
    _callback_update(b.vote == DO_COMMIT);
    _ballot.vote = VOTE_UNKNOWN;
  }
}

void vote::processCohort(std::shared_ptr<comm::Message> msg) {
  std::shared_ptr<comm::communicator> comm = comm::communicator::singleton();
  Ballot &b = *msg->getMessage<Ballot>();

  _ballot.ballotnumber = b.ballotnumber;

  // std::cout << comm->id() << " : Received : " << state_names[b.vote] << " in "
  //           << state_names[_ballot.vote] << " from " << msg->src() << " ballot "
  //           << b.ballotnumber << " current " << _ballot.ballotnumber << std::endl
  //           << std::flush;

  if (b.vote == PROPOSE && _ballot.vote == VOTE_UNKNOWN) {
    _ballot.vote = b.vote = _callback_check() ? VOTE_COMMIT : VOTE_ABORT;
    comm->send(msg, 0);
    return;
  }

  if (b.vote == DO_COMMIT || b.vote == DO_ABORT) {
    _callback_update(b.vote == DO_COMMIT);
    _ballot.vote = b.vote = VOTE_UNKNOWN;
    return;
  }
}
}
}
}
