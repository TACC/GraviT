#include <gvt/core/comm/message.h>
#include <gvt/core/comm/vote/votebooth.h>

#include <cassert>
namespace gvt {
namespace comm {
std::map<std::size_t, comm::vote::vote> votebooth::_voting;

bool votebooth::aa(std::size_t VOTE_TYPE, std::function<bool(void)> _callback_check,
                   std::function<bool(bool)> _callback_update) {

  assert(_voting.find(VOTE_TYPE) == _voting.end());
  _voting[VOTE_TYPE] = comm::vote::vote(_callback_check, _callback_update);
  return true;
}

bool votebooth::bb(std::size_t VOTE_TYPE) { return true; }
}
}
