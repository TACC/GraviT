#ifndef VOTE_BOOTH_H
#define VOTE_BOOTH_H

#include <gvt/core/comm/vote/vote.h>
#include <map>

namespace gvt {
namespace comm {
class votebooth {
public:
  static std::map<std::size_t, comm::vote::vote> _voting;
  static bool aa(std::size_t VOTE_TYPE, std::function<bool(void)> _callback_check,
                 std::function<bool(bool)> _callback_update);
  static bool bb(std::size_t VOTE_TYPE);
};
}
}
#endif /*VOTE_BOOTH_H*/
