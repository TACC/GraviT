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
