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
#ifndef GVT_CORE_COMMUNICATOR_H
#define GVT_CORE_COMMUNICATOR_H

#include <gvt/core/comm/message.h>
#include <gvt/core/comm/vote/votebooth.h>

#include <memory>
#include <mutex>

#include <map>
#include <tbb/task_group.h>
#include <vector>

#include <string>
#include <type_traits>
#include <typeinfo>

namespace gvt {
namespace comm {

struct communicator {

  static std::shared_ptr<communicator> _instance;
  static tbb::task_group tg;
  volatile bool _terminate = false;

  int _id = 0;
  int _size = -1;

  std::vector<std::shared_ptr<Message> > _inbox;
  std::shared_ptr<comm::vote::vote> voting;
  std::mutex minbox;
  std::mutex mvotebooth;

  static std::vector<std::string> registry_names;
  static std::map<std::string, std::size_t> registry_ids;

  std::mutex _mcomm;
  static bool _MPI_THREAD_SERIALIZED;

  communicator();
  virtual ~communicator();

  static std::shared_ptr<communicator> singleton();
  static communicator &instance();
  static void init(int argc = 0, char *argv[] = nullptr, bool start_thread = true);
  std::size_t id();
  std::size_t lastid();

  virtual void send(std::shared_ptr<comm::Message> msg, std::size_t to);
  virtual void broadcast(std::shared_ptr<comm::Message> msg);
  virtual void run() = 0;
  virtual void terminate();
  virtual void aquireComm();
  virtual void releaseComm();

  virtual void setVote(std::shared_ptr<comm::vote::vote> vote) { voting = vote; }

  template <class M> static int RegisterMessageType() {
    static_assert(std::is_base_of<comm::Message, M>::value, "M must inherit from comm::Message");
    std::string classname = typeid(M).name();
    if (registry_ids.find(classname) != registry_ids.end()) {
      return registry_ids[classname];
    }
    registry_names.push_back(classname);
    std::size_t idx = registry_names.size() - 1;
    registry_ids[classname] = idx;
    M::COMMUNICATOR_MESSAGE_TAG = idx;
    return idx;
  }

  template <class M> static std::shared_ptr<M> SAFE_DOWN_CAST(const std::shared_ptr<comm::Message> &msg) {
    if (msg->tag() >= registry_names.size()) return nullptr;
    std::string classname = registry_names[msg->tag()];
    if (registry_ids.find(classname) == registry_ids.end()) return nullptr;
    if (registry_ids[classname] == msg->tag()) return std::static_pointer_cast<M>(msg);
    return nullptr;
  }
};
}
}

#endif /* GVT_CORE_COMMUNICATOR_H */
