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

/*!
   \file communicator.h
   \brief Communication handling thread
*/

namespace gvt {
namespace comm {

struct communicator {

  static std::shared_ptr<communicator> _instance; /**< Stores the communicator singleton pointer*/
  static tbb::task_group tg;                      /**< Stores the current TBB thread arena */
  volatile bool _terminate = false;               /**< True if the terminate procedure as been invoked */

  int _id = 0;    /**< MPI rank */
  int _size = -1; /**< MPI world size */

  std::vector<std::shared_ptr<Message> > _inbox; /**< Queue of messages received from other nodes waiting to be
                                                    processed by the scheduler*/
  std::shared_ptr<comm::vote::vote> voting;      /**< Voting state pointer for process agreement */
  std::mutex minbox;                             /**< Inbox mutex */
  std::mutex mvotebooth;

  static std::vector<std::string> registry_names;         /**< Global message name registry */
  static std::map<std::string, std::size_t> registry_ids; /**< Global message type registry */

  std::mutex _mcomm;
  static bool _MPI_THREAD_SERIALIZED;

  /*!
     \brief Get current communicator instance
     \return returns a shared pointer to the instance
  */
  static std::shared_ptr<communicator> singleton();
  /*!
     \brief Get current communicator instance
     \return returns a reference to the instance
  */
  static communicator &instance();

  /*!
     \brief Initialize communicator threads
     \param argc Command line argument counter
     \param argv Command parameters as strings
     \param start_thread If true (or ignored) it will launch a communication thread, set to false if only a single
     compute node is used
  */
  static void init(int argc = 0, char *argv[] = nullptr, bool start_thread = true);
  /*!
     \brief Get current node id (MPI Rank)
     \return Current compute node id
  */
  std::size_t id();
  /*!
     \brief Get current node count (MPI world size)
     \return Total number of compute nodes
  */
  std::size_t lastid();

  /*!
     \brief Send a message buffer to dst compute node
     \param msg Shared pointer to msg description
     \param dst Compute node targe
  */
  virtual void send(std::shared_ptr<comm::Message> msg, std::size_t dst);
  /*!
     \brief Send message(msg) to all compute nodes
  */
  virtual void broadcast(std::shared_ptr<comm::Message> msg);

  /*!
     \brief Terminate communicator
  */
  virtual void terminate();

  /*!
     \brief Terminate communicator
  */
  virtual void setVote(std::shared_ptr<comm::vote::vote> vote) { voting = vote; }

  /*!
     \brief Register a message type with the communicator
     The communicator needs a message type identifier to deterime what to do with a message when it is received.
     \return Message type identifer in the communicator
  */

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

  /*!
     \brief Convert a generic message received by the buffer into the proper message type
     \return Return the generic msg to converted to the proper class or nullptr if the conversion is incorrect.
  */

  template <class M> static std::shared_ptr<M> SAFE_DOWN_CAST(const std::shared_ptr<comm::Message> &msg) {
    if (msg->tag() >= registry_names.size()) return nullptr;
    std::string classname = registry_names[msg->tag()];
    if (registry_ids.find(classname) == registry_ids.end()) return nullptr;
    if (registry_ids[classname] == msg->tag()) return std::static_pointer_cast<M>(msg);
    return nullptr;
  }

protected:
  /*!
     \brief Constructor
  */
  communicator();
  /*!
     \brief Destructor
  */
  virtual ~communicator();
  virtual void aquireComm();
  virtual void releaseComm();
  virtual void run() = 0;
};
}
}

#endif /* GVT_CORE_COMMUNICATOR_H */
