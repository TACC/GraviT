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
#ifndef GVT_CORE_MESSAGE_H
#define GVT_CORE_MESSAGE_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace gvt {
namespace comm {

enum SYSTEM_COMM_TAG { CONTROL_SYSTEM_TAG = 0x8, CONTROL_USER_TAG, CONTROL_VOTE_TAG };

#define REGISTERABLE_MESSAGE(ClassName) static int COMMUNICATOR_MESSAGE_TAG;
#define REGISTER_INIT_MESSAGE(ClassName) int ClassName::COMMUNICATOR_MESSAGE_TAG = -1;

struct Message {

  typedef unsigned char Byte;

  REGISTERABLE_MESSAGE(Message);

  struct header {
    std::size_t USER_TAG;
    std::size_t SYSTEM_TAG = CONTROL_USER_TAG;
    std::size_t USER_MSG_SIZE;
    long dst;
    long src;
  };

  Message(const std::size_t &size = 0);
  Message(const Message &msg);
  Message(Message &&msg);

  ~Message();

  header &getHeader();
  std::size_t tag();
  void tag(const std::size_t tag);
  std::size_t size();
  void size(const std::size_t size);
  template <typename T> std::size_t sizehas() { return getHeader().USER_MSG_SIZE / sizeof(T); };

  std::size_t system_tag();
  void system_tag(std::size_t);
  std::size_t buffer_size() const { return _buffer_size; };

  long &dst() { return getHeader().dst; }
  long &src() { return getHeader().src; }

  void dst(long d) { getHeader().dst = d; }
  void src(long s) { getHeader().src = s; }

  template <typename T> T *getMessage() { return reinterpret_cast<T *>(content); }
  template <typename T> void setMessage(T *orig, const std::size_t &os) {
    header mhi = getHeader();
    std::size_t bs = sizeof(T) * os;
    if (_buffer_size > 0) {
      content = static_cast<Byte *>(std::realloc(content, bs + sizeof(header)));
    } else {
      content = static_cast<Byte *>(std::malloc(bs + sizeof(header)));
    }
    _buffer_size = bs + sizeof(header);
    std::memcpy(content, orig, bs);
    std::memcpy(content + bs, &mhi, sizeof(header));
    size(bs);
  }

protected:
  std::size_t _buffer_size = 0;
  Byte *content = nullptr;
};
}
}

#endif
