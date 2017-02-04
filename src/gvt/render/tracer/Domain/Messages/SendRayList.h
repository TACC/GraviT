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

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */

#ifndef GVT_DOMAIN_SEND_RAY_LIST_H
#define GVT_DOMAIN_SEND_RAY_LIST_H

#include <gvt/core/comm/message.h>
#include <gvt/render/actor/Ray.h>

namespace gvt {
namespace comm {

struct EmptyMessage : public gvt::comm::Message {
  REGISTERABLE_MESSAGE(EmptyMessage);

protected:
public:
  EmptyMessage() : gvt::comm::Message() { tag(COMMUNICATOR_MESSAGE_TAG); };
  EmptyMessage(const size_t &n) : gvt::comm::Message(n) {
    tag(COMMUNICATOR_MESSAGE_TAG);
  };
  // EmptyMessage(const long src, const long dst, gvt::render::actor::RayVector &raylist);
};

struct SendRayList : public gvt::comm::Message {
  REGISTERABLE_MESSAGE(SendRayList);

protected:
  // size_t number_rays;
  // long src = -1;
  // long dst = -1;

public:
  SendRayList() : gvt::comm::Message(){};
  SendRayList(const size_t &n) : gvt::comm::Message(n){};
  SendRayList(const long src, const long dst, gvt::render::actor::RayVector &raylist);
};
}
}

#endif /*GVT_DOMAIN_SEND_RAY_LIST_H*/
