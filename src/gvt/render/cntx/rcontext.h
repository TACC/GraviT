/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2017 Texas Advanced Computing Center, The University of Texas at Austin
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

//
// Created by Joao Barbosa on 9/13/17.
//

#ifndef GVT_CONTEXT_RENDER_CONTEXT_H
#define GVT_CONTEXT_RENDER_CONTEXT_H

#include "variant_def.h"
#include <gvt/core/cntx/context.h>

namespace cntx {

using node = cntx::anode<Variant>;

struct rcontext : public cntx::context<Variant, rcontext> {

public:
  rcontext() : context<Variant, rcontext>() {}

  anode<Variant> &create_children(anode<Variant> const &n, const std::string &type) {

    std::size_t rank = n.getid().getrank();

    if (type == std::string("Camera")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("focus"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("eyePoint"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("upVector"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("fov"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("cam2wrld"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("rayMaxDepth"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("raySamples"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("jitterWindowSize"), identifier(), n.getid());
    } else if (type == std::string("Film")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("width"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("height"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("outputPath"), identifier(), n.getid());
    } else if (type == std::string("Mesh")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("file"), nullptr, n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("type"), std::string("MESH"), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("ptr"), nullptr, n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("bbox"), nullptr, n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("Locations"), nullptr, n.getid());
    }  else if (type == std::string("Volume")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("file"), nullptr, n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("type"), std::string("VOLUME"), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("ptr"), nullptr, n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("bbox"), nullptr, n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("Locations"), nullptr, n.getid());
    } else if (type == std::string("Instance")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("id"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("meshRef"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("bbox"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("centroid"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("mat"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("matinv"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("normi"), identifier(), n.getid());
    } else if (type == std::string("PointLight")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("position"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("color"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("type"), identifier(), n.getid());
      _map[tid] = type;
    } else if (type == std::string("AreaLight")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("position"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("color"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("normal"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("height"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("width"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("type"), identifier(), n.getid());
      _map[tid] = type;
    } else if (type == std::string("Scheduler")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("type"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("volume"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("adapter"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("camera"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("film"), identifier(), n.getid());
    }
    return _map[n.getid()];
  }
#if 0
    void printtree(std::ostream &os, children_vector const &v = children_vector(), const unsigned depth = 0) {

    int rank = context::instance().cntx_comm.rank;

    if (depth == 0) {
      os << "DB Unique names .... ======================" << std::endl;

      for (auto &n : _unique) {
        os << n.first << ",";
      }

      os << std::endl;
      os << "Database print =======================" << std::endl;
      printtree(os, getChildren(_root), depth + 1);
      return;
    }

    if (v.empty()) return;

    for (auto &c : v) {
      os << "[" << rank << "] ";
      for (int i = 0; i < depth - 1; i++) os << ".";

      auto children = getChildren(c);


      if(c.get().name == "bbox") {
      }
      if(c.get().v.is<std::shared_ptr<gvt::render::data::primitives::Box3D>>()) {
        os << "BBOX " << c.get().v.to<std::shared_ptr<gvt::render::data::primitives::Box3D>>() << std::endl;
      }

      os << c << " {{ " << children.size() << " }} " <<std::endl;



      printtree(os, children, depth + 1);
    }
  }
#endif

  void addLocation(const cntx::node &n) {}
};

} // namespace cntx

#endif // GVT_CONTEXT_RENDER_CONTEXT_H
