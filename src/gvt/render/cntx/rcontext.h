//
// Created by Joao Barbosa on 9/13/17.
//

#ifndef CONTEXT_RCONTEXT_H
#define CONTEXT_RCONTEXT_H

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
    } else if (type == std::string("Scheduler")) {
      identifier tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("type"), identifier(), n.getid());
      tid = identifier(rank, _identifier_counter++);
      _map[tid] = anode<Variant>(tid, std::string("adapter"), identifier(), n.getid());
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

#endif // CONTEXT_RCONTEXT_H
