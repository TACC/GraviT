#ifndef GVT_CONTEXT_DATABASE_H
#define GVT_CONTEXT_DATABASE_H

#include "identifier.h"
#include "variant.h"

#include "anode.h"
#include "mpi/decoder.h"
#include "mpi/encoder.h"
#include "mpi/mpigroup.h"

#include <atomic>
#include <climits>
#include <functional>
#include <iostream>
#include <map>
#include <memory>

#include <stdio.h>
#include <tuple>
#include <vector>

namespace cntx {

template <typename Variant, typename Derived> struct context {
  typedef std::vector<std::reference_wrapper<anode<Variant>>> children_vector;

  static std::shared_ptr<context> _singleton;

  struct idCompare {
    bool operator()(const identifier &lhs, const identifier &rhs) const {
      return lhs.getid() < rhs.getid();
    }
  };

  std::map<identifier, anode<Variant>, idCompare> _map;
  std::map<std::string, identifier> _unique;
  std::atomic<unsigned> _identifier_counter;

  anode<Variant> _root;

  cntx::mpi::MPIGroup cntx_comm;

  context() {

    cntx_comm = mpi::MPIGroup(MPI_COMM_WORLD).duplicate();

    _root = createnode_allranks("Root", "Root", true);

    createnode_allranks("Data", "Data", true, _root.getid());
    createnode_allranks("Instances", "Instances", true, _root.getid());
    createnode_allranks("Lights", "Lights", true, _root.getid());
    createnode_allranks("Cameras", "Cameras", true, _root.getid());
    createnode_allranks("Camera", "Camera", true, getUnique("Cameras").getid());
    createnode_allranks("Films", "Films", true, _root.getid());
    createnode_allranks("Film", "Film", true, getUnique("Films").getid());
    createnode_allranks("Schedulers", "Schedulers", true, _root.getid());
    createnode_allranks("Scheduler", "Scheduler", true,  getUnique("Schedulers").getid());
  }

  static std::shared_ptr<Derived> singleton() {
    if (_singleton == nullptr) {
      _singleton = std::make_shared<Derived>();
    }
    return std::static_pointer_cast<Derived>(_singleton);
  }

  static Derived &instance() {
    if (_singleton == nullptr) {
      _singleton = std::make_shared<Derived>();
    }
    return *std::static_pointer_cast<Derived>(_singleton).get();
  }

  static identifier rootid() { return instance()._root.getid(); }

  static anode<Variant> &root() { return instance()._root; }

  anode<Variant> &createnode(const std::string type = "node",
                             const std::string name = "",
                             bool const &unique = false,
                             identifier const &parent = identifier()) {

    if (unique) {
      if (_unique.find(name) != _unique.end())
        throw std::runtime_error("Requested unique name [" + name +
                                 "] already exists");
    }

    cntx::anode<Variant> n;
    if (!unique)
      n = anode<Variant>(identifier(cntx_comm.rank, _identifier_counter++),
                         type, name, parent);
    else
      n = anode<Variant>(identifier(cntx_comm.rank, _identifier_counter++),
                         name, name, parent);

    n.unique = unique;
    _map[n.getid()] = n;

    if (unique)
      _unique[name] = n.getid();

    create_children(n, type);

    return _map[n.getid()];
  }

  anode<Variant> &createnode_allranks(const std::string type,
                                      const std::string name = "",
                                      bool const &unique = false,
                                      identifier const &parent = identifier()) {

    if (unique) {
      if (_unique.find(name) != _unique.end())
        throw std::runtime_error("Requested unique name [" + name +
                                 "] already exists");
    }

    cntx::anode<Variant> n;
    if (!unique)
      n = anode<Variant>(
          identifier(identifier::all_ranks, _identifier_counter++), type, name,
          parent);
    else
      n = anode<Variant>(
          identifier(identifier::all_ranks, _identifier_counter++), name, name,
          parent);

    n.unique = unique;
    _map[n.getid()] = n;

    if (unique)
      _unique[name] = n.getid();

    create_children(n, type);

    return _map[n.getid()];
  }

  friend std::ostream &operator<<(std::ostream &os, const context &other) {

    os << "Database print =======================" << std::endl;
    for (auto &kv : other._map) {
      os << kv.second << std::endl;
    }
    return os;
  }

  void printtree(std::ostream &os, children_vector const &v = children_vector(),
                 const unsigned depth = 0) {

    int rank = context::instance().cntx_comm.rank;

    if (depth == 0) {
      os << "DB Unique names ======================" << std::endl;

      for (auto &n : _unique) {
        os << n.first << ",";
      }

      os << std::endl;
      os << "Database print =======================" << std::endl;
      printtree(os, getChildren(_root), depth + 1);
      return;
    }

    if (v.empty())
      return;

    for (auto &c : v) {
      os << "[" << rank << "] ";
      for (int i = 0; i < depth - 1; i++)
        os << ".";
      os << c << std::endl;
      printtree(os, getChildren(c), depth + 1);
    }
  }

  inline std::vector<std::reference_wrapper<anode<Variant>>>
  getChildren(const anode<Variant> &n) {
    std::vector<std::reference_wrapper<anode<Variant>>> c;
    for (auto &kv : _map) {
      if (kv.second.getparent() == n.getid()) {
        c.push_back(std::ref(kv.second));
      }
    }
    return c;
  }

  inline anode<Variant> &getUnique(const std::string name) {

    if (_unique.find(name) != _unique.end()) {
      return _map[_unique[name]];
    }

    return anode<Variant>::error_node;
  }

  static inline anode<Variant> &getChild(const anode<Variant> &n,
                                         const std::string name) {

    context::children_vector children = context::instance().getChildren(n);
    for (auto &c : children) {
      if (c.get().name == name)
        return c.get();
    }
    throw std::runtime_error("Child " + name + " does not exist for " + n.name);
  }

  static inline std::vector<std::reference_wrapper<anode<Variant>>>
  getDirty(int all = -1) {
    std::vector<std::reference_wrapper<anode<Variant>>> _d;
    for (auto &kv : context::instance()._map) {
      if (kv.second.id.isDirty() &&
          (all == -1 || kv.second.id.getrank() == all) &&
          !(kv.first == context::instance().rootid())) {
        _d.push_back(kv.second);
      }
    }
    return _d;
  }

  static inline void sync() {

    context &db = context::instance();

    if (db.cntx_comm.size <= 1)
      return;

    context::children_vector d =
        getDirty((db.cntx_comm.rank == 0) ? -1 : db.cntx_comm.rank);
    unsigned long *counter = new unsigned long[db.cntx_comm.size];
    counter[db.cntx_comm.rank] = d.size();

    unsigned long s = d.size();

    MPI_Allgather(&s, 1, MPI_UNSIGNED_LONG, counter, 1, MPI_UNSIGNED_LONG,
                  db.cntx_comm.comm);

    for (int i = 0; i < db.cntx_comm.size; ++i) {

      unsigned char *buf;

      if (db.cntx_comm.rank == i && d.size() > 0) {

        for (auto &n : d) {

          n.get().id.resetDirty();

          cntx::mpi::encode enc;
          n.get().pack(enc);
          size_t s = enc.size();
          MPI_Bcast(&s, 1, MPI_UNSIGNED_LONG, i, db.cntx_comm.comm);
          MPI_Bcast(enc.getBuffer(), enc.size(), MPI_BYTE, i,
                    db.cntx_comm.comm);

          MPI_Barrier(db.cntx_comm.comm);
        }

      } else {
        for (int nn = 0; nn < counter[i]; ++nn) {
          size_t s;
          MPI_Bcast(&s, 1, MPI_UNSIGNED_LONG, i, db.cntx_comm.comm);
          std::shared_ptr<cntx::mpi::decode::BYTE> buff(
              (cntx::mpi::decode::BYTE *)malloc(s), free);

          cntx::mpi::decode dec(buff, s);
          MPI_Bcast(dec.buffer.get(), s, MPI_BYTE, i, db.cntx_comm.comm);

          anode<Variant> n;
          n.unpack(dec);
          db._map[n.getid()] = n;
          db._map[n.getid()].id.resetDirty();
          if (n.unique)
            db._unique[n.name] = n.getid();

          MPI_Barrier(db.cntx_comm.comm);
        }
      }
    }
  }

  anode<Variant> &create_children(anode<Variant> const &n,
                                  const std::string &type) {
    return (*reinterpret_cast<Derived*>(this)).create_children(n,type);
  }
};

} // namespace cntx

#endif
