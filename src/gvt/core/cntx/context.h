#ifndef GVT_CONTEXT_DATABASE_H
#define GVT_CONTEXT_DATABASE_H

#include "identifier.h"
#include "variant.h"

#include "anode.h"
#include "mpi/decoder.h"
#include "mpi/encoder.h"
#include "mpi/mpigroup.h"

#include <gvt/core/tracer/tracer.h>

#include <atomic>
#include <climits>
#include <functional>
#include <iostream>
#include <map>
#include <memory>

#include <set>
#include <stdio.h>
#include <tuple>
#include <vector>
#include <algorithm>

namespace cntx {

template <typename Variant, typename Derived> struct context {

  typedef anode<Variant> cnode;

  typedef std::vector<std::reference_wrapper<anode<Variant> > > children_vector;

  static context<Variant, Derived> *_singleton;

  struct idCompare {
    bool operator()(const identifier &lhs, const identifier &rhs) const { return lhs.getid() < rhs.getid(); }
  };

  std::shared_ptr<gvt::core::Scheduler> tracer = nullptr;

  std::map<identifier, anode<Variant>, idCompare> _map;
  std::map<std::string, identifier> _unique;

  std::atomic<unsigned> _identifier_counter;

  anode<Variant> _root;

  cntx::mpi::MPIGroup cntx_comm;

  context() : _identifier_counter(0) {

    cntx_comm = mpi::MPIGroup(MPI_COMM_WORLD).duplicate();
    _root = createnode_allranks("Root", "Root", true);
    createnode_allranks("Data", "Data", true, _root);
    createnode_allranks("Instances", "Instances", true, _root);
    createnode_allranks("Lights", "Lights", true, _root);
    createnode_allranks("Cameras", "Cameras", true, _root);
    createnode_allranks("Camera", "Camera", true, getUnique("Cameras"));
    createnode_allranks("Films", "Films", true, _root);
    createnode_allranks("Film", "Film", true, getUnique("Films"));
    createnode_allranks("Schedulers", "Schedulers", true, _root);
    createnode_allranks("Scheduler", "Scheduler", true, getUnique("Schedulers"));
    createnode_allranks("DataLocality", "DataLocality", true, _root);

    //    createnode("DataLoc",std::to_string(cntx_comm.rank),false,getUnique("DataLocality").getid());
  }

  static std::shared_ptr<Derived> singleton() {
    if (_singleton == nullptr) {
      _singleton = new Derived();
    }
    return static_cast<Derived *>(_singleton);
  }

  static Derived &instance() {
    if (_singleton == nullptr) {
      _singleton = new Derived();
    }
    return *static_cast<Derived *>(_singleton);
  }

  static identifier rootid() { return instance()._root.getid(); }

  static anode<Variant> &root() { return instance()._root; }

  static anode<Variant> &deRef(const identifier &id) {
    if(instance()._map.find(id) == instance()._map.end()) throw std::runtime_error("[Error] : identifier  does not exist in database");
    return instance()._map[id];
  }

  anode<Variant> &createnode(const std::string type = "node", const std::string name = "", bool const &unique = false,
                             anode<Variant> const &parent = identifier()) {

    if (unique) {
      if (_unique.find(name) != _unique.end())
        throw std::runtime_error("Requested unique name [" + name + "] already exists");
    }

    cntx::anode<Variant> n;
    if (!unique)
      n = anode<Variant>(identifier(cntx_comm.rank, _identifier_counter++), type, name, parent.getid());
    else {

      n = anode<Variant>(identifier(cntx_comm.rank, _identifier_counter++), name, name, parent.getid());
    }

    n.unique = unique;
    _map[n.getid()] = n;

    if (unique) _unique[name] = n.getid();

    create_children(n, type);

    return _map[n.getid()];
  }

  anode<Variant> &createnode_allranks(const std::string type, const std::string name = "", bool const &unique = false,
                                      anode<Variant> const &parent = identifier()) {

    if (unique) {
      if (_unique.find(name) != _unique.end())
        throw std::runtime_error("Requested unique name [" + name + "] already exists");
    }

    cntx::anode<Variant> n;

    if (!unique)
      n = anode<Variant>(identifier(identifier::all_ranks, _identifier_counter++), type, name, parent.getid());
    else {
      getUniqueIdentifier();
      n = anode<Variant>(identifier(identifier::all_ranks, _identifier_counter++), name, name, parent.getid());
    }

    n.unique = unique;
    _map[n.getid()] = n;

    if (unique) _unique[name] = n.getid();

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

  void printdb() {

    for (const auto &p : _map) {

      std::cout << p.second << std::endl;
    }
  }

  virtual void printtree(std::ostream &os, children_vector const &v = children_vector(), const unsigned depth = 0) {

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

    if (v.empty()) return;

    for (auto &c : v) {
      os << "[" << rank << "] ";
      for (int i = 0; i < depth - 1; i++) os << ".";

      auto children = getChildren(c);

      os << c << std::endl;//<< " {{ " << children.size() << " }} " << std::endl;

      printtree(os, children, depth + 1);
    }
  }

  void printtreebyrank(std::ostream &os, children_vector const &v = children_vector(), const unsigned depth = 0) {

    int rank = context::instance().cntx_comm.rank;

    for (int i = 0; i < context::instance().cntx_comm.size; i++) {
      if (rank == i) printtree(os, v, depth);
      MPI_Barrier(context::instance().cntx_comm.comm);
    }
  }

  inline std::vector<std::reference_wrapper<anode<Variant> > > getChildren(const anode<Variant> &n) {
    std::vector<std::reference_wrapper<anode<Variant> > > c;

    for (auto &kv : _map) {

      if (kv.second.getparent() == n.getid()) {
        c.push_back(std::ref(kv.second));
      }
    }

    return c;
  }

  inline unsigned getUniqueIdentifier() {
    unsigned nid = 0;
    MPI_Allreduce(&_identifier_counter, &nid, 1, MPI_UNSIGNED, MPI_MAX, cntx_comm.comm);
    _identifier_counter = ++nid;
    return _identifier_counter++;
  }

  inline unsigned nextIdentifier() { return _identifier_counter++; }

  inline anode<Variant> &getUnique(const std::string name) {

    if (_unique.find(name) != _unique.end()) {
      return _map[_unique[name]];
    }

    return anode<Variant>::error_node;
  }

  static inline anode<Variant> &getChild(const anode<Variant> &n, const std::string name) {

    context::children_vector children = context::instance().getChildren(n);
    for (auto &c : children) {
      if (c.get().name == name) return c.get();
    }
    throw std::runtime_error("Child " + name + " does not exist for " + n.name);
  }

  static inline std::vector<std::reference_wrapper<anode<Variant> > > getDirty(int all = -1) {
    std::vector<std::reference_wrapper<anode<Variant> > > _d;

    for (auto &kv : context::instance()._map) {
      if (kv.second.id.isDirty() && (all == -1 || kv.second.id.getrank() == all) &&
          !(kv.first == context::instance().rootid())) {
        _d.push_back(kv.second);
      }
    }

    return _d;
  }

  static inline void replaceNodeID(identifier oid,  identifier nid) {

    context &db = context::instance();
    auto n = db._map[oid];
    for (auto &v : db.getChildren(n)) {
      v.get().setparent(nid);
    }

    n.setid(nid);
    n.getid().setDirty();
    db._map.erase(db._map.find(oid));
    db._map[n.getid()] = n;
    db._unique[n.name] = n.getid();

  }

  static inline void recursiveChildReplace(anode<Variant> &v, unsigned id = 0) {
    context &db = context::instance();
    identifier nid = identifier(identifier::all_ranks, (id != 0) ? id : db.nextIdentifier());
    for (auto &rn : db.getChildren(v)) {
      cnode &n = rn.get();
      recursiveChildReplace(n);
    }
    replaceNodeID(v.getid(), nid);
  }

  static inline size_t countChildrenRecursively(anode<Variant> &v) {
    context &db = context::instance();
    size_t count = 0;
    for (auto &rn : db.getChildren(v)) {
      cnode n = rn.get();
      count = countChildrenRecursively(n);
    }
    return count + 1;
  }

  static inline context::children_vector gatherChildrenRecursively(anode<Variant> &v) {
    context &db = context::instance();
    context::children_vector c;
    for (auto &rn : db.getChildren(v)) {
      // c.push_back(rn);
      context::children_vector r = gatherChildrenRecursively(rn.get());
      c.insert(c.end(), r.begin(), r.end());
    }
    c.push_back(v);
    return c;
  }

  static inline void sendString(const std::string &s, const unsigned &root) {
    context &db = context::instance();
    unsigned size = s.length();
    char *str = new char[size];
    memcpy(str, s.c_str(), size);
    MPI_Bcast(&size, 1, MPI_UNSIGNED, root, db.cntx_comm.comm);
    MPI_Bcast(str, s.length(), MPI_CHAR, root, db.cntx_comm.comm);
    delete[] str;
  }

  static inline std::string receiveString(const unsigned &root) {
    context &db = context::instance();
    unsigned size = 0;
    MPI_Bcast(&size, 1, MPI_UNSIGNED, root, db.cntx_comm.comm);
    char *str = new char[size];
    MPI_Bcast(str, size, MPI_CHAR, root, db.cntx_comm.comm);
    std::string s(str, size);
    delete[] str;
    return s;
  }

  static inline void garantyUnique() {
    context &db = context::instance();

    if (db.cntx_comm.size <= 1) return;

    MPI_Barrier(db.cntx_comm.comm);

    unsigned long *unique_counter = new unsigned long[db.cntx_comm.size];
    std::set<identifier> _unique_first;
    std::set<std::string> _unique;
    std::set<std::string> visited;
    for (auto &v : db._unique) {
      auto &n = v.second;
      if (n.isDirty() && !v.second.isGlobal()) {
        _unique_first.insert(v.second);
        _unique.insert(v.first);
      }
    }

    MPI_Barrier(db.cntx_comm.comm);

    unsigned long s = _unique_first.size();
    MPI_Allgather(&s, 1, MPI_UNSIGNED_LONG, unique_counter, 1, MPI_UNSIGNED_LONG, db.cntx_comm.comm);

    for (int i = 0; i < db.cntx_comm.size; ++i) {
      if (db.cntx_comm.rank == i) {
        for (auto &s : _unique) {
          unsigned id = db.getUniqueIdentifier();
          sendString(s, i);
          if (visited.find(s) == visited.end()) {
            recursiveChildReplace(db._map[db._unique[s]], id);
          } else {
          }


          db.getUniqueIdentifier();
          visited.insert(s);
        }
      } else {
        for (int uu = 0; uu < unique_counter[i]; uu++) {
          unsigned id = db.getUniqueIdentifier();
          std::string s = receiveString(i);
          if (db._unique.find(s) != db._unique.end() && visited.find(s) == visited.end()) {
            recursiveChildReplace(db._map[db._unique[s]], id);
          } else {
          }
          db.getUniqueIdentifier();
          visited.insert(s);
        }
      }
      MPI_Barrier(db.cntx_comm.comm);
    }
    return;
  }

  static inline void sync() {
    garantyUnique();
    blindsync();
  }

  static inline void blindsync() {

    context &db = context::instance();

    if (db.cntx_comm.size <= 1) return;

    context::children_vector d = getDirty(-1);
    unsigned long *counter = new unsigned long[db.cntx_comm.size];
    counter[db.cntx_comm.rank] = d.size();

    unsigned long s = d.size();

    MPI_Allgather(&s, 1, MPI_UNSIGNED_LONG, counter, 1, MPI_UNSIGNED_LONG, db.cntx_comm.comm);

    for (int i = 0; i < db.cntx_comm.size; ++i) {

      unsigned char *buf;

      if (db.cntx_comm.rank == i && d.size() > 0) {

        for (auto rn : d) {

          auto& n = rn.get();
          n.id.resetDirty();
          cntx::mpi::encode enc;
          n.pack(enc);
          size_t s = enc.size();
          MPI_Bcast(&s, 1, MPI_UNSIGNED_LONG, i, db.cntx_comm.comm);
          MPI_Bcast(enc.getBuffer(), enc.size(), MPI_BYTE, i, db.cntx_comm.comm);
          MPI_Barrier(db.cntx_comm.comm);
        }

      } else {

        for (int nn = 0; nn < counter[i]; ++nn) {
          size_t s;
          MPI_Bcast(&s, 1, MPI_UNSIGNED_LONG, i, db.cntx_comm.comm);
          std::shared_ptr<cntx::mpi::decode::BYTE> buff((cntx::mpi::decode::BYTE *)malloc(s), free);
          cntx::mpi::decode dec(buff, s);
          MPI_Bcast(dec.buffer.get(), s, MPI_BYTE, i, db.cntx_comm.comm);
          anode<Variant> n;
          n.unpack(dec);

          n.id.resetDirty();

          if (db._map.find(n.getid()) != db._map.end()) {

             if(n.name == std::string("Locations")) {
              std::shared_ptr<std::vector<int>> vec = n;
              std::vector<int>& cv = *vec;
              std::shared_ptr<std::vector<int>> lvec = db._map[n.getid()];
              std::vector<int>& lcv = *lvec;

              cv.insert(cv.end(),lcv.begin(), lcv.end());
              std::sort(cv.begin(),cv.end());
              cv.erase(std::unique(cv.begin(),cv.end()),cv.end());

              if(db.cntx_comm.rank > i) n.id.setDirty();
            } else if (n.name == std::string("ptr")) {
              n.v = db._map[n.getid()].v;
            }
          }
          db._map[n.getid()] = n;
          if (n.unique) db._unique[n.name] = n.getid();
          MPI_Barrier(db.cntx_comm.comm);
        }
      }
    }
  }

  anode<Variant> &create_children(anode<Variant> const &n, const std::string &type) {
    return (*reinterpret_cast<Derived *>(this)).create_children(n, type);
  }
};

template <typename V, typename D> cntx::context<V, D> *cntx::context<V, D>::_singleton = nullptr;
} // namespace cntx

#endif
