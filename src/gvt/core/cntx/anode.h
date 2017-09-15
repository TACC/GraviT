//
// Created by Joao Barbosa on 9/12/17.
//

#ifndef CONTEXT_DBNODE_H_H
#define CONTEXT_DBNODE_H_H

#include "identifier.h"
#include "mpi/decoder.h"
#include "mpi/encoder.h"
#include "variant.h"

namespace cntx {

typedef details::identifier identifier;

template <typename Variant = cntx::details::variant<bool> > struct anode {

  static anode error_node;
  identifier id;
  std::string name;
  std::string type;
  Variant v;
  identifier parent;

  bool unique;

  anode() : unique(false) {}

  anode(const identifier &id, const std::string name = "", const Variant v = Variant(false),
        const identifier parent = identifier(), bool unique = false)
      : id(id), name(name), v(v), parent(parent), unique(unique) {
    this->id.setDirty();
  }

  template <typename T>
  anode(const identifier &id, const std::string name = "", const T v = T(), const identifier parent = identifier(),
        bool unique = false)
      : id(id), name(name), v(v), parent(parent), unique(unique) {
    this->id.setDirty();
  }

  ~anode() {}

  anode(const anode &n) : id(n.id), name(n.name), v(n.v), parent(n.parent), unique(n.unique) { this->id.setDirty(); }

  anode &operator=(anode const &other) {
    this->id = other.id;
    this->id.setDirty();
    name = other.name;
    v = other.v;
    parent = other.parent;
    unique = other.unique;
    return *this;
  }

  template <typename T> anode &operator=(T const &value) {
    this->id.setDirty();
    v = value;
    return *this;
  }

  template <typename T> operator T() { return v.template to<T>(); }

  template <typename T> T to() { return v.template to<T>(); }

  template <typename T> T to() const { return v.template to<T>(); }

  void setid(identifier const &id) { this->id = id; }

  identifier &getid() { return this->id; }

  identifier &getparent() { return this->parent; }

  identifier getid() const { return this->id; }

  identifier getparent() const { return this->parent; }

  bool hasParent() { return !parent.isInvalid(); }

  std::string getname() const { return name; }

  friend std::ostream &operator<<(std::ostream &os, const anode &other) {

    return (os << other.id << " " << other.getparent() << "[ " << other.name << " ] = " << other.v);
  }

  void pack(cntx::mpi::encode &enc) {
    id.pack(enc);
    parent.pack(enc);
    enc.pack<std::string>(name);
    v.pack(enc);
    enc.pack<bool>(unique);
  }

  void unpack(cntx::mpi::decode &dec) {
    id.unpack(dec);
    parent.unpack(dec);
    name = dec.unpack<std::string>();
    v.unpack(dec);
    unique = dec.unpack<bool>();
  }
};

  template <typename V> cntx::anode<V> cntx::anode<V>::error_node = cntx::anode<V>();

} // namespace cntx
#endif // CONTEXT_DBNODE_H_H
