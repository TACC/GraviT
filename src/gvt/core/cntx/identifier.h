//
// Created by Joao Barbosa on 9/12/17.
//

#ifndef CONTEXT_IDENTIFIER_H
#define CONTEXT_IDENTIFIER_H

#include "mpi/decoder.h"
#include "mpi/encoder.h"
#include <climits>

namespace cntx {
namespace details {

template <typename R> static constexpr R bitmask(unsigned int const onecount) {
  return static_cast<R>(-(onecount != 0)) & (static_cast<R>(-1) >> ((sizeof(R) * CHAR_BIT) - onecount));
}

struct identifier {

  static const unsigned int shift = sizeof(std::size_t) * 4;
  static const std::size_t identifier_filter = bitmask<std::size_t>((sizeof(size_t) * CHAR_BIT) >> 1);
  static const std::size_t dirty_bit = (1ul << 63);
  static const std::size_t id_filter = ~dirty_bit;
  static const std::size_t rank_filter = (~identifier_filter) ^ dirty_bit;
  static const std::size_t all_ranks = rank_filter;
  static const std::size_t invalid_id = bitmask<std::size_t>(sizeof(size_t) * CHAR_BIT);

  size_t id;

  identifier(const std::size_t r = rank_filter, const std::size_t i = identifier_filter) {
    setrank(r);
    setlid(i);
    // id = (r << shift) | i;
  }

  void setrank(const std::size_t rank) {
    std::size_t v = rank;

    if (rank != all_ranks) v = (rank << shift);
    id = ((id & identifier_filter) | v);
  }

  void setlid(const std::size_t aa) { id = aa | (id & rank_filter); }

  std::size_t getrank() const { return (id & id_filter & rank_filter) >> 32; }

  std::size_t getlid() const { return (id & identifier_filter); }

  std::size_t getid() const { return (id & id_filter); }

  std::size_t getIDENTIFIER() const { return id; }

  void setDirty() { id |= dirty_bit; }
  void resetDirty() { id &= id_filter; }

  bool isDirty() { return (id & dirty_bit); }

  static std::size_t allranks() { return all_ranks >> 32; }

  bool isInvalid() { return (id == invalid_id); }

  int operator<(const identifier &other) const { return getid() < other.getid(); }

  bool operator==(const identifier &other) const { return getid() == other.getid(); }

  friend std::ostream &operator<<(std::ostream &os, const identifier &other) {

    return (os << std::hex << "{ " << other.getid() << " }" << std::dec);
  }

  void pack(cntx::mpi::encode &enc) const { enc.pack<size_t>(id); }

  void unpack(cntx::mpi::decode &dec) { id = dec.unpack<size_t>(); }
};
} // namespace details
namespace mpi {
pack_function_signature(details::identifier) { v.pack(*this); }

unpack_function_signature(details::identifier) {
  details::identifier id;
  id.unpack(*this);
  return id;
}
} // namespace mpi

} // namespace cntx

#endif // CONTEXT_IDENTIFIER_H
