#ifndef GVT_CONTEXT_VARIANT_H
#define GVT_CONTEXT_VARIANT_H

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <glm/glm.hpp>

#include "mpi/decoder.h"
#include "mpi/encoder.h"

#include "utils.h"

namespace cntx {

namespace details {

inline std::ostream &operator<<(std::ostream &os, const std::nullptr_t &ptr) { return os << " NULL pointer"; }

template <typename... Ts> struct variant {

  template <typename U, typename = typename std::enable_if<check<U, Ts...>::value>::type> variant(const U &value) {
    safe_alloc(value);
    tid = tindex<U, Ts...>::value;
  }

  variant(const variant &other) {
    safe_copy<Ts...>(other);
    tid = other.tid;
  }

  variant(variant &&other) {
    safe_move<Ts...>(std::move(other));
    tid = other.tid;
  }
  variant &operator=(const variant &other) {
    safe_copy<Ts...>(other);
    tid = other.tid;
    return *this;
  }

  std::ostream &print(std::ostream &os) const { return _print<Ts...>(os); }

  template <typename U, typename = typename std::enable_if<check<U, Ts...>::value>::type>
  variant &operator=(const U &value) {
    safe_alloc(value);

    tid = tindex<U, Ts...>::value;


    return *this;
  }

  ~variant() { safe_dealloc<Ts...>(); }

  template <typename U, typename = typename std::enable_if<check<U, Ts...>::value>::type> U to() {

    if (!is<U>())
      throw std::runtime_error("Error: incorrect data type request : " + std::to_string(tid) +
                               " :: " + std::string(typeid(U).name()));
    return *reinterpret_cast<U *>(_storage);
  }

  template <typename U, typename = typename std::enable_if<check<U, Ts...>::value>::type> U to() const {
    if (!is<U>()) throw std::runtime_error("Error: incorrect data type request");
    return *reinterpret_cast<const U *>(_storage);
  }

  template <typename U> bool valid() { return check<U, Ts...>::value; }

  template <typename U> bool is(void) const { return tid == tindex<U, Ts...>::value; }

  bool isPointer() { return _isPointer<Ts...>(); }

  variant() {}

  void pack(cntx::mpi::encode &enc) {
    enc.pack<size_t>(tid);
    _pack<Ts...>(enc);
  }

  void unpack(cntx::mpi::decode &dec) {
    tid = dec.unpack<size_t>();
    _unpack<Ts...>(dec);
  }

private:
  // Variables

  template <typename U, typename... Us> U *convert() {
    if (tindex<U, Ts...>::value == tid) {

      return reinterpret_cast<U *>(_storage);
    }
    return convert<Us...>();
  };

  template <size_t N> struct TypeNumber {
    enum { value = N };
  };

  template <typename U> void safe_dealloc(TypeNumber<0> = TypeNumber<0>()) {
    if (is<U>()) to<U>().~U();
  }

  template <typename U, typename... Us> void safe_dealloc(TypeNumber<sizeof...(Us)> = TypeNumber<sizeof...(Us)>()) {

    if (is<U>() && !std::is_same<U, std::string>::value) {

      if (!is_shared_pointer<U>::value) {
        to<U>().~U();
      }
    }
    safe_dealloc<Us...>(TypeNumber<sizeof...(Us) - 1>());
  }

  template <typename U> void safe_copy(const variant &other, TypeNumber<0> = TypeNumber<0>()) {
    new (_storage) U(other.to<U>());
  }

  template <typename U, typename... Us>
  void safe_copy(const variant &other, TypeNumber<sizeof...(Us)> = TypeNumber<sizeof...(Us)>()) {
    if (!other.is<U>()) return safe_copy<Us...>(other, TypeNumber<sizeof...(Us) - 1>());
    new (_storage) U(other.to<U>());
  }

  template <typename U> void safe_move(variant &&other, TypeNumber<0> = TypeNumber<0>()) {
    new (_storage) U(std::move(other.to<U>()));
  }

  template <typename U, typename... Us>
  void safe_move(variant &&other, TypeNumber<sizeof...(Us)> = TypeNumber<sizeof...(Us)>()) {
    if (!other.is<U>()) return safe_move<Us...>(std::move(other), TypeNumber<sizeof...(Us) - 1>());
    new (_storage) U(std::move(other.to<U>()));
  }

  template <typename T> T *safe_alloc(const T &value) {

    if (tid != invalid_type) {
      safe_dealloc<Ts...>();
    }

    return new (_storage) T(value);
  };

  template <typename U> std::ostream &_print(std::ostream &os, TypeNumber<0> = TypeNumber<0>()) const {
    return os << to<U>();
  }

  template <typename U, typename... Us>
  std::ostream &_print(std::ostream &os, TypeNumber<sizeof...(Us)> = TypeNumber<sizeof...(Us)>()) const {

    if (not is<U>()) return _print<Us...>(os, TypeNumber<sizeof...(Us) - 1>());

    return os << to<U>();
  }

  template <typename U> bool _isPointer(TypeNumber<0> = TypeNumber<0>()) { return is_shared_pointer<U>::value; }

  template <typename U, typename... Us> bool _isPointer(TypeNumber<sizeof...(Us)> = TypeNumber<sizeof...(Us)>()) {
    if (is<U>()) {
      return is_shared_pointer<U>::value;
    } else {
      return _isPointer<Us...>(TypeNumber<sizeof...(Us) - 1>());
    }
  }

  template <typename U> void _pack(cntx::mpi::encode &enc, TypeNumber<0> = TypeNumber<0>()) { enc.pack<U>(to<U>()); }

  template <typename U, typename... Us>
  void _pack(cntx::mpi::encode &enc, TypeNumber<sizeof...(Us)> = TypeNumber<sizeof...(Us)>()) {
    if (is<U>()) {
      enc.pack<U>(to<U>());
    } else {
      _pack<Us...>(enc, TypeNumber<sizeof...(Us) - 1>());
    }
  }

  template <typename U> void _unpack(cntx::mpi::decode &dec, TypeNumber<0> = TypeNumber<0>()) {
    if (is<U>()) (*this) = dec.unpack<U>();
  }

  template <typename U, typename... Us>
  void _unpack(cntx::mpi::decode &dec, TypeNumber<sizeof...(Us)> TPS = TypeNumber<sizeof...(Us)>()) {

    if (is<std::vector<int> >()) {

      std::cout << "Ok Vector detected" << std::endl;
      dec.unpack<U>();
      std::cout << "Error above" << std::endl;
      return;
    }

    if (is<U>()) {
      (*this) = dec.unpack<U>();
    } else {
      _unpack<Us...>(dec, TypeNumber<sizeof...(Us) - 1>());
    }
  }

public:
  const std::size_t invalid_type = tindex<void, Ts...>::value;
  size_t tid = invalid_type;
  using max_type = typename variant_max_type<Ts...>::type;
  static constexpr size_t max_size = variant_max_size<Ts...>::value;
  typename std::aligned_storage<sizeof(max_type), alignof(max_type)>::type _storage[max_size];

public:
  template <typename... Us> friend std::ostream &operator<<(std::ostream &os, const variant<Us...> &other) {
    ;
    return other.print(os);
  }
};
}; // namespace details

} // namespace cntx
#endif
