//
// Created by Joao Barbosa on 9/11/17.
//

#ifndef CONTEXT_UTILS_H
#define CONTEXT_UTILS_H

#include <vector>

namespace cntx {

namespace details {

template <typename... Ts> struct variant_max_type;

template <typename T> struct variant_max_type<T> { using type = T; };

template <typename T, typename U, typename... Ts> struct variant_max_type<T, U, Ts...> {
  using type = typename variant_max_type<typename std::conditional<(sizeof(U) <= sizeof(T)), T, U>::type, Ts...>::type;
};

template <typename... Ts> struct variant_max_size {
  enum { value = sizeof(typename variant_max_type<Ts...>::type) };
};

template <typename C, typename... Ts> struct check;

template <typename C, typename T> struct check<C, T> {

  enum { value = std::is_same<C, T>::value };
};

template <typename C, typename T, typename... Ts> struct check<C, T, Ts...> {
  enum { value = std::is_same<C, T>::value || check<C, Ts...>::value };
};

template <typename C, typename... Ts> struct tindex;

template <typename U, typename T> struct tindex<U, T> {
  enum { value = std::is_same<U, T>::value ? 0 : -1 };
};

template <typename U, typename T, typename... Ts> struct tindex<U, T, Ts...> {
  enum { value = std::is_same<U, T>::value ? sizeof...(Ts) : tindex<U, Ts...>::value };
};

template <typename T> struct is_shared_pointer {
  enum { value = false };
};

template <typename T> struct is_shared_pointer<std::shared_ptr<T> > {
  enum { value = true };
};

  inline std::ostream &operator<<(std::ostream &os, const std::shared_ptr<std::vector<int>> &other) {

    std::vector<int>& vec = *other.get();
    os << " { ";
    for(auto& v : vec) os << v << " ";
    os << " } ";
    return os;
  }

} // namespace details



} // namespace cntx

#endif // CONTEXT_UTILS_H
