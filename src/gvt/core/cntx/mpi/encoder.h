//
// Created by jbarbosa on 9/7/17.
//

#ifndef CONTEXT_ENCDEC_H
#define CONTEXT_ENCDEC_H

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace cntx {
namespace mpi {

typedef unsigned char BYTE;

#define pack_basic(BASIC)                                                      \
  template <> inline void encode::pack<BASIC>(const BASIC &v) {                \
    increase(sizeof(BASIC));                                                   \
    BASIC &buf = offset<BASIC>(current_buffer_offset);                         \
    buf = v;                                                                   \
    current_buffer_offset += sizeof(BASIC);                                    \
  }

#define pack_function_signature(BASIC)                                         \
  template <> inline void encode::pack<BASIC>(const BASIC &v)

struct encode {

  std::shared_ptr<BYTE> buffer;
  std::size_t current_buffer_size;
  std::size_t current_buffer_offset;

  encode()
      : buffer(nullptr), current_buffer_size(0), current_buffer_offset(0) {}

  template <typename T> inline void pack(const T &v) {
    throw std::runtime_error("Buffer packing of " +
                             std::string(typeid(T).name()) + " node defined. ");
  };

  template <typename T> inline void pack(const T* v, const size_t& size) {

    if( size % sizeof(T) !=0) {
      throw std::runtime_error("Encode size does not match");
    }

    increase(size);

    T* dst = offset<T*>(size);
    std::memcpy(dst,v,size);

//    throw std::runtime_error("Buffer packing of " +
//                             std::string(typeid(T).name()) + " node defined. ");
  };


  BYTE *getBuffer() const { return buffer.get(); }

  std::size_t size() { return current_buffer_size; }

private:
  template <typename T,
            typename = typename std::enable_if<std::is_pointer<T>::value>::type>
  inline T offset(const size_t &off) const {
    return reinterpret_cast<T>(buffer.get() + off);
  }

  template <typename T, typename = typename std::enable_if<
                            !std::is_pointer<T>::value>::type>
  inline T &offset(const size_t &off) {
    return *reinterpret_cast<T *>(buffer.get() + off);
  }

  void increase(const std::size_t &size) {

    if (buffer == nullptr) {
      buffer = std::shared_ptr<BYTE>((BYTE *)malloc(size), free);
      current_buffer_size = size;
    } else if ((current_buffer_size - current_buffer_offset) < size) {
      BYTE *tmpbuf =
          reinterpret_cast<BYTE *>(malloc(current_buffer_size + size));
      std::memcpy(tmpbuf, buffer.get(), current_buffer_offset);
      buffer = std::shared_ptr<BYTE>(tmpbuf, free);
      current_buffer_size += size;
    }
  }
};

pack_basic(int);
pack_basic(unsigned int);
pack_basic(char);
pack_basic(unsigned char);
pack_basic(bool);
pack_basic(float);
pack_basic(double);
pack_basic(long);
pack_basic(long long);
pack_basic(unsigned long);
pack_basic(unsigned long long);

pack_function_signature(std::string) {
  pack<std::size_t>(v.length());
  increase(v.length());
  char *str = offset<char *>(current_buffer_offset);
  memcpy(str, v.c_str(), v.length());
  current_buffer_offset += v.length();
};

} // namespace mpi
} // namespace cntx

#endif // CONTEXT_ENCDEC_H
