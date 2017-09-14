//
// Created by jbarbosa on 9/7/17.
//

#ifndef CONTEXT_DECODER_H
#define CONTEXT_DECODER_H

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace cntx {
namespace mpi {

#define unpack_basic(BASIC)                                                    \
  template <> inline BASIC decode::unpack<BASIC>() {                           \
    BASIC &buf = offset<BASIC>(current_buffer_offset);                         \
    current_buffer_offset += sizeof(BASIC);                                    \
    return buf;                                                                \
  }

#define unpack_function_signature(BASIC)                                       \
  template <> inline BASIC decode::unpack<BASIC>()

struct decode {

  typedef unsigned char BYTE;

  std::shared_ptr<BYTE> buffer;
  std::size_t current_buffer_size;
  std::size_t current_buffer_offset;

  decode(const std::shared_ptr<BYTE> &buffer, const std::size_t size)
      : buffer(buffer), current_buffer_size(size), current_buffer_offset(0) {}

  template <typename T> inline T unpack() {
    throw std::runtime_error("Buffer unpacking of " +
                             std::string(typeid(T).name()) + " node defined. ");
  };

  std::size_t remaining() const {
    return current_buffer_size - current_buffer_offset;
  }

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
};

unpack_basic(int);
unpack_basic(unsigned int);
unpack_basic(char);
unpack_basic(unsigned char);
unpack_basic(bool);
unpack_basic(float);
unpack_basic(double);
unpack_basic(long);
unpack_basic(long long);
unpack_basic(unsigned long);
unpack_basic(unsigned long long);

unpack_function_signature(std::string) {
  std::size_t size = unpack<std::size_t>();   \
  char *str = offset<char *>(current_buffer_offset);
  current_buffer_offset += size;
  return std::string(str, size);
}

} // namespace cntx
} // namespace cntx

#endif // CONTEXT_DECODER_H
