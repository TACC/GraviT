#include <gvt/core/comm/communicator.h>
#include <gvt/core/comm/message.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>

REGISTER_INIT_MESSAGE(gvt::comm::Message)

namespace gvt {

namespace comm {

Message::Message(const std::size_t &s) {
  _buffer_size = s + sizeof(header);
  content = static_cast<Byte *>(std::malloc(_buffer_size));
  tag(COMMUNICATOR_MESSAGE_TAG);
  system_tag(CONTROL_USER_TAG);
  size(s);
}

Message::Message(const Message &msg) {
  if (_buffer_size > 0) {
    content = static_cast<Byte *>(std::realloc(content, msg.buffer_size()));
  } else {
    content = static_cast<Byte *>(std::malloc(msg.buffer_size()));
  }
  _buffer_size = msg.buffer_size();
  std::memcpy(content, msg.content, msg.buffer_size());
}

Message::Message(Message &&msg) {
  std::swap(content, msg.content);
  std::swap(_buffer_size, msg._buffer_size);
}

Message::~Message() { std::free(content); }

Message::header &Message::getHeader() {
  return *reinterpret_cast<header *>(content + (_buffer_size - sizeof(header)));
}
std::size_t Message::tag() { return getHeader().USER_TAG; };
void Message::tag(const std::size_t tag) { getHeader().USER_TAG = tag; };

std::size_t Message::size() { return getHeader().USER_MSG_SIZE; };
void Message::size(const std::size_t size) { getHeader().USER_MSG_SIZE = size; };

std::size_t Message::system_tag() { return getHeader().SYSTEM_TAG; };
void Message::system_tag(const std::size_t tag) { getHeader().SYSTEM_TAG = tag; };
}
}
