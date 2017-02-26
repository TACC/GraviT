#include <gvt/core/comm/communicator.h>

#include <cassert>
#include <iostream>
#include <mpi.h>

namespace gvt {
namespace comm {
std::shared_ptr<communicator> communicator::_instance = nullptr;
tbb::task_group communicator::tg;

std::vector<std::string> communicator::registry_names;
std::map<std::string, std::size_t> communicator::registry_ids;

bool communicator::_MPI_THREAD_SERIALIZED = true;

communicator::communicator() {
  // _id = MPI::COMM_WORLD.Get_rank();
  // _size = MPI::COMM_WORLD.Get_size();
  MPI_Comm_rank(MPI_COMM_WORLD, &_id);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
}
communicator::~communicator() {}

void communicator::init(int argc, char *argv[], bool start_thread) {
  assert(communicator::_instance);
  if (start_thread) tg.run([&]() { communicator::_instance->run(); });
  RegisterMessageType<gvt::comm::Message>();
}

std::shared_ptr<communicator> communicator::singleton() {
  assert(communicator::_instance);
  return communicator::_instance;
}

communicator &communicator::instance() {
  assert(communicator::_instance);
  return *communicator::_instance.get();
}

std::size_t communicator::id() {
  assert(communicator::_instance);
  return _id;
}

std::size_t communicator::lastid() {
  assert(communicator::_instance);
  return _size;
}

void communicator::terminate() {
  _terminate = true;
  tg.wait();
  MPI_Finalize();
}

inline void communicator::aquireComm() {
  if (communicator::_MPI_THREAD_SERIALIZED) _mcomm.lock();
}

inline void communicator::releaseComm() {
  if (communicator::_MPI_THREAD_SERIALIZED) _mcomm.unlock();
}

void communicator::send(std::shared_ptr<comm::Message> msg, std::size_t to) {
  assert(msg->tag() >= 0 && msg->tag() < registry_names.size());
  const std::string classname = registry_names[msg->tag()];
  assert(registry_ids.find(classname) != registry_ids.end());
  msg->src(id());
  msg->dst(to);

  //  std::cout << "Send : " << msg->buffer_size() << " on " << id() << " to " << to
  //            << std::flush << std::endl;
  aquireComm();
  MPI_Send(msg->getMessage<void>(), msg->buffer_size(), MPI_BYTE, to, CONTROL_SYSTEM_TAG, MPI_COMM_WORLD);
  releaseComm();
};
void communicator::broadcast(std::shared_ptr<comm::Message> msg) {
  assert(msg->tag() >= 0 && msg->tag() < registry_names.size());
  const std::string classname = registry_names[msg->tag()];
  assert(registry_ids.find(classname) != registry_ids.end());
  msg->src(id());
  for (int i = 0; i < lastid(); i++) {
    if (i == id()) continue;

    std::shared_ptr<gvt::comm::Message> new_msg = std::make_shared<gvt::comm::Message>(*msg);

    new_msg->dst(i);

    aquireComm();
    MPI_Send(msg->getMessage<void>(), msg->buffer_size(), MPI_BYTE, i, CONTROL_SYSTEM_TAG, MPI_COMM_WORLD);
    releaseComm();
  }
};
}
}
