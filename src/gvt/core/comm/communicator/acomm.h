#ifndef MPICOMMLIB_ACOMM
#define MPICOMMLIB_ACOMM

#include <gvt/core/comm/communicator.h>

namespace gvt {
namespace comm {
struct acomm : public communicator {
  acomm();
  static void init(int argc = 0, char *argv[] = nullptr, bool start_thread = true);
  virtual void send(std::shared_ptr<comm::Message> msg, std::size_t to);
  virtual void broadcast(std::shared_ptr<comm::Message> msg);
  virtual void run();

  std::vector<std::shared_ptr<Message> > _outbox;
  std::mutex moutbox;
};
}
}

#endif /*MPICOMMLIB*/
