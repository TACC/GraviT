#ifndef COMPOSITE_H
#define COMPOSITE_H

#include <IceT.h>
#include <glm/glm.hpp>
#include <mpi.h>
namespace gvt {
namespace render {
namespace composite {

struct composite {

  IceTInt num_proc;

  composite() {}

  ~composite() {}

  bool initIceT();

  glm::vec4 *execute(glm::vec4 *buffer_in, const size_t width, const size_t height);
};
}
}
}
#endif /* COMPOSITE_H */
