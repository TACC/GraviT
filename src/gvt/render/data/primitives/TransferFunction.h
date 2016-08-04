#ifndef GVT_RENDER_DATA_PRIMITIVES_TRANSFERFUNCTION_H
#define GVT_RENDER_DATA_PRIMITIVES_TRANSFERFUNCTION_H


#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "ospray/ospray.h"

namespace gvt {
namespace render {
namespace data {
namespace primitives {

class TransferFunction {
public:
  TransferFunction();
  ~TransferFunction();
  void load(std::string cname, std::string oname);
  OSPTransferFunction GetTheOSPTransferFunction() {return theOSPTransferFunction; }
  bool DeviceCommit();
protected:
  glm::vec4 *colormap;
  glm::vec2 *opacitymap;
  int n_colors, n_opacities;
  OSPTransferFunction theOSPTransferFunction;
};
}
}
}
}
#endif
