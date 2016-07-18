#ifndef GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPRAY_ADAPTER_H
#define GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPRAY_ADAPTER_H

#include <iostream>
#include "ospray/ospray.h"
#include "ospray/ExternalAPI.h"

namespace gvt {
namespace render {
namespace adapter {
namespace ospray {
namespace data {

class OSPRayAdapter {
public:
  /** 
   * Construct the OSPRayAdapter base class 
   */
  OSPRayAdapter(int *argc, char **argv[]);

protected:
  /**
   * Variable: adapter has been initialized if true. 
   */
  static bool init;
private:
  OSPRenderer theOSPRenderer;
};
}
}
}
}
}
#endif // GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPRAY_ADAPTER_H
