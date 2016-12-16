#ifndef GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPRAY_ADAPTER_H
#define GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPRAY_ADAPTER_H

#include <iostream>
#include "gvt/render/Adapter.h"
#include <gvt/render/RenderContext.h>
#include "ospray/ospray.h"
//#include "ospray/ExternalAPI.h"
#include "ospray/OSPExternalRays.h"

namespace gvt {
namespace render {
namespace adapter {
namespace ospray {
namespace data {

class OSPRayAdapter : public gvt::render::Adapter {
public:
  /** 
   * Construct the OSPRayAdapter. 
   */
  OSPRayAdapter(gvt::render::data::primitives::Data*);
  OSPRayAdapter(gvt::render::data::primitives::Mesh*);
  OSPRayAdapter(gvt::render::data::primitives::Volume*);
  OSPVolume GetTheOSPVolume() {return theOSPVolume;}
  OSPModel GetTheOSPModel() {return theOSPModel;}
  ~OSPRayAdapter();
  static void initospray(int *argc,char **argv) ;
  OSPExternalRays GVT2OSPRays(gvt::render::actor::RayVector &rayList);
  void OSP2GVTMoved_Rays(OSPExternalRays &out, OSPExternalRays &rl, gvt::render::actor::RayVector &moved_rays);
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays, glm::mat4 *m,
            glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
                  size_t begin = 0, size_t end = 0);

protected:
  /**
   * Variable: adapter has been initialized if true. 
   */
  static bool init;
  OSPData theOSPData;
  size_t begin, end;
  int width, height;
public:
  OSPRenderer theOSPRenderer;
  OSPVolume theOSPVolume;
  OSPModel theOSPModel;
};
}
}
}
}
}
#endif // GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPRAY_ADAPTER_H
