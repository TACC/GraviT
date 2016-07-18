#define TBB_PREVIEW_STATIC_PARTITIONER 1
#include "gvt/render/adapter/ospray/data/OSPRayAdapter.h"
#include "gvt/core/CoreContext.h"

using namespace gvt::render::adapter::ospray::data;

bool OSPRayAdapter::init = false;

OSPRayAdapter::OSPRayAdapter(int *argc, char *argv[]) {
  ospInit(argc,(const char**)argv);
  theOSPRenderer = ospNewRenderer("ptracer");
}
OSPRayAdapter::~OSPRayAdapter() {
  ospRelease(theOSPRenderer);
}

