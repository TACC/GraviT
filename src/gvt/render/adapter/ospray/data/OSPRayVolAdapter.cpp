#include "gvt/render/adapter/ospray/data/OSPRayVolAdapter.h"

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/primitives/Volume.h>
#include "gvt/core/CoreContext.h"

gvt::render::adapter::ospray::data::OSPRayVolAdapter::OSPRayVolAdapter(int *argc, char *argv[], gvt::render::data::primitives::Volume *vol): gvt::render::adapter::ospray::data::OSPRayAdapter(argc,argv) {
  
}
gvt::render::adapter::ospray::data::OSPRayVolAdapter::OSPRayVolAdapter(gvt::render::data::primitives::Volume *vol) {

}
