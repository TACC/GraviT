#include "gvt/render/adapter/ospray/data/OSPRayMeshAdapter.h"

#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/primitives/Mesh.h>
#include "gvt/core/CoreContext.h"

gvt::render::adapter::ospray::data::OSPRayMeshAdapter::OSPRayMeshAdapter(int *argc, char *argv[], gvt::render::data::primitives::Mesh *mesh):gvt::render::adapter::ospray::data::OSPRayAdapter(argc,argv) {
}
