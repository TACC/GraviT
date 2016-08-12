#include <gvt/render/data/primitives/Volume.h>


gvt::render::data::primitives::Volume::Volume() {
  n_slices = 0;
  counts = {0,0,0};
  n_isovalues = 0;
  slices = NULL;
  counts = {1,1,1};
  origin = {0.0,0.0,0.0};
  spacing = {1.0,1.0,1.0};
  }

void gvt::render::data::primitives::Volume::GetDeltas(glm::vec3 &del) { del = deltas; }
void gvt::render::data::primitives::Volume::GetGlobalOrigin(glm::vec3 &orig) { orig = origin; }
gvt::render::data::primitives::Volume::~Volume() {}
gvt::render::data::primitives::Box3D *gvt::render::data::primitives::Volume::getBoundingBox() {return &boundingBox;}
