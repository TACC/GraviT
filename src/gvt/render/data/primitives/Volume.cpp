#include <gvt/render/data/primitives/Volume.h>


gvt::render::data::primitives::Volume::Volume() {
  }

gvt::render::data::primitives::Volume::GetSamples() { return floatsamples; }
gvt::render::data::primitives::Volume::GetDeltas(glm::vec3 &del) { del = deltas; }
gvt::render::data::primitives::Volume::GetGlobalOrigin(glm::vec3 &orig) { orig = origin; }
gvt::render::data::primitives::Volume::GetGlobalOrigin(glm::vec3 &lorig) { 
  lorig.x = origin.x + ; }
gvt::render::data::primitives::Volume::~Volume() {}
gvt::render::data::primitives::Box3D *gvt::render::data::primitives::Volume::getBoundingBox() {return &boundingBox;}
