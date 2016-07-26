#include <gvt/render/data/primitives/Volume.h>

gvt::render::data::primitives::Volume::Volume() {}
gvt::render::data::primitives::Volume::~Volume() {}
gvt::render::data::primitives::Box3D *gvt::render::data::primitives::Volume::getBoundingBox() {return &boundingBox;}
