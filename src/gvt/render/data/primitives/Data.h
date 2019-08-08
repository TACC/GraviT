//
// Created by Joao Barbosa on 11/7/17.
//

#ifndef GVT_RENDER_DATA_PRIMITIVES_DATA_H
#define GVT_RENDER_DATA_PRIMITIVES_DATA_H


#include <memory>

#include <gvt/render/data/primitives/BBox.h>

namespace gvt {
namespace render {
namespace data {
namespace primitives {

/// base class for mesh
// class AbstractMesh {
class Data {
public:

  Data() {}
  ~Data() {}

  virtual std::shared_ptr<Data> getData() { return nullptr;};
  virtual gvt::render::data::primitives::Box3D* getBoundingBox() {return nullptr;};

};
} // namespace primitives
} // namespace data
} // namespace render
} // namespace gvt

#endif //GRAVIT_DATA_H
