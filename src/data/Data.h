#ifndef GVT2_DATA_H
#define GVT2_DATA_H


#include <memory>

#include "BBox.h"

namespace gvt2 {

/// base class for mesh
// class AbstractMesh {
class Data {
public:

  Data() {}
  ~Data() {}

  virtual std::shared_ptr<Data> getData() { return nullptr;};
  virtual gvt2::Box3D* getBoundingBox() {return nullptr;};

};
} // namespace gvt2

#endif //GRAVIT_DATA_H
