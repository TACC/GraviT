#ifndef GVT_RENDER_DATA_PRIMITIVES_VOLUME_H
#define GVT_RENDER_DATA_PRIMITIVES_VOLUME_H

#include <gvt/render/data/primitives/BBox.h>
#include <gvt/render/data/primitives/Material.h>
#include <gvt/render/data/scene/Light.h>

#include <vector>
namespace gvt {
namespace render {
namespace data {
namespace primitives {

class Volume {
public:
  Volume();
  ~Volume();
  Box3D boundingBox;
  Box3D *getBoundingBox();
  enum DataType
  {
    FLOAT, UCHAR
  };
  unsigned char *get_samples() ;
  float *get_samples();
protected:
  glm::vec4 *slices;
  glm::vec3 counts;
  glm::vec3 origin;
  glm::vec3 spacing;
  int n_slices;
  float *isovalues;
  int n_isovalues;
private:
  DataType type;
  glm::vec3 deltas;
  unsigned char *uchar_samples;
  float *floatsamples;
};
}
}
}
}
#endif
